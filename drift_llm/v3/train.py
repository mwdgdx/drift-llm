"""
v3 Text Drifting: unconditional text generation via embedding-space drift loss.

Three feature modes:
  direct:    Features directly from generator output (no softmax) + vocab proximity reg
  gpt2_soft: Soft embeddings → frozen GPT-2 features (captures sequential structure)
  emb_stats: Embedding statistics via soft-vocab projection (bag-of-words)

Usage:
    python train.py --mode preprocess
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --nproc_per_node=6 train.py --mode train
"""

import argparse
import logging
import math
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, GPT2Model

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v3.model import EmbeddingGenerator
from drift_loss import drift_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def cosine_lr(step, max_steps, warmup, peak_lr):
    if step < warmup:
        return peak_lr * step / max(warmup, 1)
    t = (step - warmup) / max(max_steps - warmup, 1)
    return peak_lr * 0.5 * (1 + math.cos(math.pi * t))


# ======================== Preprocessing ========================

def preprocess(args):
    """Compute GPT-2 features for dataset, then K-means cluster."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load data ----
    logger.info("Loading AG News …")
    from datasets import load_dataset
    ds = load_dataset("ag_news", split="train")
    texts = ds["text"]
    gt_labels = np.array(ds["label"])

    # ---- Tokenize ----
    logger.info("Tokenizing …")
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    all_ids, all_masks = [], []
    for i in range(0, len(texts), 1024):
        enc = tok(
            texts[i : i + 1024],
            max_length=args.seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
    token_ids = torch.cat(all_ids)
    masks = torch.cat(all_masks)
    logger.info(f"  {token_ids.shape[0]} samples, seq_len={token_ids.shape[1]}")

    # ---- GPT-2 features ----
    logger.info("Extracting GPT-2 features …")
    gpt2 = GPT2Model.from_pretrained("gpt2").to(device).eval()
    feats = []
    bs = 256
    for i in range(0, len(token_ids), bs):
        ids = token_ids[i : i + bs].to(device)
        m = masks[i : i + bs].to(device)
        with torch.no_grad():
            h = gpt2(input_ids=ids, attention_mask=m).last_hidden_state
            mf = m.unsqueeze(-1).float()
            feats.append(((h * mf).sum(1) / mf.sum(1).clamp(min=1)).cpu())
        if (i // bs) % 100 == 0:
            logger.info(f"    {i}/{len(token_ids)}")
    features = torch.cat(feats)
    logger.info(f"  Features: {features.shape}")

    # ---- K-means ----
    K = args.num_clusters
    logger.info(f"K-means (K={K}) …")
    from sklearn.cluster import MiniBatchKMeans
    km = MiniBatchKMeans(n_clusters=K, batch_size=2048, n_init=3, random_state=42)
    cluster_ids = torch.tensor(km.fit_predict(features.numpy()), dtype=torch.long)
    _, counts = np.unique(cluster_ids.numpy(), return_counts=True)
    logger.info(f"  Cluster sizes: min={counts.min()} max={counts.max()} "
                f"mean={counts.mean():.0f} median={int(np.median(counts))}")

    # ---- Save ----
    os.makedirs(args.cache_dir, exist_ok=True)
    path = os.path.join(args.cache_dir, "preprocessed.pt")
    torch.save(dict(
        token_ids=token_ids,
        masks=masks,
        features=features,
        cluster_ids=cluster_ids,
        gt_labels=torch.tensor(gt_labels),
        num_clusters=K,
    ), path)
    logger.info(f"Saved → {path}")


# ======================== Feature helpers ========================

def emb_stats_features(embeddings, n_chunks=8):
    """Embedding statistics features: mean + std + chunk means + bigram transitions.

    Captures both bag-of-words statistics AND local sequential structure
    via element-wise products of consecutive embeddings.

    Args:
        embeddings: [B, L, D]
        n_chunks: number of sequential chunks for positional features
    Returns:
        [B, (3 + n_chunks) * D] feature vector
    """
    parts = [embeddings.mean(dim=1), embeddings.std(dim=1)]
    L = embeddings.shape[1]
    chunk_size = max(1, L // n_chunks)
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, L)
        parts.append(embeddings[:, start:end].mean(dim=1))
    if L > 1:
        bigram = embeddings[:, :-1] * embeddings[:, 1:]
        parts.append(bigram.mean(dim=1))
    return torch.cat(parts, dim=-1)


def gpt2_soft_features(gpt2, logits, vocab_emb, temperature, top_k=0):
    """Soft embeddings → frozen GPT-2 → mean pool.

    top_k > 0: mask all but the top-k logits before softmax.
    This keeps entropy low (≤ ln(k)) so GPT-2 sees meaningful inputs.
    """
    if top_k > 0:
        topk_vals, _ = logits.topk(top_k, dim=-1)
        threshold = topk_vals[..., -1:]
        logits = logits.masked_fill(logits < threshold, -1e9)
    soft_emb = F.softmax(logits / temperature, dim=-1) @ vocab_emb
    h = gpt2(inputs_embeds=soft_emb).last_hidden_state
    return h.mean(dim=1)


# ======================== Training ========================

def train(args):
    # ---- DDP ----
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    # ---- Load preprocessed data ----
    data_path = os.path.join(args.cache_dir, "preprocessed.pt")
    if not os.path.exists(data_path):
        if is_main():
            preprocess(args)
        if ddp:
            dist.barrier()

    data = torch.load(data_path, map_location="cpu", weights_only=False)
    cluster_ids = data["cluster_ids"]              # [N]
    token_ids = data["token_ids"]                  # [N, L]
    num_clusters = int(data["num_clusters"])

    # per-cluster index lists
    cluster_idx = {}
    for c in range(num_clusters):
        idx = (cluster_ids == c).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            cluster_idx[c] = idx
    valid_clusters = np.array(sorted(cluster_idx.keys()))

    # ---- Models ----
    generator = EmbeddingGenerator(
        emb_dim=768, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, seq_len=args.seq_len, dropout=args.dropout,
    ).to(device)
    if ddp:
        generator = DDP(generator, device_ids=[local_rank])
    raw_gen = generator.module if ddp else generator

    use_gpt2 = (args.feature_mode == "gpt2_soft")

    if use_gpt2:
        gpt2 = GPT2Model.from_pretrained("gpt2").to(device).eval()
        for p in gpt2.parameters():
            p.requires_grad = False
        vocab_emb = gpt2.wte.weight.detach()           # [V, 768]
    else:
        _gpt2 = GPT2Model.from_pretrained("gpt2")
        vocab_emb = _gpt2.wte.weight.detach().clone().to(device)
        del _gpt2
        gpt2 = None

    n_params = sum(p.numel() for p in raw_gen.parameters()) / 1e6
    if is_main():
        logger.info(f"Generator: {n_params:.1f}M params  |  Feature mode: {args.feature_mode}")

    # ---- Memory bank features ----
    if use_gpt2:
        all_features = data["features"]  # pre-computed GPT-2 mean pool (768-dim)
        all_features_gpu = all_features.to(device)
        D_feat = all_features.shape[1]
    else:
        if is_main():
            logger.info("Computing embedding stats features for memory bank …")
        emb_bank_bs = 1024
        all_feat_parts = []
        for i in range(0, len(token_ids), emb_bank_bs):
            ids_batch = token_ids[i : i + emb_bank_bs].to(device)
            emb_batch = vocab_emb[ids_batch]
            feat_batch = emb_stats_features(emb_batch)
            all_feat_parts.append(feat_batch.cpu())
        all_features = torch.cat(all_feat_parts)
        all_features_gpu = all_features.to(device)
        D_feat = all_features.shape[1]
    if is_main():
        logger.info(f"Data: {all_features.shape[0]} samples, {len(valid_clusters)} non-empty clusters, "
                     f"feat_dim={D_feat}")

    # ---- Optim / schedule ----
    optimizer = torch.optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=0.01)
    R_list = tuple(args.R_list)

    # ---- WandB ----
    if args.wandb_project and is_main():
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"v3_d{args.d_model}_L{args.n_layers}_K{num_clusters}_G{args.G}",
            config=vars(args),
        )

    os.makedirs(args.output_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    if is_main():
        logger.info(f"Features: {args.feature_mode}  "
                     f"R_list={R_list}  C={args.cluster_batch} G={args.G} P={args.P} N={args.N}")
        logger.info(f"τ={args.temperature}  λ_div={args.lambda_diversity}  λ_reg={args.lambda_reg}  λ_intra={args.lambda_intra}")
        logger.info(f"Training for {args.max_steps} steps …")

    # ======================== Main loop ========================
    for step in range(1, args.max_steps + 1):
        generator.train()
        lr = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        C = args.cluster_batch
        G = args.G

        # 1. Sample clusters
        sampled = np.random.choice(valid_clusters, size=C, replace=False)

        # 2. Generate C*G hidden states → soft-vocabulary → embeddings
        noise = torch.randn(C * G, args.seq_len, args.d_model, device=device)
        gen_h = generator(noise)                     # [C*G, L, d_model]  (has grad)

        # 3. Features
        if args.feature_mode == "direct":
            gen_feat = emb_stats_features(gen_h)
            # Vocab proximity: push each position toward nearest token embedding (cosine)
            gen_norm = F.normalize(gen_h.reshape(-1, gen_h.shape[-1]), dim=-1)
            vocab_norm = F.normalize(vocab_emb, dim=-1)
            cos_sim = gen_norm @ vocab_norm.T          # [C*G*L, V]
            max_cos = cos_sim.max(dim=-1).values       # [C*G*L]
            reg_loss = -max_cos.mean()
        elif args.feature_mode == "gpt2_soft":
            logits = gen_h @ vocab_emb.T
            gen_feat = gpt2_soft_features(gpt2, logits, vocab_emb, args.temperature, args.top_k)
            reg_loss = torch.tensor(0.0, device=device)
        else:  # emb_stats
            logits = gen_h @ vocab_emb.T
            soft_probs = F.softmax(logits / args.temperature, dim=-1)
            gen_emb = soft_probs @ vocab_emb
            gen_feat = emb_stats_features(gen_emb)
            reg_loss = torch.tensor(0.0, device=device)
        gen_feat = gen_feat.view(C, G, -1)           # [C, G, D_feat]

        # 4. Positive / negative from pre-computed bank
        pos = torch.zeros(C, args.P, D_feat, device=device)
        neg = torch.zeros(C, args.N, D_feat, device=device)
        for i, c in enumerate(sampled):
            ci = cluster_idx[c]
            pos[i] = all_features_gpu[ci[torch.randint(len(ci), (args.P,))]]
            neg[i] = all_features_gpu[torch.randint(len(all_features), (args.N,))]

        # 5. Drift loss
        loss, info = drift_loss(gen=gen_feat, fixed_pos=pos, fixed_neg=neg, R_list=R_list)
        drift_val = loss.mean()

        # 6. Diversity loss: penalize collapse within each cluster
        feat_std = gen_feat.std(dim=1).mean()
        div_loss = -feat_std

        # 7. Intra-sequence diversity: penalize repeated tokens at every position
        pos_var = gen_h.var(dim=1).mean()  # variance across positions within each sequence
        intra_loss = -pos_var

        total_loss = (drift_val + args.lambda_diversity * div_loss
                      + args.lambda_reg * reg_loss + args.lambda_intra * intra_loss)

        # 7. Backward + step
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        optimizer.step()

        # ---- Logging ----
        if step % args.log_every == 0 and is_main():
            scale = info.get("scale", torch.tensor(0.0))
            scale_val = scale.item() if isinstance(scale, torch.Tensor) else scale
            with torch.no_grad():
                if args.feature_mode == "direct":
                    ent_val = -reg_loss.item()  # mean max cosine sim (higher = closer to vocab)
                else:
                    sp = F.softmax(logits / args.temperature, dim=-1)
                    ent_val = -(sp * sp.clamp(min=1e-8).log()).sum(-1).mean().item()
            extra = f"max_cos={-reg_loss.item():.3f}" if args.feature_mode == "direct" else f"sm_ent={ent_val:.2f}"
            logger.info(
                f"step={step:>6}/{args.max_steps}  loss={total_loss.item():.4f}  "
                f"drift={drift_val.item():.4f}  div={div_loss.item():.4f}  "
                f"reg={reg_loss.item():.4f}  intra={intra_loss.item():.4f}  "
                f"pvar={pos_var.item():.4f}  {extra}  "
                f"gnorm={grad_norm:.3f}  lr={lr:.2e}"
            )
            if args.wandb_project:
                import wandb
                m = {"loss": total_loss.item(), "drift_loss": drift_val.item(),
                     "diversity_loss": div_loss.item(), "feat_std": feat_std.item(),
                     "reg_loss": reg_loss.item(), "intra_loss": intra_loss.item(),
                     "pos_variance": pos_var.item(),
                     "lr": lr, "grad_norm": grad_norm, "step": step}
                if args.feature_mode == "direct":
                    m["mean_max_cosine"] = -reg_loss.item()
                else:
                    m["softmax_entropy"] = ent_val
                for k, v in info.items():
                    m[f"drift/{k}"] = v.item() if isinstance(v, torch.Tensor) else v
                wandb.log(m)

        # ---- Eval ----
        if (step % args.eval_every == 0 or step == args.max_steps) and is_main():
            _evaluate(raw_gen, gpt2, vocab_emb, all_features_gpu, token_ids, tok,
                      args, step, device)

        # ---- Save ----
        if (step % args.save_every == 0 or step == args.max_steps) and is_main():
            p_ = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save(dict(step=step, model=raw_gen.state_dict(), args=vars(args)), p_)
            logger.info(f"Saved → {p_}")

    if ddp:
        dist.destroy_process_group()
    if is_main():
        logger.info("Done.")


# ======================== Evaluation ========================

@torch.no_grad()
def _evaluate(gen, gpt2, vocab_emb, all_feats, token_ids, tok, args, step, device):
    gen.eval()

    n_samples = 16
    noise = torch.randn(n_samples, args.seq_len, args.d_model, device=device)
    gen_h = gen(noise)                               # [n, L, d_model]

    if args.feature_mode == "direct":
        tokens = gen.decode_to_tokens(gen_h, vocab_emb)
        feats = emb_stats_features(gen_h)
    else:
        logits = gen_h @ vocab_emb.T
        tokens = logits.argmax(dim=-1)
        if args.feature_mode == "gpt2_soft" and gpt2 is not None:
            emb = vocab_emb[tokens]
            feats = gpt2(inputs_embeds=emb).last_hidden_state.mean(dim=1)
        else:
            soft_emb = F.softmax(logits / args.temperature, dim=-1) @ vocab_emb
            feats = emb_stats_features(soft_emb)

    logger.info("─── Generated samples ───")
    table_rows = []
    for j in range(n_samples):
        text = tok.decode(tokens[j], skip_special_tokens=True)
        # nearest real sample (by feature distance)
        d = torch.cdist(feats[j : j + 1], all_feats)
        nearest_idx = d.argmin(dim=-1).item()
        near_text = tok.decode(token_ids[nearest_idx], skip_special_tokens=True)
        logger.info(f"  [{j:2d}] {text[:140]}")
        if j < 4:
            logger.info(f"       nearest: {near_text[:140]}")
        table_rows.append((text[:200], near_text[:200]))

    # Diversity: unique unigrams / total tokens
    all_tokens = tokens.cpu().tolist()
    flat = [t for seq in all_tokens for t in seq]
    diversity = len(set(flat)) / max(len(flat), 1)
    logger.info(f"  Vocab diversity (unique / total): {diversity:.3f}")

    # Entropy of token distribution
    counts = torch.bincount(tokens.view(-1).cpu(), minlength=len(tok))
    p = counts.float() / counts.sum()
    p = p[p > 0]
    entropy = -(p * p.log()).sum().item()
    logger.info(f"  Token entropy: {entropy:.2f}  (uniform={math.log(len(tok)):.2f})")

    if args.wandb_project:
        import wandb
        tbl = wandb.Table(columns=["step", "idx", "generated", "nearest_real"])
        for j, (gt, nt) in enumerate(table_rows):
            tbl.add_data(step, j, gt, nt)
        wandb.log({"samples": tbl, "eval/diversity": diversity,
                   "eval/entropy": entropy, "step": step})


# ======================== CLI ========================

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="train", choices=["preprocess", "train"])

    # data
    p.add_argument("--seq_len", type=int, default=32)
    p.add_argument("--num_clusters", type=int, default=256)
    p.add_argument("--cache_dir", default="data/v3_cache")

    # generator arch
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    # drift hyper-params (following image drifting defaults)
    p.add_argument("--G", type=int, default=16, help="generators per cluster")
    p.add_argument("--P", type=int, default=32, help="positive samples per cluster")
    p.add_argument("--N", type=int, default=16, help="negative samples per cluster")
    p.add_argument("--cluster_batch", type=int, default=8, help="clusters per step")
    p.add_argument("--R_list", type=float, nargs="+", default=[0.02, 0.05, 0.2])
    p.add_argument("--feature_mode", type=str, default="direct",
                   choices=["direct", "gpt2_soft", "emb_stats"],
                   help="Feature extraction: direct (raw output + vocab reg), gpt2_soft, or emb_stats")
    p.add_argument("--temperature", type=float, default=3.0,
                   help="Softmax temperature for soft-vocab projection (higher=softer)")
    p.add_argument("--top_k", type=int, default=0,
                   help="Top-k logit filtering before softmax (0=disabled, 64 recommended for gpt2_soft)")
    p.add_argument("--lambda_diversity", type=float, default=1.0,
                   help="Weight of diversity regularization loss")
    p.add_argument("--lambda_reg", type=float, default=0.1,
                   help="Weight of vocab proximity regularization (direct mode only)")
    p.add_argument("--lambda_intra", type=float, default=1.0,
                   help="Weight of intra-sequence diversity (penalizes repeated tokens)")

    # optim
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=50000)

    # logging / saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--output_dir", default="runs/v3")
    p.add_argument("--wandb_project", default=None)

    args = p.parse_args()
    if args.mode == "preprocess":
        preprocess(args)
    else:
        train(args)
