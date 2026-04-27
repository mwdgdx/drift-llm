"""
Drift-LLM v2: one-step text generation via drifting.

Supports three modes (controlled by flags):
  1. CE-only baseline:     --lambda_drift 0 --G 1
  2. Drift (embed only):   --feature_model "" --lambda_drift 1.0
  3. Drift + Feature LLM:  --feature_model gpt2 --lambda_drift 1.0  [recommended]

Supports multi-GPU via PyTorch DDP (torchrun).
"""

import argparse
import logging
import os
import sys
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.transformer import TextGenerator
from v2.data import load_alpaca, make_dataloader, make_owt_dataloader
from drift_loss import drift_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def extract_features(hidden, response_mask=None, chunk_sizes=(16,)):
    B, T, D = hidden.shape
    feats = {}
    if response_mask is not None:
        mask_f = response_mask.float().unsqueeze(-1)
        feats["global"] = (hidden * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp(min=1)
    else:
        feats["global"] = hidden.mean(dim=1, keepdim=True)
    for s in chunk_sizes:
        if T >= s:
            n_chunks = T // s
            truncated = hidden[:, :n_chunks * s, :]
            chunked = rearrange(truncated, "b (n s) d -> b n s d", s=s)
            feats[f"chunk_{s}"] = chunked.mean(dim=2)
    return feats


def compute_multiscale_drift(gen_feats, gt_feats, G, R_list, scales):
    total_loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    info_all = {}
    for scale_name in scales:
        if scale_name not in gen_feats or scale_name not in gt_feats:
            continue
        gf = gen_feats[scale_name]
        pf = gt_feats[scale_name]
        BG, N, D = gf.shape
        B = BG // G
        gen_r = rearrange(gf, "(b g) n d -> (b n) g d", g=G)
        pos_r = rearrange(pf, "b n d -> (b n) 1 d")
        loss, info = drift_loss(gen=gen_r, fixed_pos=pos_r, R_list=R_list)
        total_loss = total_loss + loss.mean()
        for k, v in info.items():
            info_all[f"{scale_name}/{k}"] = v
    return total_loss, info_all


def get_cosine_lr(step, max_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
    # --- DDP setup ---
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_rank = 0

    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = TextGenerator(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_prompt_len + args.max_response_len,
        dropout=args.dropout,
    ).to(device, dtype=dtype)

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    raw_model = model.module if ddp else model

    n_params = sum(p.numel() for p in raw_model.parameters()) / 1e6
    if is_main():
        logger.info(f"Generator: {n_params:.1f}M params, d={args.d_model}, L={args.n_layers}, H={args.n_heads}")

    # --- Feature LLM (frozen) ---
    feature_model = None
    feature_emb_weight = None
    if args.feature_model and args.lambda_drift > 0:
        if is_main():
            logger.info(f"Loading Feature LLM: {args.feature_model}")
        feature_model = AutoModel.from_pretrained(args.feature_model).to(device, dtype=dtype)
        feature_model.eval()
        for p in feature_model.parameters():
            p.requires_grad = False
        feature_emb_weight = feature_model.get_input_embeddings().weight
        if is_main():
            feat_dim = feature_emb_weight.shape[1]
            logger.info(f"Feature LLM: {sum(p.numel() for p in feature_model.parameters())/1e6:.1f}M params, d={feat_dim}")

    use_feature_llm = feature_model is not None

    # --- Data ---
    if args.dataset == "owt":
        if is_main():
            logger.info("Loading OpenWebText (streaming, packed sequences)")
        loader = make_owt_dataloader(tokenizer, args.batch_size, seq_len=args.max_response_len,
                                     num_workers=args.num_workers)
        loader_iter = iter(loader)
    else:
        if is_main():
            logger.info("Loading Alpaca dataset")
        ds = load_alpaca(tokenizer, args.max_prompt_len, args.max_response_len)
        if ddp:
            sampler = DistributedSampler(ds, shuffle=True)
            from functools import partial
            from v2.data import collate_fn
            fn = partial(collate_fn, pad_token_id=tokenizer.pad_token_id,
                         max_prompt_len=args.max_prompt_len, max_response_len=args.max_response_len)
            loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                                                 collate_fn=fn, drop_last=True)
        else:
            loader = make_dataloader(ds, tokenizer, args.batch_size, args.max_prompt_len, args.max_response_len)
        loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.999))

    if args.wandb_project and is_main():
        import wandb
        mode_name = "drift_feat" if use_feature_llm else ("drift_emb" if args.lambda_drift > 0 else "ce_only")
        wandb.init(project=args.wandb_project,
                   name=f"{mode_name}_G{args.G}_d{args.d_model}_L{args.n_layers}_{args.dataset}",
                   config=vars(args))

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)

    drift_scales = list(args.drift_scales)
    R_list = tuple(args.R_list)
    chunk_sizes = tuple(args.chunk_sizes)
    G = args.G if args.lambda_drift > 0 else 1

    if is_main():
        logger.info(f"Starting training: {args.max_steps} steps, BS={args.batch_size}"
                     + (f" x {dist.get_world_size()} GPUs" if ddp else "")
                     + f", G={G}, dataset={args.dataset}")

    for step in range(1, args.max_steps + 1):
        # LR schedule with warmup
        lr = get_cosine_lr(step, args.max_steps, args.warmup_steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        try:
            batch = next(loader_iter)
        except StopIteration:
            if args.dataset != "owt" and ddp:
                sampler.set_epoch(step)
            loader_iter = iter(loader)
            batch = next(loader_iter)

        prompt_ids = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)
        B, T_r = response_ids.shape

        tau = 1.0

        prompt_G = prompt_ids.repeat_interleave(G, dim=0)
        response_G = response_ids.repeat_interleave(G, dim=0)
        mask_G = response_mask.repeat_interleave(G, dim=0)

        noise = torch.randn(B * G, T_r, args.d_model, device=device, dtype=dtype)
        logits = model(prompt_G, noise)

        # --- CE loss ---
        ce_loss = torch.tensor(0.0, device=device)
        if args.lambda_ce > 0:
            ce_flat = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                response_G.reshape(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction="none",
            )
            ce_loss = (ce_flat.view(B * G, T_r) * mask_G.float()).sum() / mask_G.float().sum()

        # --- Drift loss ---
        d_loss = torch.tensor(0.0, device=device)
        d_info = {}
        if args.lambda_drift > 0 and G > 1:
            tau = max(args.tau_min, 1.0 - step / args.max_steps * (1.0 - args.tau_min))
            soft_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)

            if use_feature_llm:
                gen_soft_emb = soft_probs @ feature_emb_weight
                gen_out = feature_model(inputs_embeds=gen_soft_emb, output_hidden_states=True)
                gen_hidden = gen_out.last_hidden_state

                with torch.no_grad():
                    gt_out = feature_model(input_ids=response_ids, output_hidden_states=True)
                    gt_hidden = gt_out.last_hidden_state

                gen_feats = extract_features(gen_hidden, mask_G, chunk_sizes)
                gt_feats = extract_features(gt_hidden, response_mask, chunk_sizes)
            else:
                emb_weight = raw_model.get_embedding_weight()
                gen_emb = soft_probs @ emb_weight
                with torch.no_grad():
                    gt_emb = emb_weight[response_ids]
                gen_feats = extract_features(gen_emb, mask_G, chunk_sizes)
                gt_feats = extract_features(gt_emb, response_mask, chunk_sizes)

            d_loss, d_info = compute_multiscale_drift(gen_feats, gt_feats, G, R_list, drift_scales)

        loss = args.lambda_drift * d_loss + args.lambda_ce * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # --- Logging ---
        if step % args.log_every == 0 and is_main():
            tau_val = tau if args.lambda_drift > 0 and G > 1 else 0.0
            log_str = (
                f"step={step}/{args.max_steps} loss={loss.item():.4f} "
                f"drift={d_loss.item():.4f} ce={ce_loss.item():.4f} "
                f"tau={tau_val:.3f} lr={lr:.6f}"
            )
            logger.info(log_str)

            if args.wandb_project:
                import wandb
                metrics = {"loss": loss.item(), "drift_loss": d_loss.item(),
                           "ce_loss": ce_loss.item(), "lr": lr, "step": step}
                for k, v in d_info.items():
                    if isinstance(v, torch.Tensor):
                        metrics[f"drift/{k}"] = v.item()
                wandb.log(metrics)

        # --- Eval (main process only) ---
        if (step % args.eval_every == 0 or step == args.max_steps) and is_main():
            raw_model.eval()
            with torch.no_grad():
                sample_prompt = prompt_ids[:1]
                tokens = raw_model.generate(sample_prompt, T_r)
                text = tokenizer.decode(tokens[0], skip_special_tokens=True)
                prompt_text = tokenizer.decode(sample_prompt[0], skip_special_tokens=True)
                logger.info(f"  Prompt:    {prompt_text[:80]}")
                logger.info(f"  Generated: {text[:200]}")

                gen_all = raw_model.generate(prompt_ids[:min(B, 16)], T_r)
                sub_resp = response_ids[:min(B, 16)]
                sub_mask = response_mask[:min(B, 16)]
                acc = ((gen_all == sub_resp) * sub_mask).float().sum() / sub_mask.float().sum()
                logger.info(f"  Token acc: {acc.item():.4f}")

                if args.wandb_project:
                    import wandb
                    wandb.log({"eval/token_acc": acc.item(), "step": step})
            raw_model.train()

        # --- Save ---
        if (step % args.save_every == 0 or step == args.max_steps) and is_main():
            path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": raw_model.state_dict(), "args": vars(args)}, path)
            logger.info(f"  Saved: {path}")

    if ddp:
        dist.destroy_process_group()
    if is_main():
        logger.info("Training complete!")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Drift-LLM v2 training")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--feature_model", type=str, default="",
                   help="Pretrained model for Feature LLM (e.g. 'gpt2'). Empty = embedding-only drift.")
    p.add_argument("--dataset", type=str, default="owt", choices=["owt", "alpaca"],
                   help="Dataset: 'owt' for OpenWebText, 'alpaca' for Alpaca")
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--max_prompt_len", type=int, default=1)
    p.add_argument("--max_response_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2500)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--lambda_drift", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.1)
    p.add_argument("--R_list", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--drift_scales", type=str, nargs="+", default=["global", "chunk_16"])
    p.add_argument("--chunk_sizes", type=int, nargs="+", default=[16])
    p.add_argument("--tau_min", type=float, default=0.5)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--output_dir", default="runs/v2_drift")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)
