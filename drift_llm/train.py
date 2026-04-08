"""
Drift-LLM training script.

Trains a one-step text generator using drift loss + optional CE loss.
Supports two bridge modes:
  - "gumbel": Gumbel-Softmax → soft_probs @ Embedding → Feature Encoder
  - "embedding": softmax(logits) @ E (no external Feature Encoder, Level 1)
"""

import os
import json
import argparse
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from einops import rearrange, repeat

from drift_loss import drift_loss
from bridge import gumbel_softmax_bridge, softmax_bridge
from generator import OneStepGenerator
from feature_encoder import (
    EmbeddingFeatureEncoder,
    TransformerFeatureEncoder,
    extract_multiscale_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    model_name: str = "GSAI-ML/LLaDA-8B-Base"
    feature_encoder_name: Optional[str] = None
    dataset_name: str = "tatsu-lab/alpaca"

    bridge_mode: str = "embedding"

    G: int = 4
    response_length: int = 128
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 10000
    warmup_steps: int = 500
    log_every: int = 10
    save_every: int = 1000
    eval_every: int = 500

    tau_init: float = 1.0
    tau_final: float = 0.1
    tau_anneal_steps: int = 5000

    lambda_ce: float = 0.1
    lambda_drift: float = 1.0
    R_list: Tuple[float, ...] = (0.5, 1.0, 2.0)

    multiscale_chunks: Tuple[int, ...] = (32,)
    drift_scales: Tuple[str, ...] = ("per_token", "global")

    max_grad_norm: float = 1.0
    output_dir: str = "./runs/drift_llm"
    seed: int = 42
    bf16: bool = True

    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    resume_from: Optional[str] = None
    save_total_limit: int = 3


def get_tau(step: int, config: TrainConfig) -> float:
    if step >= config.tau_anneal_steps:
        return config.tau_final
    frac = step / config.tau_anneal_steps
    return config.tau_init + (config.tau_final - config.tau_init) * frac


def load_generator(config: TrainConfig, device: torch.device) -> OneStepGenerator:
    from transformers import AutoModelForCausalLM

    logger.info(f"Loading generator: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    return OneStepGenerator(model)


def load_feature_encoder(config: TrainConfig, generator: OneStepGenerator, device: torch.device):
    if config.feature_encoder_name is None or config.bridge_mode == "embedding":
        logger.info("Using EmbeddingFeatureEncoder (Level 1, no external model)")
        enc = EmbeddingFeatureEncoder(generator.embedding.weight)
        enc.to(device)
        return enc

    logger.info(f"Loading TransformerFeatureEncoder: {config.feature_encoder_name}")
    enc = TransformerFeatureEncoder(config.feature_encoder_name)
    enc.to(device)
    enc.eval()
    return enc


def load_tokenizer(config: TrainConfig):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)


def load_dataset(config: TrainConfig, tokenizer):
    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading dataset: {config.dataset_name}")
    ds = hf_load_dataset(config.dataset_name, split="train")

    def tokenize_fn(example):
        if "instruction" in example and "output" in example:
            prompt = example["instruction"]
            if example.get("input", ""):
                prompt = prompt + "\n" + example["input"]
            response = example["output"]
        elif "prompt" in example and "completion" in example:
            prompt, response = example["prompt"], example["completion"]
        elif "messages" in example:
            msgs = example["messages"]
            prompt = msgs[0]["content"] if len(msgs) > 0 else ""
            response = msgs[1]["content"] if len(msgs) > 1 else ""
        else:
            prompt = example.get("text", "")[:200]
            response = example.get("text", "")[200:]

        p_ids = tokenizer(
            prompt, truncation=True, max_length=256,
            add_special_tokens=True, return_tensors=None,
        )["input_ids"]
        r_ids = tokenizer(
            response, truncation=True, max_length=config.response_length,
            add_special_tokens=False, return_tensors=None,
        )["input_ids"]
        return {"prompt_ids": p_ids, "response_ids": r_ids}

    ds = ds.map(tokenize_fn, num_proc=4, desc="Tokenizing")
    ds = ds.filter(lambda x: len(x["response_ids"]) >= 8)
    return ds


def collate_fn(batch, pad_token_id: int, max_prompt_len: int, max_response_len: int):
    prompts, responses = [], []
    for item in batch:
        p = item["prompt_ids"][:max_prompt_len]
        r = item["response_ids"][:max_response_len]
        prompts.append(p)
        responses.append(r)

    max_p = max(len(p) for p in prompts)
    max_r = max(len(r) for r in responses)

    prompt_ids = torch.full((len(batch), max_p), pad_token_id, dtype=torch.long)
    response_ids = torch.full((len(batch), max_r), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros(len(batch), max_r, dtype=torch.bool)

    for i, (p, r) in enumerate(zip(prompts, responses)):
        prompt_ids[i, max_p - len(p):] = torch.tensor(p, dtype=torch.long)
        response_ids[i, :len(r)] = torch.tensor(r, dtype=torch.long)
        response_mask[i, :len(r)] = True

    return {"prompt_ids": prompt_ids, "response_ids": response_ids, "response_mask": response_mask}


def compute_drift_features_embedding(
    logits: torch.Tensor,
    gt_ids: torch.Tensor,
    feat_enc: EmbeddingFeatureEncoder,
    config: TrainConfig,
    tau: float,
    bridge_mode: str = "embedding",
) -> Tuple[dict, dict]:
    """
    Compute gen/gt features for drift loss using embedding projection (Level 1).

    logits: [B*G, T_r, V]
    gt_ids: [B, T_r]

    Returns: gen_features, gt_features (dicts of {scale: tensor})
    """
    if bridge_mode == "gumbel":
        soft_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
    else:
        soft_probs = F.softmax(logits, dim=-1)

    gen_emb = soft_probs @ feat_enc.weight       # [B*G, T_r, D]
    gt_emb = feat_enc.encode_tokens(gt_ids)       # [B, T_r, D]

    gen_feats = extract_multiscale_features(gen_emb, config.multiscale_chunks)
    gt_feats = extract_multiscale_features(gt_emb, config.multiscale_chunks)

    return gen_feats, gt_feats


def compute_drift_features_transformer(
    logits: torch.Tensor,
    gt_ids: torch.Tensor,
    feat_enc: TransformerFeatureEncoder,
    config: TrainConfig,
    tau: float,
) -> Tuple[dict, dict]:
    """
    Compute gen/gt features using a frozen Transformer Feature Encoder (Level 2).

    logits: [B*G, T_r, V]
    gt_ids: [B, T_r]
    """
    soft_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
    soft_emb = soft_probs @ feat_enc.embedding_weight  # [B*G, T_r, D_feat]
    gen_hidden = feat_enc.forward_from_embeds(soft_emb)  # [B*G, T_r, D_feat]

    with torch.no_grad():
        gt_hidden = feat_enc.forward_from_tokens(gt_ids)  # [B, T_r, D_feat]

    gen_feats = extract_multiscale_features(gen_hidden, config.multiscale_chunks)
    gt_feats = extract_multiscale_features(gt_hidden, config.multiscale_chunks)

    return gen_feats, gt_feats


def compute_total_drift_loss(gen_feats: dict, gt_feats: dict, G: int, config: TrainConfig):
    """
    Multi-scale drift loss: for each scale, reshape to [particles, G, D] and compute.
    """
    total_loss = 0.0
    info_all = {}

    for scale_name in config.drift_scales:
        if scale_name not in gen_feats:
            continue

        gf = gen_feats[scale_name]   # [B*G, N, D]
        pf = gt_feats[scale_name]    # [B, N, D]

        BG, N, D = gf.shape
        B = BG // G

        gen_r = rearrange(gf, "(b g) n d -> (b n) g d", g=G)
        pos_r = rearrange(pf, "b n d -> (b n) 1 d")

        loss, info = drift_loss(
            gen=gen_r,
            fixed_pos=pos_r,
            R_list=config.R_list,
        )
        total_loss = total_loss + loss.mean()
        for k, v in info.items():
            info_all[f"{scale_name}/{k}"] = v

    return total_loss, info_all


def save_checkpoint(generator, optimizer, scheduler, step, config):
    """Save full training state for resume."""
    ckpt_path = os.path.join(config.output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save({
        "step": step,
        "generator": generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, os.path.join(ckpt_path, "state.pt"))

    # Keep only last N checkpoints
    import glob
    ckpts = sorted(glob.glob(os.path.join(config.output_dir, "checkpoint-*")))
    while len(ckpts) > config.save_total_limit:
        old = ckpts.pop(0)
        import shutil
        shutil.rmtree(old)
        logger.info(f"Removed old checkpoint: {old}")

    logger.info(f"Saved checkpoint to {ckpt_path}")


def load_checkpoint(generator, optimizer, scheduler, config):
    """Load training state if resume_from is set. Returns start step."""
    if not config.resume_from:
        return 0
    ckpt_file = os.path.join(config.resume_from, "state.pt")
    if not os.path.exists(ckpt_file):
        logger.warning(f"No checkpoint found at {ckpt_file}, starting from scratch")
        return 0
    state = torch.load(ckpt_file, map_location="cpu")
    generator.load_state_dict(state["generator"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    step = state["step"]
    logger.info(f"Resumed from {config.resume_from} at step {step}")
    return step


def init_wandb(config: TrainConfig):
    """Initialize wandb if project name is set."""
    if not config.wandb_project:
        return None
    import wandb
    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name or f"G{config.G}_{config.bridge_mode}_ce{config.lambda_ce}_drift{config.lambda_drift}",
        config=asdict(config),
        resume="allow",
    )
    return run


def train(config: TrainConfig):
    torch.manual_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    wandb_run = init_wandb(config)

    generator = load_generator(config, device)
    feat_enc = load_feature_encoder(config, generator, device)
    tokenizer = load_tokenizer(config)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = load_dataset(config, tokenizer)

    from functools import partial
    collate = partial(
        collate_fn,
        pad_token_id=tokenizer.pad_token_id,
        max_prompt_len=256,
        max_response_len=config.response_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate, num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    num_training_steps = config.max_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_training_steps, eta_min=config.lr * 0.1,
    )

    step = load_checkpoint(generator, optimizer, scheduler, config)

    logger.info(f"Starting training for {num_training_steps} steps (from step {step})")
    logger.info(f"  Bridge mode: {config.bridge_mode}")
    logger.info(f"  G (candidates): {config.G}")
    logger.info(f"  Response length: {config.response_length}")
    logger.info(f"  Drift scales: {config.drift_scales}")
    logger.info(f"  Lambda CE: {config.lambda_ce}, Lambda Drift: {config.lambda_drift}")
    logger.info(f"  WandB: {config.wandb_project or 'disabled'}")

    data_iter = iter(dataloader)

    while step < num_training_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        prompt_ids = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)

        B = prompt_ids.shape[0]
        G = config.G
        T_r = response_ids.shape[1]
        tau = get_tau(step, config)

        # --- Generator forward: G candidates per prompt ---
        result = generator.generate_candidates(
            prompt_ids=prompt_ids,
            response_length=T_r,
            num_candidates=G,
        )
        logits = result["logits"]  # [B*G, T_r, V]

        # --- Drift loss ---
        if isinstance(feat_enc, TransformerFeatureEncoder):
            gen_feats, gt_feats = compute_drift_features_transformer(
                logits, response_ids, feat_enc, config, tau,
            )
        else:
            gen_feats, gt_feats = compute_drift_features_embedding(
                logits, response_ids, feat_enc, config, tau, config.bridge_mode,
            )

        drift_total, drift_info = compute_total_drift_loss(gen_feats, gt_feats, G, config)

        # --- CE loss (on first candidate only) ---
        ce_loss = torch.tensor(0.0, device=device)
        if config.lambda_ce > 0:
            logits_ce = rearrange(logits, "(b g) t v -> b g t v", g=G)[:, 0]  # [B, T_r, V]
            ce_loss = F.cross_entropy(
                logits_ce.reshape(-1, logits_ce.size(-1)),
                response_ids.reshape(-1),
                reduction="none",
            )
            ce_loss = (ce_loss * response_mask.reshape(-1).float()).sum() / response_mask.sum().clamp(min=1)

        # --- Total loss ---
        total_loss = config.lambda_drift * drift_total + config.lambda_ce * ce_loss

        # --- Backward ---
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(generator.parameters(), config.max_grad_norm)
        optimizer.step()
        scheduler.step()

        step += 1

        # --- Logging ---
        if step % config.log_every == 0:
            metrics = {
                "train/loss": total_loss.item(),
                "train/drift_loss": drift_total.item(),
                "train/ce_loss": ce_loss.item(),
                "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "train/tau": tau,
                "train/lr": scheduler.get_last_lr()[0],
                "train/step": step,
            }
            for k, v in drift_info.items():
                if isinstance(v, torch.Tensor):
                    metrics[f"drift/{k}"] = v.item()

            log_str = " ".join(f"{k.split('/')[-1]}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
            logger.info(f"step={step} {log_str}")

            if wandb_run:
                wandb_run.log(metrics, step=step)

        # --- Save ---
        if step % config.save_every == 0:
            save_checkpoint(generator, optimizer, scheduler, step, config)

        # --- Eval (always compute, regardless of loss config) ---
        if step % config.eval_every == 0:
            eval_metrics = {}
            with torch.no_grad():
                # --- Eval on current batch: one-step generate vs GT ---
                eval_result = generator.forward(
                    prompt_ids=prompt_ids, response_length=T_r,
                )
                eval_logits = eval_result["logits"]  # [B, T_r, V]
                eval_tokens = eval_logits.argmax(dim=-1)  # [B, T_r]

                # Token accuracy: what fraction of positions match GT?
                match = (eval_tokens == response_ids).float()
                masked_match = match * response_mask.float()
                token_acc = masked_match.sum() / response_mask.sum().clamp(min=1)
                eval_metrics["eval/token_accuracy"] = token_acc.item()

                # Eval CE loss (always computed, even when lambda_ce=0)
                eval_ce = F.cross_entropy(
                    eval_logits.reshape(-1, eval_logits.size(-1)),
                    response_ids.reshape(-1),
                    reduction="none",
                )
                eval_ce = (eval_ce * response_mask.reshape(-1).float()).sum() / response_mask.sum().clamp(min=1)
                eval_metrics["eval/ce_loss"] = eval_ce.item()

                # Sample text
                sample_prompt = prompt_ids[:1]
                tokens = generator.generate(sample_prompt, T_r)
                generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
                gt_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
                logger.info(f"[Eval step={step}] GT:        {gt_text[:200]}")
                logger.info(f"[Eval step={step}] Generated: {generated_text[:200]}")

                # Diversity: different noise → different output?
                diversity_tokens = []
                for _ in range(4):
                    t = generator.generate(sample_prompt, T_r)
                    diversity_tokens.append(t[0])
                diversity_tokens = torch.stack(diversity_tokens)
                n = diversity_tokens.shape[0]
                agree = sum(
                    (diversity_tokens[i] == diversity_tokens[j]).float().mean().item()
                    for i in range(n) for j in range(i+1, n)
                ) / max(1, n * (n-1) / 2)
                eval_metrics["eval/token_agreement"] = agree

                logger.info(f"[Eval step={step}] Token acc: {token_acc.item():.3f}  "
                           f"CE: {eval_ce.item():.3f}  "
                           f"Diversity (1-agreement): {1-agree:.3f}")

            if wandb_run:
                wandb_run.log(eval_metrics, step=step)

    logger.info("Training complete! Running final evaluation...")
    save_checkpoint(generator, optimizer, scheduler, step, config)

    # --- Final evaluation on multiple batches ---
    generator.model.eval()
    final_metrics = {"token_acc": [], "ce_loss": [], "token_agreement": []}
    n_eval_batches = 50
    eval_iter = iter(dataloader)

    with torch.no_grad():
        for i in range(n_eval_batches):
            try:
                batch = next(eval_iter)
            except StopIteration:
                eval_iter = iter(dataloader)
                batch = next(eval_iter)

            p_ids = batch["prompt_ids"].to(device)
            r_ids = batch["response_ids"].to(device)
            r_mask = batch["response_mask"].to(device)
            T_r_eval = r_ids.shape[1]

            result = generator.forward(prompt_ids=p_ids, response_length=T_r_eval)
            logits_eval = result["logits"]
            tokens_eval = logits_eval.argmax(dim=-1)

            match = (tokens_eval == r_ids).float() * r_mask.float()
            acc = match.sum() / r_mask.sum().clamp(min=1)
            final_metrics["token_acc"].append(acc.item())

            ce = F.cross_entropy(
                logits_eval.reshape(-1, logits_eval.size(-1)),
                r_ids.reshape(-1),
                reduction="none",
            )
            ce = (ce * r_mask.reshape(-1).float()).sum() / r_mask.sum().clamp(min=1)
            final_metrics["ce_loss"].append(ce.item())

            # Diversity on first prompt
            if i == 0:
                div_tokens = [generator.generate(p_ids[:1], T_r_eval) for _ in range(8)]
                div_tokens = torch.stack([t[0] for t in div_tokens])
                n = div_tokens.shape[0]
                agree = sum(
                    (div_tokens[j] == div_tokens[k]).float().mean().item()
                    for j in range(n) for k in range(j+1, n)
                ) / max(1, n * (n-1) / 2)
                final_metrics["token_agreement"].append(agree)

        # Print 5 samples
        logger.info(f"\n{'='*60}")
        logger.info("FINAL EVALUATION SAMPLES")
        logger.info(f"{'='*60}")
        for s in range(min(5, p_ids.shape[0])):
            gen_text = tokenizer.decode(tokens_eval[s], skip_special_tokens=True)
            gt_text = tokenizer.decode(r_ids[s], skip_special_tokens=True)
            prompt_text = tokenizer.decode(p_ids[s], skip_special_tokens=True)
            logger.info(f"\nPrompt:    {prompt_text[:100]}")
            logger.info(f"GT:        {gt_text[:200]}")
            logger.info(f"Generated: {gen_text[:200]}")

    avg_acc = sum(final_metrics["token_acc"]) / len(final_metrics["token_acc"])
    avg_ce = sum(final_metrics["ce_loss"]) / len(final_metrics["ce_loss"])
    avg_agree = sum(final_metrics["token_agreement"]) / max(1, len(final_metrics["token_agreement"]))

    logger.info(f"\n{'='*60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Token accuracy:   {avg_acc:.4f}")
    logger.info(f"  CE loss:          {avg_ce:.4f}")
    logger.info(f"  Token agreement:  {avg_agree:.4f}  (lower = more diverse)")
    logger.info(f"  Diversity:        {1-avg_agree:.4f}")

    if wandb_run:
        wandb_run.log({
            "final/token_accuracy": avg_acc,
            "final/ce_loss": avg_ce,
            "final/token_agreement": avg_agree,
            "final/diversity": 1 - avg_agree,
        })
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Drift-LLM Training")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--feature_encoder_name", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca")
    parser.add_argument("--bridge_mode", type=str, default="embedding", choices=["embedding", "gumbel"])
    parser.add_argument("--G", type=int, default=4)
    parser.add_argument("--response_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--lambda_ce", type=float, default=0.1)
    parser.add_argument("--lambda_drift", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="./runs/drift_llm")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()
    config = TrainConfig(**{k: v for k, v in vars(args).items() if v is not None})
    train(config)


if __name__ == "__main__":
    main()
