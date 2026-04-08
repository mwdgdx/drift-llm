"""
Drift-LLM training: one-step text generation via drifting.

The generator sees prompt_tokens + noise, outputs logits in one forward pass.
Drift loss pushes the generated distribution toward the real data distribution.
"""

import argparse
import logging
import os
import sys
import math

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.transformer import TextGenerator
from v2.data import load_alpaca, make_dataloader
from drift_loss import drift_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model: {n_params:.1f}M params, d={args.d_model}, L={args.n_layers}, H={args.n_heads}")

    ds = load_alpaca(tokenizer, args.max_prompt_len, args.max_response_len)
    loader = make_dataloader(ds, tokenizer, args.batch_size, args.max_prompt_len, args.max_response_len)
    loader_iter = iter(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name=f"drift_G{args.G}_d{args.d_model}_L{args.n_layers}",
                   config=vars(args))

    os.makedirs(args.output_dir, exist_ok=True)

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        prompt_ids = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)
        B, T_r = response_ids.shape

        # --- G candidates per prompt ---
        prompt_G = prompt_ids.repeat_interleave(args.G, dim=0)
        response_G = response_ids.repeat_interleave(args.G, dim=0)
        mask_G = response_mask.repeat_interleave(args.G, dim=0)

        noise = torch.randn(B * args.G, T_r, args.d_model, device=device, dtype=dtype)
        logits = model(prompt_G, noise)  # [B*G, T_r, V]

        # --- CE loss (auxiliary) ---
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            response_G.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="none",
        )
        ce_loss = (ce_loss.view(B * args.G, T_r) * mask_G.float()).sum() / mask_G.float().sum()

        # --- Drift loss ---
        tau = max(0.5, 1.0 - step / args.max_steps * 0.5)
        soft_probs = F.gumbel_softmax(logits, tau=tau, hard=False, dim=-1)
        emb_weight = model.get_embedding_weight()
        gen_emb = soft_probs @ emb_weight  # [B*G, T_r, D]
        gen_feat = gen_emb.mean(dim=1)     # [B*G, D] — global mean pooling

        with torch.no_grad():
            gt_emb = emb_weight[response_ids]  # [B, T_r, D]
            gt_mask_f = response_mask.float().unsqueeze(-1)
            gt_feat = (gt_emb * gt_mask_f).sum(dim=1) / gt_mask_f.sum(dim=1).clamp(min=1)  # [B, D]

        gen_feat_grouped = gen_feat.view(B, args.G, -1)
        gt_feat_grouped = gt_feat.unsqueeze(1)

        d_loss, d_info = drift_loss(
            gen=gen_feat_grouped,
            fixed_pos=gt_feat_grouped,
            R_list=tuple(args.R_list),
        )
        d_loss = d_loss.mean()

        loss = args.lambda_drift * d_loss + args.lambda_ce * ce_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            log_str = (
                f"step={step} loss={loss.item():.4f} "
                f"drift={d_loss.item():.4f} ce={ce_loss.item():.4f} "
                f"tau={tau:.3f} lr={scheduler.get_last_lr()[0]:.6f}"
            )
            for k, v in d_info.items():
                if isinstance(v, torch.Tensor):
                    log_str += f" {k}={v.item():.4f}"
            logger.info(log_str)

            if args.wandb_project:
                import wandb
                wandb.log({"loss": loss.item(), "drift_loss": d_loss.item(),
                           "ce_loss": ce_loss.item(), "tau": tau, "step": step})

        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                sample_prompt = prompt_ids[:1]
                tokens = model.generate(sample_prompt, T_r)
                text = tokenizer.decode(tokens[0], skip_special_tokens=True)
                prompt_text = tokenizer.decode(sample_prompt[0], skip_special_tokens=True)
                logger.info(f"  Prompt: {prompt_text[:60]}")
                logger.info(f"  Generated: {text[:120]}")

                # token accuracy vs GT
                gen_all = model.generate(prompt_ids, T_r)
                acc = ((gen_all == response_ids) * response_mask).float().sum() / response_mask.float().sum()
                logger.info(f"  Token acc: {acc.item():.4f}")

                if args.wandb_project:
                    import wandb
                    wandb.log({"eval/token_acc": acc.item(), "step": step})
            model.train()

        if step % args.save_every == 0 or step == args.max_steps:
            path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, path)
            logger.info(f"  Saved: {path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--max_prompt_len", type=int, default=128)
    p.add_argument("--max_response_len", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--G", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--lambda_drift", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.1)
    p.add_argument("--R_list", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--output_dir", default="runs/v2_drift")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()
    train(args)
