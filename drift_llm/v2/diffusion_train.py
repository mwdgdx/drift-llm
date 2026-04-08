"""
Diffusion-LM baseline: standard continuous diffusion in embedding space.

Same model architecture as Drift, but trained with DDPM-style denoising loss.
At inference, requires T steps of iterative denoising (vs Drift's 1 step).
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cosine_noise_schedule(t, s=0.008):
    """Cosine schedule: alpha_bar(t) = cos((t+s)/(1+s) * pi/2)^2"""
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


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
        wandb.init(project=args.wandb_project, name=f"diffusion_d{args.d_model}_L{args.n_layers}",
                   config=vars(args))

    os.makedirs(args.output_dir, exist_ok=True)
    emb_weight = model.get_embedding_weight()

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

        # Clean embeddings of GT response
        with torch.no_grad():
            x_0 = emb_weight[response_ids]  # [B, T_r, D]

        # Sample random timestep t ~ U(0, 1)
        t = torch.rand(B, 1, 1, device=device)
        alpha_bar = cosine_noise_schedule(t)

        # Forward diffusion: x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * eps
        eps = torch.randn_like(x_0)
        x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps

        # Model predicts x_0 from x_t (x-prediction, same as Diffusion-LM)
        logits = model(prompt_ids, x_t)  # [B, T_r, V]

        # Loss: CE on predicted tokens (model predicts which token each position should be)
        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            response_ids.reshape(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction="none",
        )
        loss = (ce_loss.view(B, T_r) * response_mask.float()).sum() / response_mask.float().sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0:
            logger.info(
                f"step={step} loss={loss.item():.4f} "
                f"lr={scheduler.get_last_lr()[0]:.6f}"
            )
            if args.wandb_project:
                import wandb
                wandb.log({"loss": loss.item(), "step": step})

        if step % args.eval_every == 0 or step == args.max_steps:
            model.eval()
            with torch.no_grad():
                # 1-step generation (from pure noise, same as drift)
                tokens_1step = model.generate(prompt_ids[:1], T_r)
                text_1step = tokenizer.decode(tokens_1step[0], skip_special_tokens=True)

                # Multi-step generation (diffusion's advantage)
                tokens_multi = diffusion_sample(
                    model, prompt_ids[:1], T_r, args.d_model,
                    n_steps=args.diffusion_steps, device=device,
                )
                text_multi = tokenizer.decode(tokens_multi[0], skip_special_tokens=True)

                prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                logger.info(f"  Prompt: {prompt_text[:60]}")
                logger.info(f"  1-step: {text_1step[:120]}")
                logger.info(f"  {args.diffusion_steps}-step: {text_multi[:120]}")

                # Token accuracy (1-step)
                gen_all = model.generate(prompt_ids, T_r)
                acc = ((gen_all == response_ids) * response_mask).float().sum() / response_mask.float().sum()
                logger.info(f"  Token acc (1-step): {acc.item():.4f}")

                if args.wandb_project:
                    import wandb
                    wandb.log({"eval/token_acc_1step": acc.item(), "step": step})
            model.train()

        if step % args.save_every == 0 or step == args.max_steps:
            path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, path)
            logger.info(f"  Saved: {path}")

    logger.info("Training complete!")


@torch.no_grad()
def diffusion_sample(model, prompt_ids, response_length, d_model, n_steps=50, device="cuda"):
    """DDPM-style iterative denoising: x_T → x_{T-1} → ... → x_0 → tokens."""
    B = prompt_ids.shape[0]
    emb_weight = model.get_embedding_weight()

    x = torch.randn(B, response_length, d_model, device=device)

    for i in range(n_steps, 0, -1):
        t = torch.tensor(i / n_steps, device=device)
        t_prev = torch.tensor((i - 1) / n_steps, device=device)

        alpha_bar_t = cosine_noise_schedule(t)
        alpha_bar_prev = cosine_noise_schedule(t_prev)

        logits = model(prompt_ids, x)
        pred_x0 = emb_weight[logits.argmax(dim=-1)]  # predicted clean embeddings

        if i > 1:
            noise = torch.randn_like(x)
            x = alpha_bar_prev.sqrt() * pred_x0 + (1 - alpha_bar_prev).sqrt() * noise
        else:
            x = pred_x0

    final_logits = model(prompt_ids, x)
    return final_logits.argmax(dim=-1)


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
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--output_dir", default="runs/v2_diffusion")
    p.add_argument("--wandb_project", default=None)
    args = p.parse_args()
    train(args)
