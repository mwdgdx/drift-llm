"""
Diffusion-LM baseline: standard continuous diffusion in embedding space.

Same model architecture as Drift, but trained with DDPM-style denoising loss.
At inference, requires T steps of iterative denoising (vs Drift's 1 step).

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
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.transformer import TextGenerator
from v2.data import load_alpaca, make_dataloader, make_owt_dataloader

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def cosine_noise_schedule(t, s=0.008):
    return torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2


def get_cosine_lr(step, max_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return lr * 0.5 * (1 + math.cos(math.pi * progress))


def train(args):
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
        logger.info(f"Model: {n_params:.1f}M params, d={args.d_model}, L={args.n_layers}, H={args.n_heads}")

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
        wandb.init(project=args.wandb_project,
                   name=f"diffusion_d{args.d_model}_L{args.n_layers}_{args.dataset}",
                   config=vars(args))

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Starting training: {args.max_steps} steps, BS={args.batch_size}"
                     + (f" x {dist.get_world_size()} GPUs" if ddp else ""))

    for step in range(1, args.max_steps + 1):
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

        with torch.no_grad():
            emb_weight = raw_model.get_embedding_weight()
            x_0 = emb_weight[response_ids]

        t = torch.rand(B, 1, 1, device=device)
        alpha_bar = cosine_noise_schedule(t)
        eps = torch.randn_like(x_0)
        x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps

        logits = model(prompt_ids, x_t)

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

        if step % args.log_every == 0 and is_main():
            logger.info(f"step={step}/{args.max_steps} loss={loss.item():.4f} lr={lr:.6f}")
            if args.wandb_project:
                import wandb
                wandb.log({"loss": loss.item(), "lr": lr, "step": step})

        if (step % args.eval_every == 0 or step == args.max_steps) and is_main():
            raw_model.eval()
            with torch.no_grad():
                tokens_1step = raw_model.generate(prompt_ids[:1], T_r)
                text_1step = tokenizer.decode(tokens_1step[0], skip_special_tokens=True)

                tokens_multi = diffusion_sample(
                    raw_model, prompt_ids[:1], T_r, args.d_model,
                    n_steps=args.diffusion_steps, device=device,
                )
                text_multi = tokenizer.decode(tokens_multi[0], skip_special_tokens=True)

                prompt_text = tokenizer.decode(prompt_ids[0], skip_special_tokens=True)
                logger.info(f"  Prompt: {prompt_text[:60]}")
                logger.info(f"  1-step: {text_1step[:150]}")
                logger.info(f"  {args.diffusion_steps}-step: {text_multi[:150]}")

                gen_all = raw_model.generate(prompt_ids[:min(B, 16)], T_r)
                sub_resp = response_ids[:min(B, 16)]
                sub_mask = response_mask[:min(B, 16)]
                acc = ((gen_all == sub_resp) * sub_mask).float().sum() / sub_mask.float().sum()
                logger.info(f"  Token acc (1-step): {acc.item():.4f}")

                if args.wandb_project:
                    import wandb
                    wandb.log({"eval/token_acc_1step": acc.item(), "step": step})
            raw_model.train()

        if (step % args.save_every == 0 or step == args.max_steps) and is_main():
            path = os.path.join(args.output_dir, f"step_{step}.pt")
            torch.save({"step": step, "model": raw_model.state_dict(), "args": vars(args)}, path)
            logger.info(f"  Saved: {path}")

    if ddp:
        dist.destroy_process_group()
    if is_main():
        logger.info("Training complete!")


@torch.no_grad()
def diffusion_sample(model, prompt_ids, response_length, d_model, n_steps=50, device="cuda"):
    B = prompt_ids.shape[0]
    emb_weight = model.get_embedding_weight()
    x = torch.randn(B, response_length, d_model, device=device)

    for i in range(n_steps, 0, -1):
        t = torch.tensor(i / n_steps, device=device)
        t_prev = torch.tensor((i - 1) / n_steps, device=device)
        alpha_bar_t = cosine_noise_schedule(t)
        alpha_bar_prev = cosine_noise_schedule(t_prev)

        logits = model(prompt_ids, x)
        pred_x0 = emb_weight[logits.argmax(dim=-1)]

        if i > 1:
            noise = torch.randn_like(x)
            x = alpha_bar_prev.sqrt() * pred_x0 + (1 - alpha_bar_prev).sqrt() * noise
        else:
            x = pred_x0

    final_logits = model(prompt_ids, x)
    return final_logits.argmax(dim=-1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--dataset", type=str, default="owt", choices=["owt", "alpaca"])
    p.add_argument("--d_model", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--max_prompt_len", type=int, default=1)
    p.add_argument("--max_response_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=2500)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--output_dir", default="runs/v2_diffusion")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)
