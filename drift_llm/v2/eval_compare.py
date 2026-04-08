"""
Unified evaluation: compare Drift vs Diffusion-LM on the same test data.

For each model, tests:
  - 1-step generation (Drift's sweet spot)
  - Multi-step generation (Diffusion-LM's advantage)

Metrics: BLEU, Distinct-n, Repetition ratio, Token accuracy (soft).
"""

import argparse
import logging
import math
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.transformer import TextGenerator
from v2.data import load_alpaca, make_dataloader
from v2.diffusion_train import diffusion_sample, cosine_noise_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))
    log_avg = 0.0
    for n in range(1, max_n + 1):
        ref_ng = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ng = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        p = clipped / total
        if p == 0:
            return 0.0
        log_avg += math.log(p)
    return bp * math.exp(log_avg / max_n)


def compute_distinct(text: str):
    tokens = text.lower().split()
    if not tokens:
        return 0.0, 0.0
    d1 = len(set(tokens)) / len(tokens)
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    return d1, d2


def rep_ratio(text: str):
    tokens = text.lower().split()
    if len(tokens) <= 1:
        return 0.0
    return sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1]) / (len(tokens) - 1)


@torch.no_grad()
def evaluate_model(model, loader, tokenizer, device, d_model, n_eval=50, diffusion_steps=50):
    model.eval()
    results = {"1step": [], "multi": []}
    count = 0

    for batch in loader:
        if count >= n_eval:
            break
        prompt_ids = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)
        B, T_r = response_ids.shape

        # 1-step
        tokens_1 = model.generate(prompt_ids, T_r)

        # Multi-step diffusion
        tokens_m = diffusion_sample(model, prompt_ids, T_r, d_model, n_steps=diffusion_steps, device=device)

        for i in range(B):
            if count >= n_eval:
                break
            gt_text = tokenizer.decode(response_ids[i][response_mask[i]], skip_special_tokens=True)
            text_1 = tokenizer.decode(tokens_1[i], skip_special_tokens=True)
            text_m = tokenizer.decode(tokens_m[i], skip_special_tokens=True)

            # Token accuracy
            valid = response_mask[i]
            acc_1 = (tokens_1[i][:T_r][valid] == response_ids[i][valid]).float().mean().item()
            acc_m = (tokens_m[i][:T_r][valid] == response_ids[i][valid]).float().mean().item()

            bleu_1 = compute_bleu(gt_text, text_1)
            bleu_m = compute_bleu(gt_text, text_m)
            d1_1, d2_1 = compute_distinct(text_1)
            d1_m, d2_m = compute_distinct(text_m)
            rep_1 = rep_ratio(text_1)
            rep_m = rep_ratio(text_m)

            results["1step"].append({"acc": acc_1, "bleu": bleu_1, "d1": d1_1, "d2": d2_1, "rep": rep_1, "text": text_1, "gt": gt_text})
            results["multi"].append({"acc": acc_m, "bleu": bleu_m, "d1": d1_m, "d2": d2_m, "rep": rep_m, "text": text_m})

            if count < 3:
                prompt_text = tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                logger.info(f"\nPrompt: {prompt_text[:80]}")
                logger.info(f"GT:      {gt_text[:100]}")
                logger.info(f"1-step:  {text_1[:100]}")
                logger.info(f"Multi:   {text_m[:100]}")

            count += 1

    return results


def avg(lst, key):
    vals = [x[key] for x in lst]
    return sum(vals) / max(len(vals), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True, help="Paths to .pt checkpoint files")
    p.add_argument("--labels", nargs="+", default=None, help="Labels for each checkpoint")
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--n_eval", type=int, default=50)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_results = {}

    for ckpt_path, label in zip(args.checkpoints, args.labels):
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {label} ({ckpt_path})")
        logger.info(f"{'='*60}")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt["args"]

        model = TextGenerator(
            vocab_size=len(tokenizer),
            d_model=ckpt_args["d_model"],
            n_layers=ckpt_args["n_layers"],
            n_heads=ckpt_args["n_heads"],
            max_seq_len=ckpt_args["max_prompt_len"] + ckpt_args["max_response_len"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        ds = load_alpaca(tokenizer, ckpt_args["max_prompt_len"], ckpt_args["max_response_len"])
        loader = make_dataloader(ds, tokenizer, args.batch_size,
                                 ckpt_args["max_prompt_len"], ckpt_args["max_response_len"])

        results = evaluate_model(
            model, loader, tokenizer, device,
            d_model=ckpt_args["d_model"],
            n_eval=args.n_eval,
            diffusion_steps=args.diffusion_steps,
        )
        all_results[label] = results

    # --- Print comparison table ---
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'='*80}")
    header = f"{'Method':<25} {'Mode':<8} {'BLEU':>6} {'TokAcc':>7} {'Dist-1':>7} {'Dist-2':>7} {'Rep':>5}"
    logger.info(header)
    logger.info("-" * len(header))

    for label, results in all_results.items():
        for mode in ["1step", "multi"]:
            r = results[mode]
            mode_label = "1-step" if mode == "1step" else f"multi"
            logger.info(
                f"{label:<25} {mode_label:<8} "
                f"{avg(r, 'bleu'):>6.4f} {avg(r, 'acc'):>7.4f} "
                f"{avg(r, 'd1'):>7.3f} {avg(r, 'd2'):>7.3f} {avg(r, 'rep'):>5.3f}"
            )


if __name__ == "__main__":
    main()
