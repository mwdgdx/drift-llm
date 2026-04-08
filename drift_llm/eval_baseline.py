"""
Baseline evaluation: test LLaDA with different inference strategies.

Uses actual Alpaca dataset (same as training) so format is consistent.

Metrics that actually matter for generative models:
  - BLEU (n-gram overlap, tolerates different wording)
  - Semantic similarity (sentence embedding cosine sim)
  - Distinct-1/2 (diversity, penalizes repetition)
  - Coherence (is it real language? measured by repeat ratio)

Compares:
  1. Multi-step with [MASK] (standard LLaDA inference)
  2. One-step with [MASK]
  3. One-step with Gaussian noise (our current approach)

Usage:
    python eval_baseline.py --model_name /path/to/LLaDA-8B-Base
    python eval_baseline.py --model_name /path/to/LLaDA-8B-Base --steps 10 50 128 --num_examples 20
"""

import argparse
import logging
import math
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str, max_n: int = 4) -> dict:
    """Corpus-level BLEU (single reference). Returns bleu-1 through bleu-n and combined."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return {f"bleu_{i}": 0.0 for i in range(1, max_n + 1)}

    brevity_penalty = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    scores = {}
    log_avg = 0.0
    n_valid = 0
    for n in range(1, max_n + 1):
        ref_ngrams = Counter(tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i:i+n]) for i in range(len(hyp_tokens) - n + 1))

        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = max(sum(hyp_ngrams.values()), 1)
        precision = clipped / total
        scores[f"bleu_{n}"] = precision

        if precision > 0:
            log_avg += math.log(precision)
            n_valid += 1

    if n_valid > 0:
        scores["bleu"] = brevity_penalty * math.exp(log_avg / max_n)
    else:
        scores["bleu"] = 0.0

    return scores


def compute_distinct(text: str) -> dict:
    """Distinct-1 and Distinct-2: fraction of unique n-grams."""
    tokens = text.lower().split()
    if len(tokens) == 0:
        return {"distinct_1": 0.0, "distinct_2": 0.0}

    unigrams = [tokens[i] for i in range(len(tokens))]
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]

    d1 = len(set(unigrams)) / max(len(unigrams), 1)
    d2 = len(set(bigrams)) / max(len(bigrams), 1)
    return {"distinct_1": d1, "distinct_2": d2}


def compute_repetition_ratio(text: str) -> float:
    """Fraction of repeated tokens (higher = more repetitive, bad)."""
    tokens = text.lower().split()
    if len(tokens) <= 1:
        return 0.0
    repeated = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i-1])
    return repeated / (len(tokens) - 1)


# ──────────────────────────────────────────────
# Generation methods
# ──────────────────────────────────────────────

def get_mask_token_id(model, tokenizer):
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        return model.config.mask_token_id
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    logger.warning("No mask_token_id found in config or tokenizer!")
    return None


@torch.no_grad()
def generate_multistep_mask(model, prompt_ids, response_length, mask_token_id, num_steps):
    """Standard LLaDA multi-step: start with [MASK], iteratively unmask most confident."""
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    T_r = response_length

    response_ids = torch.full((B, T_r), mask_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    T_p = prompt_ids.shape[1]

    is_masked = torch.ones(B, T_r, device=device, dtype=torch.bool)

    for step in range(num_steps):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, T_p:, :]

        pred_tokens = logits.argmax(dim=-1)
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        confidence[~is_masked] = -1.0

        n_masked = is_masked.sum(dim=-1)
        if n_masked.sum() == 0:
            break

        n_to_unmask = torch.clamp(
            (n_masked.float() / max(1, num_steps - step)).long(),
            min=1,
        )

        for b in range(B):
            if n_masked[b] == 0:
                continue
            k = min(n_to_unmask[b].item(), n_masked[b].item())
            _, top_idx = confidence[b].topk(k)
            input_ids[b, T_p + top_idx] = pred_tokens[b, top_idx]
            is_masked[b, top_idx] = False

    if is_masked.any():
        outputs = model(input_ids=input_ids)
        final_pred = outputs.logits[:, T_p:, :].argmax(dim=-1)
        input_ids[:, T_p:][is_masked] = final_pred[is_masked]

    return input_ids[:, T_p:]


@torch.no_grad()
def generate_onestep_mask(model, prompt_ids, response_length, mask_token_id):
    """One-step with [MASK]: single forward pass, argmax."""
    device = prompt_ids.device
    B = prompt_ids.shape[0]

    response_ids = torch.full((B, response_length), mask_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    T_p = prompt_ids.shape[1]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, T_p:, :]
    pred_tokens = logits.argmax(dim=-1)

    return pred_tokens


@torch.no_grad()
def generate_onestep_noise(model, prompt_ids, response_length):
    """One-step with Gaussian noise in embedding space."""
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    D = model.get_input_embeddings().weight.shape[1]

    prompt_emb = model.get_input_embeddings()(prompt_ids)
    noise = torch.randn(B, response_length, D, device=device, dtype=prompt_emb.dtype)
    input_emb = torch.cat([prompt_emb, noise], dim=1)
    T_p = prompt_ids.shape[1]

    outputs = model(inputs_embeds=input_emb)
    logits = outputs.logits[:, T_p:, :]
    pred_tokens = logits.argmax(dim=-1)

    return pred_tokens


def evaluate_generation(pred_ids, gt_text, tokenizer):
    """Compute all meaningful metrics for one generation."""
    gen_text = tokenizer.decode(pred_ids[0], skip_special_tokens=True)

    bleu = compute_bleu(gt_text, gen_text)
    distinct = compute_distinct(gen_text)
    rep_ratio = compute_repetition_ratio(gen_text)

    return {
        "text": gen_text,
        "bleu_1": bleu["bleu_1"],
        "bleu_2": bleu["bleu_2"],
        "bleu": bleu["bleu"],
        "distinct_1": distinct["distinct_1"],
        "distinct_2": distinct["distinct_2"],
        "rep_ratio": rep_ratio,
    }


def main(model_name: str, steps_list=(10, 50, 128), num_examples=10, response_length=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    mask_id = get_mask_token_id(model, tokenizer)
    if mask_id is None:
        logger.error("CRITICAL: No mask_token_id found!")
        return

    logger.info(f"Mask token ID: {mask_id} (token: '{tokenizer.decode([mask_id])}')")

    logger.info("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    all_results = {}
    method_order = []

    for n_steps in steps_list:
        key = f"mask_{n_steps}step"
        all_results[key] = []
        method_order.append(key)
    all_results["mask_1step"] = []
    method_order.append("mask_1step")
    all_results["noise_1step"] = []
    method_order.append("noise_1step")

    for idx in range(num_examples):
        example = ds[idx]
        prompt = example["instruction"]
        if example.get("input", ""):
            prompt = prompt + "\n" + example["input"]
        gt_text = example["output"]

        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
        gt_ids = tokenizer(gt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

        T_r = min(response_length, gt_ids.shape[1])
        if T_r < 4:
            continue

        logger.info(f"\n{'='*70}")
        logger.info(f"Example {idx+1}: '{prompt[:80]}'")
        logger.info(f"GT: {gt_text[:120]}")

        for n_steps in steps_list:
            key = f"mask_{n_steps}step"
            pred = generate_multistep_mask(model, p_ids, T_r, mask_id, n_steps)
            res = evaluate_generation(pred, gt_text, tokenizer)
            all_results[key].append(res)
            logger.info(f"  [{n_steps:>3}-step MASK ] bleu={res['bleu']:.3f} b1={res['bleu_1']:.3f} d1={res['distinct_1']:.2f} rep={res['rep_ratio']:.2f} → {res['text'][:100]}")

        key = "mask_1step"
        pred = generate_onestep_mask(model, p_ids, T_r, mask_id)
        res = evaluate_generation(pred, gt_text, tokenizer)
        all_results[key].append(res)
        logger.info(f"  [  1-step MASK ] bleu={res['bleu']:.3f} b1={res['bleu_1']:.3f} d1={res['distinct_1']:.2f} rep={res['rep_ratio']:.2f} → {res['text'][:100]}")

        key = "noise_1step"
        pred = generate_onestep_noise(model, p_ids, T_r)
        res = evaluate_generation(pred, gt_text, tokenizer)
        all_results[key].append(res)
        logger.info(f"  [  1-step NOISE] bleu={res['bleu']:.3f} b1={res['bleu_1']:.3f} d1={res['distinct_1']:.2f} rep={res['rep_ratio']:.2f} → {res['text'][:100]}")

    # --- Summary ---
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY (averaged over all examples)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':<20} {'BLEU':>6} {'BLEU-1':>7} {'BLEU-2':>7} {'Dist-1':>7} {'Dist-2':>7} {'RepRatio':>9}")
    logger.info(f"{'-'*20} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*9}")

    for key in method_order:
        results = all_results[key]
        if not results:
            continue
        n = len(results)
        avg = lambda field: sum(r[field] for r in results) / n
        logger.info(
            f"{key:<20} {avg('bleu'):>6.3f} {avg('bleu_1'):>7.3f} {avg('bleu_2'):>7.3f} "
            f"{avg('distinct_1'):>7.3f} {avg('distinct_2'):>7.3f} {avg('rep_ratio'):>9.3f}"
        )

    logger.info(f"\n  BLEU:      higher = more overlap with GT (tolerates different wording)")
    logger.info(f"  Dist-1/2:  higher = more diverse vocabulary (1.0 = all unique)")
    logger.info(f"  RepRatio:  lower = less repetition (0.0 = no consecutive repeats)")
    logger.info(f"\nMask token ID: {mask_id} | Model: {model_name}")
    logger.info(f"Response length: {response_length} | Num examples: {num_examples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 50, 128])
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--response_length", type=int, default=64)
    args = parser.parse_args()

    main(
        model_name=args.model_name,
        steps_list=args.steps,
        num_examples=args.num_examples,
        response_length=args.response_length,
    )
