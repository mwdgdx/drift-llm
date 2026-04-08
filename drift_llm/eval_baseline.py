"""
Baseline evaluation: test LLaDA with different inference strategies.

Uses actual Alpaca dataset (same as training) so format is consistent.

Compares:
  1. Multi-step with [MASK] (standard LLaDA inference)
  2. One-step with [MASK]
  3. One-step with Gaussian noise (our current approach)

Usage:
    python eval_baseline.py --model_name ./models/LLaDA-8B-Base
    python eval_baseline.py --model_name ./models/LLaDA-8B-Base --steps 10 50 128 --num_examples 20
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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

    generated_ids = input_ids[:, T_p:]

    final_outputs = model(input_ids=input_ids)
    final_logits = final_outputs.logits[:, T_p:, :]

    return generated_ids, final_logits


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

    return pred_tokens, logits


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

    return pred_tokens, logits


def compute_metrics(pred_tokens, logits, gt_ids):
    T = min(pred_tokens.shape[1], gt_ids.shape[1])
    pred = pred_tokens[:, :T]
    gt = gt_ids[:, :T]
    log = logits[:, :T, :]

    acc = (pred == gt).float().mean().item()
    ce = F.cross_entropy(log.reshape(-1, log.size(-1)), gt.reshape(-1)).item()
    return acc, ce


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
    if mask_id is not None:
        logger.info(f"Mask token ID: {mask_id} (token: '{tokenizer.decode([mask_id])}')")
    else:
        logger.error("CRITICAL: No mask_token_id found! LLaDA model config might be wrong.")
        logger.info("Checking model config for any mask-related fields...")
        for k, v in vars(model.config).items():
            if "mask" in k.lower():
                logger.info(f"  config.{k} = {v}")
        return

    # --- Load Alpaca dataset (same as training) ---
    logger.info("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    all_results = {}

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

        gt_ids = gt_ids[:, :T_r]

        logger.info(f"\n{'='*70}")
        logger.info(f"Example {idx+1}: '{prompt[:80]}...'")
        logger.info(f"GT: {gt_text[:120]}")

        # --- Multi-step with [MASK] ---
        for n_steps in steps_list:
            key = f"mask_{n_steps}step"
            pred, logits = generate_multistep_mask(model, p_ids, T_r, mask_id, n_steps)
            acc, ce = compute_metrics(pred, logits, gt_ids)
            text = tokenizer.decode(pred[0], skip_special_tokens=True)
            all_results.setdefault(key, {"acc": [], "ce": []})
            all_results[key]["acc"].append(acc)
            all_results[key]["ce"].append(ce)
            logger.info(f"  [{n_steps:>3}-step MASK ] acc={acc:.3f} ce={ce:.2f} → {text[:100]}")

        # --- One-step with [MASK] ---
        key = "mask_1step"
        pred, logits = generate_onestep_mask(model, p_ids, T_r, mask_id)
        acc, ce = compute_metrics(pred, logits, gt_ids)
        text = tokenizer.decode(pred[0], skip_special_tokens=True)
        all_results.setdefault(key, {"acc": [], "ce": []})
        all_results[key]["acc"].append(acc)
        all_results[key]["ce"].append(ce)
        logger.info(f"  [  1-step MASK ] acc={acc:.3f} ce={ce:.2f} → {text[:100]}")

        # --- One-step with Gaussian noise ---
        key = "noise_1step"
        pred, logits = generate_onestep_noise(model, p_ids, T_r)
        acc, ce = compute_metrics(pred, logits, gt_ids)
        text = tokenizer.decode(pred[0], skip_special_tokens=True)
        all_results.setdefault(key, {"acc": [], "ce": []})
        all_results[key]["acc"].append(acc)
        all_results[key]["ce"].append(ce)
        logger.info(f"  [  1-step NOISE] acc={acc:.3f} ce={ce:.2f} → {text[:100]}")

    # --- Summary ---
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY (averaged over all examples)")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':<20} {'Token Acc':>10} {'CE Loss':>10}")
    logger.info(f"{'-'*20} {'-'*10} {'-'*10}")

    for key in ["mask_1step"] + [f"mask_{s}step" for s in sorted(steps_list)] + ["noise_1step"]:
        if key not in all_results:
            continue
        avg_acc = sum(all_results[key]["acc"]) / len(all_results[key]["acc"])
        avg_ce = sum(all_results[key]["ce"]) / len(all_results[key]["ce"])
        logger.info(f"{key:<20} {avg_acc:>10.4f} {avg_ce:>10.2f}")

    logger.info(f"\nMask token ID used: {mask_id}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Response length: {response_length}")
    logger.info(f"Num examples: {num_examples}")


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
