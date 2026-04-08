"""
Baseline evaluation: test LLaDA with different inference strategies.

Compares:
  1. Multi-step with [MASK] (standard LLaDA, should work)
  2. One-step with [MASK]  (one-step but using mask tokens instead of noise)
  3. One-step with noise   (our current approach)

This tells us: is the problem one-step, or Gaussian noise, or both?

Usage:
    python eval_baseline.py --model_name ./models/LLaDA-8B-Base
    python eval_baseline.py --model_name ./models/LLaDA-8B-Base --steps 10 20 50 128
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


TEST_CASES = [
    ("What is 2+2?", "The answer is 4."),
    ("What are the three primary colors?", "The three primary colors are red, blue, and yellow."),
    ("What is the capital of France?", "The capital of France is Paris."),
    ("How many days in a week?", "There are seven days in a week."),
    ("Who wrote Romeo and Juliet?", "William Shakespeare wrote Romeo and Juliet."),
    ("Name three planets.", "Mars, Jupiter, and Saturn are three planets."),
    ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid."),
    ("What is gravity?", "Gravity is the force that attracts objects toward each other."),
    ("Count from one to five.", "One, two, three, four, five."),
    ("What year did World War II end?", "World War II ended in 1945."),
]


def get_mask_token_id(model, tokenizer):
    """Get the mask token ID for LLaDA."""
    if hasattr(model.config, 'mask_token_id') and model.config.mask_token_id is not None:
        return model.config.mask_token_id
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    logger.warning("No mask_token_id found, using eos_token_id as fallback")
    return tokenizer.eos_token_id


@torch.no_grad()
def generate_multistep_mask(model, prompt_ids, response_length, mask_token_id, num_steps):
    """
    Standard LLaDA multi-step inference:
    1. Start with all [MASK] for response positions
    2. Forward → predict → unmask most confident tokens
    3. Repeat for num_steps
    """
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    T_r = response_length

    response_ids = torch.full((B, T_r), mask_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    T_p = prompt_ids.shape[1]

    is_masked = torch.ones(B, T_r, device=device, dtype=torch.bool)

    for step in range(num_steps):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, T_p:, :]  # [B, T_r, V]

        probs = F.softmax(logits, dim=-1)
        pred_tokens = logits.argmax(dim=-1)  # [B, T_r]
        confidence = probs.max(dim=-1).values  # [B, T_r]

        confidence[~is_masked] = -1.0

        n_masked = is_masked.sum(dim=-1)  # [B]
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

    final_logits = model(input_ids=input_ids).logits[:, T_p:, :]
    remaining = is_masked.any(dim=-1)
    if remaining.any():
        input_ids[:, T_p:][is_masked] = final_logits.argmax(dim=-1)[is_masked]

    return input_ids[:, T_p:], final_logits


@torch.no_grad()
def generate_onestep_mask(model, prompt_ids, response_length, mask_token_id):
    """One-step with [MASK] tokens (no iteration, just one forward pass)."""
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    T_r = response_length

    response_ids = torch.full((B, T_r), mask_token_id, device=device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    T_p = prompt_ids.shape[1]

    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, T_p:, :]
    pred_tokens = logits.argmax(dim=-1)

    return pred_tokens, logits


@torch.no_grad()
def generate_onestep_noise(model, prompt_ids, response_length):
    """One-step with Gaussian noise (our current approach)."""
    device = prompt_ids.device
    B = prompt_ids.shape[0]
    T_r = response_length
    D = model.get_input_embeddings().weight.shape[1]

    prompt_emb = model.get_input_embeddings()(prompt_ids)
    noise = torch.randn(B, T_r, D, device=device, dtype=prompt_emb.dtype)
    input_emb = torch.cat([prompt_emb, noise], dim=1)
    T_p = prompt_ids.shape[1]

    outputs = model(inputs_embeds=input_emb)
    logits = outputs.logits[:, T_p:, :]
    pred_tokens = logits.argmax(dim=-1)

    return pred_tokens, logits


def evaluate_method(name, pred_tokens, logits, gt_ids, tokenizer):
    """Compute metrics for one generation method."""
    T = min(pred_tokens.shape[1], gt_ids.shape[1])
    pred = pred_tokens[:, :T]
    gt = gt_ids[:, :T]
    log = logits[:, :T, :]

    acc = (pred == gt).float().mean().item()
    ce = F.cross_entropy(log.reshape(-1, log.size(-1)), gt.reshape(-1)).item()
    gen_text = tokenizer.decode(pred[0], skip_special_tokens=True)

    return acc, ce, gen_text


def main(model_name: str, steps_list=(1, 10, 50, 128)):
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
    logger.info(f"Mask token ID: {mask_id} ('{tokenizer.decode([mask_id])}')")

    all_results = {}

    for prompt, gt_answer in TEST_CASES:
        p_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        gt_ids = tokenizer(gt_answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        T_r = gt_ids.shape[1]

        if T_r < 2:
            continue

        logger.info(f"\nPrompt: '{prompt}'")
        logger.info(f"GT:     '{gt_answer}'")

        # --- Multi-step with [MASK] ---
        for n_steps in steps_list:
            key = f"mask_{n_steps}step"
            pred, logits = generate_multistep_mask(model, p_ids, T_r, mask_id, n_steps)
            acc, ce, text = evaluate_method(key, pred, logits, gt_ids, tokenizer)
            all_results.setdefault(key, {"acc": [], "ce": []})
            all_results[key]["acc"].append(acc)
            all_results[key]["ce"].append(ce)
            logger.info(f"  [{n_steps:>3}-step MASK]  acc={acc:.3f}  ce={ce:.2f}  → {text[:80]}")

        # --- One-step with [MASK] ---
        key = "mask_1step"
        pred, logits = generate_onestep_mask(model, p_ids, T_r, mask_id)
        acc, ce, text = evaluate_method(key, pred, logits, gt_ids, tokenizer)
        all_results.setdefault(key, {"acc": [], "ce": []})
        all_results[key]["acc"].append(acc)
        all_results[key]["ce"].append(ce)
        logger.info(f"  [  1-step MASK]  acc={acc:.3f}  ce={ce:.2f}  → {text[:80]}")

        # --- One-step with noise ---
        key = "noise_1step"
        pred, logits = generate_onestep_noise(model, p_ids, T_r)
        acc, ce, text = evaluate_method(key, pred, logits, gt_ids, tokenizer)
        all_results.setdefault(key, {"acc": [], "ce": []})
        all_results[key]["acc"].append(acc)
        all_results[key]["ce"].append(ce)
        logger.info(f"  [  1-step NOISE] acc={acc:.3f}  ce={ce:.2f}  → {text[:80]}")

    # --- Summary ---
    logger.info(f"\n{'='*70}")
    logger.info("SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"{'Method':<20} {'Token Acc':>10} {'CE Loss':>10}")
    logger.info(f"{'-'*20} {'-'*10} {'-'*10}")

    for key in sorted(all_results.keys()):
        avg_acc = sum(all_results[key]["acc"]) / len(all_results[key]["acc"])
        avg_ce = sum(all_results[key]["ce"]) / len(all_results[key]["ce"])
        logger.info(f"{key:<20} {avg_acc:>10.4f} {avg_ce:>10.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=[10, 50, 128])
    args = parser.parse_args()

    main(model_name=args.model_name, steps_list=args.steps)
