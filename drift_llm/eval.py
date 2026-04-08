"""
Standalone evaluation: load a checkpoint and evaluate.
No training needed.

Usage:
    python eval.py --model_name ./models/LLaDA-8B-Base --checkpoint ./runs/ce_only/checkpoint-5000
    python eval.py --model_name ./models/LLaDA-8B-Base --checkpoint ./runs/ce_drift/checkpoint-5000
    python eval.py --model_name ./models/LLaDA-8B-Base --checkpoint ./runs/drift_only/checkpoint-5000
"""

import argparse
import logging
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from generator import OneStepGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(model_name: str, checkpoint_path: str, n_eval: int = 50, response_length: int = 64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, trust_remote_code=True,
    ).to(device)
    gen = OneStepGenerator(model)

    if checkpoint_path:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        import os
        state_file = os.path.join(checkpoint_path, "state.pt") if os.path.isdir(checkpoint_path) else checkpoint_path
        state = torch.load(state_file, map_location=device)
        gen.load_state_dict(state["generator"])
        logger.info(f"  Loaded from step {state.get('step', '?')}")
    else:
        logger.info("No checkpoint — evaluating raw pretrained model")

    gen.model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load dataset
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    results = {"token_acc": [], "ce_loss": []}
    samples = []

    with torch.no_grad():
        for i in range(n_eval):
            example = ds[i]
            prompt = example["instruction"]
            if example.get("input", ""):
                prompt = prompt + "\n" + example["input"]
            gt_response = example["output"]

            p_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
            r_ids = tokenizer(gt_response, return_tensors="pt", truncation=True, max_length=response_length, add_special_tokens=False)["input_ids"].to(device)

            if r_ids.shape[1] < 2:
                continue

            T_r = r_ids.shape[1]
            result = gen.forward(prompt_ids=p_ids, response_length=T_r)
            logits = result["logits"]  # [1, T_r, V]
            pred_tokens = logits.argmax(dim=-1)  # [1, T_r]

            acc = (pred_tokens == r_ids).float().mean().item()
            results["token_acc"].append(acc)

            ce = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                r_ids.reshape(-1),
            ).item()
            results["ce_loss"].append(ce)

            if i < 10:
                gen_text = tokenizer.decode(pred_tokens[0], skip_special_tokens=True)
                gt_text = tokenizer.decode(r_ids[0], skip_special_tokens=True)
                samples.append((prompt, gt_text, gen_text))

        # Diversity test on first prompt
        p_ids = tokenizer(ds[0]["instruction"], return_tensors="pt", truncation=True, max_length=256)["input_ids"].to(device)
        div_tokens = [gen.generate(p_ids, response_length)[0] for _ in range(8)]
        div_tokens = torch.stack(div_tokens)
        n = div_tokens.shape[0]
        agree = sum(
            (div_tokens[j] == div_tokens[k]).float().mean().item()
            for j in range(n) for k in range(j + 1, n)
        ) / max(1, n * (n - 1) / 2)

    avg_acc = sum(results["token_acc"]) / len(results["token_acc"])
    avg_ce = sum(results["ce_loss"]) / len(results["ce_loss"])

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS ({checkpoint_path or 'raw pretrained'})")
    logger.info(f"{'='*60}")
    logger.info(f"  Token accuracy:   {avg_acc:.4f}")
    logger.info(f"  CE loss:          {avg_ce:.4f}")
    logger.info(f"  Token agreement:  {agree:.4f}")
    logger.info(f"  Diversity:        {1-agree:.4f}")
    logger.info(f"  Evaluated on:     {len(results['token_acc'])} examples")

    logger.info(f"\n{'='*60}")
    logger.info("SAMPLES")
    logger.info(f"{'='*60}")
    for prompt, gt, generated in samples:
        logger.info(f"\nPrompt:    {prompt[:100]}")
        logger.info(f"GT:        {gt[:200]}")
        logger.info(f"Generated: {generated[:200]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_eval", type=int, default=50)
    parser.add_argument("--response_length", type=int, default=64)
    args = parser.parse_args()

    evaluate(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint,
        n_eval=args.n_eval,
        response_length=args.response_length,
    )
