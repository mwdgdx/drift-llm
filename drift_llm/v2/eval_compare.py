"""
Unified evaluation: compare Drift vs Diffusion-LM vs CE-only.

Metrics (aligned with FLM paper):
  - Generative Perplexity (Gen. PPL): measured by a frozen GPT-2 Large
  - Entropy: average per-sample unigram entropy (detects repetition/mode collapse)
  - BLEU, Distinct-n, Repetition ratio (additional)

For each model, tests:
  - 1-step generation (Drift's sweet spot)
  - Multi-step generation (Diffusion-LM's advantage)
"""

import argparse
import logging
import math
import os
import sys
from collections import Counter

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, GPT2LMHeadModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v2.transformer import TextGenerator
from v2.data import load_alpaca, make_dataloader, make_owt_dataloader
from v2.diffusion_train import diffusion_sample, cosine_noise_schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Generative Perplexity (GPT-2 Large as judge, same as FLM paper)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_gen_ppl(texts, ppl_model, ppl_tokenizer, device, max_len=1024):
    """Compute generative perplexity using a pretrained LM as judge."""
    total_nll = 0.0
    total_tokens = 0

    for text in texts:
        if not text.strip():
            continue
        enc = ppl_tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 2:
            continue

        outputs = ppl_model(input_ids, labels=input_ids)
        nll = outputs.loss.item() * (input_ids.shape[1] - 1)
        total_nll += nll
        total_tokens += input_ids.shape[1] - 1

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_nll / total_tokens)


def compute_entropy(texts, tokenizer):
    """Average per-sample unigram entropy (same as FLM paper)."""
    entropies = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 2:
            continue
        counter = Counter(tokens)
        total = sum(counter.values())
        ent = -sum((c / total) * math.log2(c / total) for c in counter.values())
        entropies.append(ent)
    return sum(entropies) / max(len(entropies), 1)


# ---------------------------------------------------------------------------
# Legacy metrics
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, tokenizer, device, d_model,
                   n_eval=100, diffusion_steps=50, ppl_model=None, ppl_tokenizer=None):
    model.eval()
    results = {"1step": {"texts": [], "metrics": []}, "multi": {"texts": [], "metrics": []}}
    count = 0

    for batch in loader:
        if count >= n_eval:
            break
        prompt_ids = batch["prompt_ids"].to(device)
        response_ids = batch["response_ids"].to(device)
        response_mask = batch["response_mask"].to(device)
        B, T_r = response_ids.shape

        tokens_1 = model.generate(prompt_ids, T_r)
        tokens_m = diffusion_sample(model, prompt_ids, T_r, d_model,
                                    n_steps=diffusion_steps, device=device)

        for i in range(B):
            if count >= n_eval:
                break
            gt_text = tokenizer.decode(response_ids[i][response_mask[i]], skip_special_tokens=True)
            text_1 = tokenizer.decode(tokens_1[i], skip_special_tokens=True)
            text_m = tokenizer.decode(tokens_m[i], skip_special_tokens=True)

            valid = response_mask[i]
            acc_1 = (tokens_1[i][:T_r][valid] == response_ids[i][valid]).float().mean().item()
            acc_m = (tokens_m[i][:T_r][valid] == response_ids[i][valid]).float().mean().item()

            d1_1, d2_1 = compute_distinct(text_1)
            d1_m, d2_m = compute_distinct(text_m)

            results["1step"]["texts"].append(text_1)
            results["1step"]["metrics"].append({"acc": acc_1, "d1": d1_1, "d2": d2_1, "rep": rep_ratio(text_1)})
            results["multi"]["texts"].append(text_m)
            results["multi"]["metrics"].append({"acc": acc_m, "d1": d1_m, "d2": d2_m, "rep": rep_ratio(text_m)})

            if count < 3:
                prompt_text = tokenizer.decode(prompt_ids[i], skip_special_tokens=True)
                logger.info(f"\nPrompt: {prompt_text[:80]}")
                logger.info(f"GT:      {gt_text[:120]}")
                logger.info(f"1-step:  {text_1[:120]}")
                logger.info(f"Multi:   {text_m[:120]}")

            count += 1

    # Compute Gen. PPL and Entropy for each mode
    for mode in ["1step", "multi"]:
        texts = results[mode]["texts"]
        if ppl_model is not None:
            ppl = compute_gen_ppl(texts, ppl_model, ppl_tokenizer, device)
        else:
            ppl = float("nan")
        entropy = compute_entropy(texts, tokenizer)
        results[mode]["gen_ppl"] = ppl
        results[mode]["entropy"] = entropy

    return results


def avg(lst, key):
    vals = [x[key] for x in lst]
    return sum(vals) / max(len(vals), 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--ppl_model", default="gpt2-large", help="Model for Gen. PPL (default: gpt2-large)")
    p.add_argument("--n_eval", type=int, default=100)
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dataset", type=str, default="owt", choices=["owt", "alpaca"])
    p.add_argument("--wandb_project", default=None, help="WandB project for logging eval results")
    args = p.parse_args()

    if args.labels is None:
        args.labels = [os.path.basename(os.path.dirname(c)) for c in args.checkpoints]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load GPT-2 Large for Gen. PPL evaluation
    logger.info(f"Loading PPL judge model: {args.ppl_model}")
    ppl_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
    ppl_model = GPT2LMHeadModel.from_pretrained(args.ppl_model).to(device)
    ppl_model.eval()

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
            max_seq_len=ckpt_args.get("max_prompt_len", 1) + ckpt_args["max_response_len"],
            dropout=0.0,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        dataset = ckpt_args.get("dataset", "alpaca")
        if dataset == "owt":
            loader = make_owt_dataloader(tokenizer, args.batch_size,
                                         seq_len=ckpt_args["max_response_len"], num_workers=2)
        else:
            ds = load_alpaca(tokenizer, ckpt_args["max_prompt_len"], ckpt_args["max_response_len"])
            loader = make_dataloader(ds, tokenizer, args.batch_size,
                                     ckpt_args["max_prompt_len"], ckpt_args["max_response_len"])

        results = evaluate_model(
            model, loader, tokenizer, device,
            d_model=ckpt_args["d_model"],
            n_eval=args.n_eval,
            diffusion_steps=args.diffusion_steps,
            ppl_model=ppl_model,
            ppl_tokenizer=ppl_tokenizer,
        )
        all_results[label] = results

    # --- Print comparison table ---
    logger.info(f"\n{'='*100}")
    logger.info("COMPARISON TABLE")
    logger.info(f"{'='*100}")
    header = (f"{'Method':<25} {'Mode':<8} {'Gen.PPL':>8} {'Entropy':>8} "
              f"{'TokAcc':>7} {'Dist-1':>7} {'Dist-2':>7} {'Rep':>5}")
    logger.info(header)
    logger.info("-" * len(header))

    for label, results in all_results.items():
        for mode in ["1step", "multi"]:
            r = results[mode]
            mode_label = "1-step" if mode == "1step" else "multi"
            logger.info(
                f"{label:<25} {mode_label:<8} "
                f"{r['gen_ppl']:>8.2f} {r['entropy']:>8.3f} "
                f"{avg(r['metrics'], 'acc'):>7.4f} "
                f"{avg(r['metrics'], 'd1'):>7.3f} {avg(r['metrics'], 'd2'):>7.3f} "
                f"{avg(r['metrics'], 'rep'):>5.3f}"
            )

    logger.info(f"\n(Gen. PPL measured by {args.ppl_model}; lower is better)")
    logger.info("(Entropy: higher is better, data-level ~4.3 LM1B / ~5.4 OWT)")

    # --- Log to WandB ---
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, name="eval_comparison", config=vars(args))

        # Summary table
        columns = ["Method", "Mode", "Gen.PPL", "Entropy", "TokAcc", "Dist-1", "Dist-2", "Rep"]
        table = wandb.Table(columns=columns)

        for label, results in all_results.items():
            for mode in ["1step", "multi"]:
                r = results[mode]
                mode_label = "1-step" if mode == "1step" else "multi"
                table.add_data(
                    label, mode_label,
                    r["gen_ppl"], r["entropy"],
                    avg(r["metrics"], "acc"),
                    avg(r["metrics"], "d1"), avg(r["metrics"], "d2"),
                    avg(r["metrics"], "rep"),
                )
                wandb.summary[f"{label}/{mode_label}/gen_ppl"] = r["gen_ppl"]
                wandb.summary[f"{label}/{mode_label}/entropy"] = r["entropy"]
                wandb.summary[f"{label}/{mode_label}/tok_acc"] = avg(r["metrics"], "acc")
                wandb.summary[f"{label}/{mode_label}/dist1"] = avg(r["metrics"], "d1")
                wandb.summary[f"{label}/{mode_label}/dist2"] = avg(r["metrics"], "d2")
                wandb.summary[f"{label}/{mode_label}/rep"] = avg(r["metrics"], "rep")

        wandb.log({"eval/comparison": table})

        # Log sample generations
        for label, results in all_results.items():
            sample_table = wandb.Table(columns=["text"])
            for text in results["1step"]["texts"][:20]:
                sample_table.add_data(text)
            wandb.log({f"eval/{label}_1step_samples": sample_table})

        wandb.finish()
        logger.info("Eval results logged to WandB.")


if __name__ == "__main__":
    main()
