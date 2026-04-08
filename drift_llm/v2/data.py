"""Shared data loading for all training methods."""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


def load_alpaca(tokenizer, max_prompt_len=128, max_response_len=64):
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    def tokenize_fn(example):
        prompt = example["instruction"]
        if example.get("input", ""):
            prompt = prompt + "\n" + example["input"]
        response = example["output"]

        p_ids = tokenizer(
            prompt, truncation=True, max_length=max_prompt_len,
            add_special_tokens=True, return_tensors=None,
        )["input_ids"]
        r_ids = tokenizer(
            response, truncation=True, max_length=max_response_len,
            add_special_tokens=False, return_tensors=None,
        )["input_ids"]
        return {"prompt_ids": p_ids, "response_ids": r_ids}

    ds = ds.map(tokenize_fn, num_proc=4, desc="Tokenizing")
    ds = ds.filter(lambda x: len(x["response_ids"]) >= 8)
    return ds


def collate_fn(batch, pad_token_id, max_prompt_len, max_response_len):
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


def make_dataloader(ds, tokenizer, batch_size, max_prompt_len=128, max_response_len=64):
    from functools import partial
    fn = partial(collate_fn, pad_token_id=tokenizer.pad_token_id,
                 max_prompt_len=max_prompt_len, max_response_len=max_response_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=fn, drop_last=True)
