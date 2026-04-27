"""
Shared data loading for all training methods.

Supports:
  - Alpaca (instruction → response, for small/debug experiments)
  - OpenWebText (unconditional LM, packed sequences, for scaled experiments)
"""

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset


# ---------------------------------------------------------------------------
# Alpaca: instruction-following (conditional generation)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# OpenWebText: unconditional LM with sequence packing (for scaled experiments)
# ---------------------------------------------------------------------------

class PackedOWTDataset(IterableDataset):
    """
    Streams OpenWebText and packs tokens into fixed-length sequences.
    For unconditional LM: the entire sequence is the "response" (no prompt).
    We use a short dummy prompt (BOS) so the Generator API stays consistent.
    """

    def __init__(self, tokenizer, seq_len=128, buffer_size=100_000, seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        import itertools
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed + (worker_info.id if worker_info else 0)

        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True,
                          trust_remote_code=True)
        ds = ds.shuffle(seed=worker_seed, buffer_size=self.buffer_size)

        token_buffer = []
        eos_id = self.tokenizer.eos_token_id or 0

        for example in itertools.cycle(ds):
            text = example.get("text", "")
            if not text:
                continue
            ids = self.tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
            token_buffer.extend(ids)
            token_buffer.append(eos_id)

            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[:self.seq_len]
                token_buffer = token_buffer[self.seq_len:]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                }


def owt_collate_fn(batch, bos_token_id):
    """
    For unconditional LM: prompt is just [BOS], response is the packed sequence.
    """
    B = len(batch)
    seq_len = batch[0]["input_ids"].shape[0]

    prompt_ids = torch.full((B, 1), bos_token_id, dtype=torch.long)
    response_ids = torch.stack([item["input_ids"] for item in batch])
    response_mask = torch.ones(B, seq_len, dtype=torch.bool)

    return {"prompt_ids": prompt_ids, "response_ids": response_ids, "response_mask": response_mask}


def make_owt_dataloader(tokenizer, batch_size, seq_len=128, num_workers=4):
    from functools import partial
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    ds = PackedOWTDataset(tokenizer, seq_len=seq_len)
    fn = partial(owt_collate_fn, bos_token_id=bos_id)
    return DataLoader(ds, batch_size=batch_size, collate_fn=fn,
                      num_workers=num_workers, pin_memory=True, drop_last=True)
