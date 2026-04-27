"""
Shared Transformer architecture for all training methods.

A bidirectional Transformer that maps [prompt_tokens + noise/mask] → logits.
Used by both Drift and Diffusion-LM baselines for fair comparison.

Default config (~170M params): d=768, 12 layers, 12 heads — matches FLM/MDLM literature.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, T, D]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, D]
        q, k = q.transpose(1, 2), k.transpose(1, 2)  # [B, H, T, D]
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


class TextGenerator(nn.Module):
    """
    Bidirectional Transformer: prompt_tokens + noise → response logits.

    Trained from scratch — noise is a first-class input.
    RoPE for position encoding, no absolute embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        noise_dim: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        noise_dim = noise_dim or d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.rotary = RotaryEmbedding(d_model // n_heads, max_seq_len=max_seq_len)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight  # weight tying

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, prompt_ids, noise, response_ids=None):
        """
        Args:
            prompt_ids: [B, T_p] long
            noise: [B, T_r, noise_dim] float
            response_ids: [B, T_r] long (optional, unused here but kept for API compat)
        Returns:
            logits: [B, T_r, V]
        """
        B, T_p = prompt_ids.shape
        T_r = noise.shape[1]
        T = T_p + T_r

        prompt_emb = self.token_embedding(prompt_ids)
        noise_emb = self.noise_proj(noise)

        h = torch.cat([prompt_emb, noise_emb], dim=1)
        h = self.drop(h)

        cos, sin = self.rotary(T, device=h.device)
        for block in self.blocks:
            h = block(h, cos, sin)

        h = self.ln_f(h)
        logits = self.output_head(h[:, T_p:, :])
        return logits

    def get_embedding_weight(self):
        return self.token_embedding.weight

    @torch.no_grad()
    def generate(self, prompt_ids, response_length, noise=None):
        if noise is None:
            B = prompt_ids.shape[0]
            noise = torch.randn(B, response_length, self.d_model, device=prompt_ids.device)
        logits = self.forward(prompt_ids, noise)
        return logits.argmax(dim=-1)
