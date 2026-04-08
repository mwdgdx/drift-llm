"""
Shared Transformer architecture for all training methods.

A small bidirectional Transformer that maps [prompt_tokens + noise/mask] → logits.
Used by both Drift and Diffusion-LM baselines for fair comparison.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, positions):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=positions.device, dtype=torch.float32) * -emb)
        emb = positions.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class TextGenerator(nn.Module):
    """
    Bidirectional Transformer: prompt_tokens + noise → response logits.

    This model is trained from scratch — noise is a first-class input,
    not a hack on a pretrained model.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        noise_dim: int = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        noise_dim = noise_dim or d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.pos_emb = SinusoidalPosEmb(d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)
        self.output_head.weight = self.token_embedding.weight

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
            noise: [B, T_r, noise_dim] float (Gaussian noise for response positions)
            response_ids: [B, T_r] long (optional, for teacher forcing / CE loss)

        Returns:
            logits: [B, T_r, V]
        """
        B, T_p = prompt_ids.shape
        T_r = noise.shape[1]

        prompt_emb = self.token_embedding(prompt_ids)
        noise_emb = self.noise_proj(noise)

        positions = torch.arange(T_p + T_r, device=prompt_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(positions)

        h = torch.cat([prompt_emb, noise_emb], dim=1)
        h = self.drop(h + pos_emb)

        for block in self.blocks:
            h = block(h)

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
