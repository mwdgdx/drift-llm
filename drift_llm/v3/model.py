"""
v3 EmbeddingGenerator: unconditional noise → continuous embeddings.

Bidirectional Transformer that maps Gaussian noise to continuous embedding
sequences in a reference model's word-embedding space (e.g. GPT-2 768-dim).
Decode to tokens via nearest-neighbor cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Positional encoding (RoPE) ----------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ---------- Transformer components ----------

class Attention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.training else 0.0
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.ff(self.ln2(x))
        return x


# ---------- Generator ----------

class EmbeddingGenerator(nn.Module):
    """Unconditional generator: noise → continuous embeddings."""

    def __init__(self, emb_dim=768, d_model=768, n_layers=8, n_heads=8,
                 seq_len=32, noise_dim=None, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.seq_len = seq_len
        noise_dim = noise_dim or d_model

        self.noise_proj = nn.Sequential(
            nn.Linear(noise_dim, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.rotary = RotaryEmbedding(d_model // n_heads, max_seq_len=seq_len)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_proj = (
            nn.Linear(d_model, emb_dim, bias=False) if d_model != emb_dim
            else nn.Identity()
        )
        self.pos_offset = nn.Parameter(torch.randn(seq_len, emb_dim) * 0.02)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, noise, body_scale=1.0):
        """
        Args:
            noise: [B, seq_len, noise_dim]
            body_scale: float, scales transformer body output (0=pos_offset only)
        Returns:
            embeddings: [B, seq_len, emb_dim]
        """
        h = self.noise_proj(noise)
        h = self.drop(h)
        cos, sin = self.rotary(self.seq_len, device=h.device)
        for block in self.blocks:
            h = block(h, cos, sin)
        body = self.output_proj(self.ln_f(h))
        return body_scale * body + self.pos_offset

    @torch.no_grad()
    def decode_to_tokens(self, embeddings, vocab_embeddings):
        """Nearest-neighbor cosine decode: embeddings → token IDs."""
        e = F.normalize(embeddings, dim=-1)
        v = F.normalize(vocab_embeddings, dim=-1)
        return (e @ v.T).argmax(dim=-1)
