"""
Feature Encoder wrapper — the frozen model that provides semantic feature space
for drift loss, analogous to frozen MAE in image Drifting.

Supports multiple backends:
  1. "embedding" — just use Generator's own embedding matrix (Level 1, no extra model)
  2. "transformer" — a frozen HuggingFace encoder model (e.g., BERT, E5)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from einops import rearrange


class EmbeddingFeatureEncoder(nn.Module):
    """
    Level 1: No external model. Features = embedding projections.
    gen_feat = softmax(logits) @ E   (or Gumbel-Softmax @ E)
    gt_feat  = E[gt_tokens]

    This provides per-token features with basic word-level semantics.
    No context, no sequence-level information, but zero extra cost.
    """

    def __init__(self, embedding_weight: torch.Tensor):
        """embedding_weight: [V, D] from the Generator's embedding layer."""
        super().__init__()
        self.register_buffer("weight", embedding_weight.detach().clone())

    @property
    def hidden_dim(self) -> int:
        return self.weight.shape[1]

    def encode_soft(self, soft_probs: torch.Tensor) -> torch.Tensor:
        """soft_probs: [B, T, V] → [B, T, D]"""
        return soft_probs @ self.weight

    def encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B, T] → [B, T, D]"""
        return self.weight[token_ids]


class TransformerFeatureEncoder(nn.Module):
    """
    Level 2+: Frozen HuggingFace encoder (e.g., BERT, E5, GTE).
    Provides context-aware features at multiple layers.
    """

    def __init__(self, model_name: str, layers: Optional[List[int]] = None):
        """
        Args:
            model_name: HuggingFace model name (e.g., "intfloat/e5-large-v2")
            layers: which hidden layers to extract (e.g., [6, 12, 18, 24]).
                    None = last layer only.
        """
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.layers = layers

        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @property
    def hidden_dim(self) -> int:
        return self.encoder.config.hidden_size

    @property
    def embedding_weight(self) -> torch.Tensor:
        return self.encoder.get_input_embeddings().weight

    def forward_from_embeds(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """
        Forward pass from continuous embeddings (for Gen side via Gumbel-Softmax).
        inputs_embeds: [B, T, D]
        returns: hidden_states [B, T, D]  (last layer)
        """
        outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            output_hidden_states=self.layers is not None,
        )
        if self.layers is not None:
            return self._extract_multiscale(outputs.hidden_states)
        return outputs.last_hidden_state

    def forward_from_tokens(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass from discrete tokens (for GT side).
        token_ids: [B, T]
        returns: hidden_states [B, T, D]  (last layer)
        """
        outputs = self.encoder(
            input_ids=token_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.layers is not None,
        )
        if self.layers is not None:
            return self._extract_multiscale(outputs.hidden_states)
        return outputs.last_hidden_state

    def _extract_multiscale(self, all_hidden_states) -> torch.Tensor:
        """Average selected layers for multi-scale features."""
        selected = [all_hidden_states[i] for i in self.layers]
        return torch.stack(selected, dim=0).mean(dim=0)


def extract_multiscale_features(
    hidden: torch.Tensor,
    chunk_sizes: List[int] = (32,),
) -> dict:
    """
    From hidden_states [B, T, D], extract multi-scale drift features:
      - per_token: [B, T, D]
      - chunk_{s}: [B, T//s, D]  for each chunk_size s
      - global:    [B, 1, D]

    Returns dict of {scale_name: tensor}.
    Analogous to image Drifting's patch_mean_2, patch_mean_4, global_mean.
    """
    B, T, D = hidden.shape
    features = {"per_token": hidden}

    for s in chunk_sizes:
        if T >= s:
            n_chunks = T // s
            truncated = hidden[:, :n_chunks * s, :]
            chunked = rearrange(truncated, "b (n s) d -> b n s d", s=s)
            features[f"chunk_{s}"] = chunked.mean(dim=2)

    features["global"] = hidden.mean(dim=1, keepdim=True)  # [B, 1, D]
    return features
