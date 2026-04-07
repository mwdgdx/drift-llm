"""
Bridges between Generator logits/hidden-states and Feature Encoder input space.

Provides differentiable paths from the Generator's continuous output
to vectors that a frozen Feature Encoder can process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_bridge(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
    tau: float = 1.0,
    hard: bool = False,
) -> torch.Tensor:
    """
    Gumbel-Softmax → soft embedding.

    logits:           [*, V]
    embedding_weight: [V, D]  (Feature Encoder's embedding matrix)
    returns:          [*, D]  soft embedding in Feature Encoder's input space

    Gradient flows: loss → soft_emb → soft_probs → logits → Generator
    """
    soft_probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)  # [*, V]
    soft_emb = soft_probs @ embedding_weight                           # [*, D]
    return soft_emb


def softmax_bridge(
    logits: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """
    Plain softmax → soft embedding (no Gumbel noise).
    Useful for evaluation or when noise comes from Generator input.

    logits:           [*, V]
    embedding_weight: [V, D]
    returns:          [*, D]
    """
    probs = F.softmax(logits, dim=-1)  # [*, V]
    soft_emb = probs @ embedding_weight  # [*, D]
    return soft_emb


class MLPBridge(nn.Module):
    """
    Learned projection from Generator hidden states to Feature Encoder embedding space.

    Alternative to Gumbel-Softmax: directly maps hidden_states to vectors
    that the Feature Encoder can process. Trained via:
      1) drift loss (end-to-end through frozen Feature Encoder)
      2) alignment loss (MSE to GT token embeddings, keeps output in-distribution)
    """

    def __init__(self, d_gen: int, d_feat: int, hidden_mult: int = 2):
        super().__init__()
        d_hidden = min(d_gen, d_feat) * hidden_mult
        self.proj = nn.Sequential(
            nn.Linear(d_gen, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_feat),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [B, T, D_gen] → projected: [B, T, D_feat]"""
        return self.proj(hidden_states)

    def alignment_loss(
        self,
        hidden_states: torch.Tensor,
        gt_token_ids: torch.Tensor,
        feat_embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auxiliary loss: keep MLP output close to real token embeddings.

        hidden_states:        [B, T, D_gen]
        gt_token_ids:         [B, T]
        feat_embedding_weight: [V, D_feat]
        returns: scalar loss
        """
        projected = self.forward(hidden_states)                    # [B, T, D_feat]
        gt_emb = feat_embedding_weight[gt_token_ids]               # [B, T, D_feat]
        return F.mse_loss(projected, gt_emb.detach())
