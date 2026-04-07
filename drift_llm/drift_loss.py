"""
Drift loss ported from JAX (drifting/drift_loss.py) to PyTorch.

Particle-based training: computes kernel forces between generated samples,
positive samples (GT), and negative samples (self/other), then regresses
the generator output onto the force-displaced target positions.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def cdist(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise L2 distance. x: [B,N,D], y: [B,M,D] -> [B,N,M]"""
    xydot = torch.einsum("bnd,bmd->bnm", x, y)
    xnorms = torch.einsum("bnd,bnd->bn", x, x)
    ynorms = torch.einsum("bmd,bmd->bm", y, y)
    sq_dist = xnorms.unsqueeze(-1) + ynorms.unsqueeze(-2) - 2 * xydot
    return torch.sqrt(sq_dist.clamp(min=eps))


def drift_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: Optional[torch.Tensor] = None,
    weight_gen: Optional[torch.Tensor] = None,
    weight_pos: Optional[torch.Tensor] = None,
    weight_neg: Optional[torch.Tensor] = None,
    R_list: Tuple[float, ...] = (0.02, 0.05, 0.2),
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Drift loss: kernel-based particle dynamics in feature space.

    Args:
        gen:       [B, G, D]  generated features (G candidates per sample, has grad)
        fixed_pos: [B, P, D]  positive features (GT, stop_gradient)
        fixed_neg: [B, N, D]  negative features (optional, stop_gradient)
        weight_*:  [B, *]     per-particle weights (optional)
        R_list:    kernel temperature scales

    Returns:
        loss: [B] per-sample loss
        info: dict with diagnostics
    """
    B, C_g, S = gen.shape
    C_p = fixed_pos.shape[1]

    if fixed_neg is None:
        fixed_neg = gen.new_zeros(B, 0, S)
    C_n = fixed_neg.shape[1]

    if weight_gen is None:
        weight_gen = gen.new_ones(B, C_g)
    if weight_pos is None:
        weight_pos = fixed_pos.new_ones(B, C_p)
    if weight_neg is None:
        weight_neg = fixed_neg.new_ones(B, C_n)

    gen = gen.float()
    fixed_pos = fixed_pos.float()
    fixed_neg = fixed_neg.float()
    weight_gen = weight_gen.float()
    weight_pos = weight_pos.float()
    weight_neg = weight_neg.float()

    old_gen = gen.detach()
    targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)      # [B, G+N+P, D]
    targets_w = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)  # [B, G+N+P]

    # --- Compute goal with no gradients (force field is treated as fixed target) ---
    with torch.no_grad():
        info = {}
        dist = cdist(old_gen, targets)  # [B, G, G+N+P]
        weighted_dist = dist * targets_w.unsqueeze(1)
        scale = weighted_dist.mean() / targets_w.mean()
        info["scale"] = scale

        scale_inputs = (scale / (S ** 0.5)).clamp(min=1e-3)
        old_gen_scaled = old_gen / scale_inputs
        targets_scaled = targets / scale_inputs

        dist_normed = dist / scale.clamp(min=1e-3)

        # Mask self-interaction (diagonal of gen block)
        mask_val = 100.0
        diag_mask = torch.eye(C_g, device=gen.device, dtype=gen.dtype)
        block_mask = F.pad(diag_mask, (0, C_n + C_p))  # [G, G+N+P]
        dist_normed = dist_normed + block_mask.unsqueeze(0) * mask_val

        force_across_R = torch.zeros_like(old_gen_scaled)

        for R in R_list:
            logits = -dist_normed / R  # [B, G, G+N+P]

            affinity = torch.softmax(logits, dim=-1)
            aff_transpose = torch.softmax(logits, dim=-2)
            affinity = (affinity * aff_transpose).clamp(min=1e-6).sqrt()
            affinity = affinity * targets_w.unsqueeze(1)

            split_idx = C_g + C_n
            aff_neg = affinity[:, :, :split_idx]     # gen + neg → repulsion
            aff_pos = affinity[:, :, split_idx:]      # pos → attraction

            sum_pos = aff_pos.sum(dim=-1, keepdim=True)
            r_coeff_neg = -aff_neg * sum_pos
            sum_neg = aff_neg.sum(dim=-1, keepdim=True)
            r_coeff_pos = aff_pos * sum_neg

            R_coeff = torch.cat([r_coeff_neg, r_coeff_pos], dim=2)  # [B, G, G+N+P]

            total_force = torch.einsum("biy,byx->bix", R_coeff, targets_scaled)
            total_coeffs = R_coeff.sum(dim=-1)
            total_force = total_force - total_coeffs.unsqueeze(-1) * old_gen_scaled

            f_norm = (total_force ** 2).mean()
            info[f"loss_{R}"] = f_norm

            force_scale = f_norm.clamp(min=1e-8).sqrt()
            force_across_R = force_across_R + total_force / force_scale

        goal_scaled = old_gen_scaled + force_across_R

    # --- Actual loss: MSE between gen and force-displaced goal ---
    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = (diff ** 2).mean(dim=(-1, -2))  # [B]

    info = {k: v.mean() if isinstance(v, torch.Tensor) else v for k, v in info.items()}
    return loss, info
