#!/usr/bin/env python3
"""DGG Latent: Stock-Compliant Loss Formulations and Coupling Functions.

Implements verified loss components for DiffusionGemma continuous latent training:
1. Multi-modal Distributional Cross-Entropy with rank-1 mode crash-guard.
2. Latent Feature Loss (SmoothL1 + Cosine distance in R^{256 x 1152} or R^{256 x 384}).
3. Unclamped Pixel Reconstruction Loss through differentiable decoders.
4. Per-Example Noise Reliability Weighting (preserving Jensen's inequality).
5. Balanced Slot Loss Weighting (0.5 changed / 0.5 unchanged slots).
6. PhaseController for training condition annealing schedules.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_W_THRESH = 0.25
DEFAULT_COS_WEIGHT = 2.0
DEFAULT_SMOOTH_L1_BETA = 1.0
DEFAULT_LAMBDA_PIXEL = 0.10


def distribution_ce(
    logits: torch.Tensor,
    mode_ids: torch.Tensor,
    mode_weights: torch.Tensor,
    w_thresh: float = DEFAULT_W_THRESH,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multi-modal distribution cross-entropy over top-k vocabulary modes with crash-guard.

    Supervises continuous probability chords over discrete vocabulary tokens:
        q_tilde_{i, k} = normalized(w_{i, k} * 1[w_{i, k} >= w_thresh])
        with rank-1 mode crash-guard (mask[..., 0] = 1.0 always preserved).

    Args:
        logits: (B, L, V) predicted vocabulary logits.
        mode_ids: (B, L, K) int32 / int64 tensor of mode token IDs.
        mode_weights: (B, L, K) mode probabilities or salience scores.
        w_thresh: relative threshold for filtering low-confidence modes.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss or (B, L) if reduction='none'.
    """
    mode_ids = mode_ids.long()
    mode_weights = mode_weights.float()

    w_max = mode_weights.max(dim=-1, keepdim=True).values.clamp(min=1e-6)
    w_norm = mode_weights / w_max

    mask = (w_norm >= w_thresh).float()
    mask[..., 0] = 1.0  # Crash guard: Rank-1 mode is ALWAYS preserved

    w_masked = w_norm * mask
    w_sum = w_masked.sum(dim=-1, keepdim=True).clamp(min=1e-5)
    q_tilde = w_masked / w_sum

    log_probs = F.log_softmax(logits.float(), dim=-1)
    gathered_log_probs = torch.gather(log_probs, dim=-1, index=mode_ids)
    loss_per_slot = -(q_tilde * gathered_log_probs).sum(dim=-1)

    if reduction == "mean":
        return loss_per_slot.mean()
    elif reduction == "sum":
        return loss_per_slot.sum()
    elif reduction == "none":
        return loss_per_slot
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def loss_hv(
    pred_hv: torch.Tensor,
    target_hv: torch.Tensor,
    cos_weight: float = DEFAULT_COS_WEIGHT,
    beta: float = DEFAULT_SMOOTH_L1_BETA,
    reduction: str = "mean",
) -> torch.Tensor:
    """Continuous latent space loss combining Smooth L1 and Cosine Distance.

    L_{hv} = SmoothL1(pred, target) + cos_weight * (1.0 - CosineSimilarity(pred, target))

    Args:
        pred_hv: Predicted latent tensor (B, L, D) or (B, D).
        target_hv: Target latent tensor (B, L, D) or (B, D).
        cos_weight: Weight for cosine distance penalty.
        beta: Smooth L1 transition threshold.
        reduction: 'mean', 'sum', or 'none'.
    """
    p = pred_hv.float()
    t = target_hv.float()

    diff_l1 = F.smooth_l1_loss(p, t, beta=beta, reduction="none").mean(dim=-1)
    cos_sim = F.cosine_similarity(p, t, dim=-1)
    loss_item = diff_l1 + float(cos_weight) * (1.0 - cos_sim)

    if reduction == "mean":
        return loss_item.mean()
    elif reduction == "sum":
        return loss_item.sum()
    elif reduction == "none":
        return loss_item
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def loss_pixel(
    pred_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Unclamped pixel reconstruction loss through differentiable image decoder.

    CRITICAL: Computed directly on raw decoder predictions without applying
    .clamp(0.0, 1.0) prior to loss computation. Clamping zeros out analytical
    gradients on out-of-range pixels, stalling optimization.
    """
    p = pred_rgb.float()
    t = target_rgb.float()
    mse = F.mse_loss(p, t, reduction="none").mean(dim=(-3, -2, -1))

    if reduction == "mean":
        return mse.mean()
    elif reduction == "sum":
        return mse.sum()
    elif reduction == "none":
        return mse
    else:
        raise ValueError(f"Unknown reduction: {reduction!r}")


def per_example_reliability_weight(
    t_vec: torch.Tensor,
    canvas_mode: str = "single",
    t2_vec: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute per-example D3PM reliability weight preserving Jensen's inequality.

    Evaluated strictly per example:
        Single-canvas: w_i = clamp(1.0 - t_i, 0.0, 1.0)
        Two-canvas:    w_i = clamp(1.0 - t1_i, 0.0, 1.0) * clamp(1.0 - t2_i, 0.0, 1.0)

    NEVER collapses vector t into scalar t.mean() before clamping.
    """
    t1 = t_vec.float()
    w1 = torch.clamp(1.0 - t1, 0.0, 1.0)
    if canvas_mode == "two":
        if t2_vec is None:
            raise ValueError("t2_vec is required for two-canvas reliability weighting")
        w2 = torch.clamp(1.0 - t2_vec.float(), 0.0, 1.0)
        return w1 * w2
    return w1


def balanced_slot_loss(
    pred_vecs: torch.Tensor,
    target_vecs: torch.Tensor,
    changed_mask: torch.Tensor,
    weight_changed: float = 0.5,
    weight_unchanged: float = 0.5,
    cos_weight: float = 1.0,
) -> torch.Tensor:
    """Balanced slot loss weighting across changed and unchanged latent positions.

    Prevents identity/copy collapse in structured graph refactoring tasks where
    most slots remain unchanged (e.g., >95% unchanged slots).

    Args:
        pred_vecs: (B, L, D) predicted slot vectors.
        target_vecs: (B, L, D) target slot vectors.
        changed_mask: (B, L) boolean or int mask (1 = changed slot, 0 = unchanged).
        weight_changed: Total loss weight allocated to changed slots (default 0.5).
        weight_unchanged: Total loss weight allocated to unchanged slots (default 0.5).
        cos_weight: Weight on cosine similarity component.
    """
    p = pred_vecs.float()
    t = target_vecs.float()
    mask = changed_mask.bool()

    # Per-slot loss
    diff = F.smooth_l1_loss(p, t, beta=1.0, reduction="none").mean(dim=-1)
    cos = F.cosine_similarity(p, t, dim=-1)
    slot_loss = diff + cos_weight * (1.0 - cos)  # (B, L)

    # Aggregate changed slots
    n_changed = mask.sum().float().clamp(min=1.0)
    loss_c = (slot_loss * mask.float()).sum() / n_changed

    # Aggregate unchanged slots
    unchanged_mask = (~mask).float()
    n_unchanged = unchanged_mask.sum().clamp(min=1.0)
    loss_u = (slot_loss * unchanged_mask).sum() / n_unchanged

    return weight_changed * loss_c + weight_unchanged * loss_u


class PhaseController:
    """Manages loss weights and annealing schedules across training phases.

    Condition A ('none'):
        Fixed stock weights (ce=1.0, enc=1.0, pix=0.10, hv=1.0).
    Condition B ('ce_to_readout'):
        Phase 1: ce=1.0, enc=1.0, pix=0.05, hv=0.05.
        Phase 2: ce=0.1, enc=0.1, pix=1.0,  hv=1.0.
    """

    WEIGHT_KEYS = ("ce", "enc", "pix", "hv")

    def __init__(
        self,
        condition: str = "none",
        total_steps: int = 4824,
        phase1_fraction: float = 0.5,
    ):
        self.condition = condition
        self.total_steps = total_steps
        self.phase1_steps = int(total_steps * phase1_fraction)

    def get_weights(self, step: int) -> Dict[str, float]:
        """Return dict of current loss component weights at step."""
        if self.condition == "none":
            return {"ce": 1.0, "enc": 1.0, "pix": DEFAULT_LAMBDA_PIXEL, "hv": 1.0}

        if self.condition == "ce_to_readout":
            if step < self.phase1_steps:
                return {"ce": 1.0, "enc": 1.0, "pix": 0.05, "hv": 0.05}
            else:
                progress = min(1.0, (step - self.phase1_steps) / max(1, self.total_steps - self.phase1_steps))
                ce_val = 1.0 - 0.9 * progress
                readout_val = 0.05 + 0.95 * progress
                return {
                    "ce": float(ce_val),
                    "enc": float(ce_val),
                    "pix": float(readout_val * DEFAULT_LAMBDA_PIXEL),
                    "hv": float(readout_val),
                }

        return {"ce": 1.0, "enc": 1.0, "pix": DEFAULT_LAMBDA_PIXEL, "hv": 1.0}
