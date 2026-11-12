#!/usr/bin/env python3
"""DGG Codebook: Statistical Codebook De-Biasing and RoPE Carrier Removal.

Implements token frequency discounting, background mode suppression, and
the critical 1D RoPE positional carrier subtraction that eliminates prompt-invariant
visual regression to the mean.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class FrequencyDebiaser:
    """Discounts over-represented corpus tokens to prevent vocabulary domination.

    Computes inverse frequency multipliers:
        multiplier(v) = (freq(v) / max_freq + eps) ** (-gamma)
    """

    def __init__(
        self,
        vocab_size: int = 262144,
        gamma: float = 0.25,
        eps: float = 1e-4,
        alpha: Optional[float] = None,
    ):
        self.vocab_size = vocab_size
        self.gamma = alpha if alpha is not None else gamma
        self.eps = eps
        self.counts = torch.zeros(vocab_size, dtype=torch.float64)

    def update(self, token_ids: torch.Tensor) -> None:
        """Update token counts with observed tokens."""
        flat = token_ids.reshape(-1).long()
        valid = flat[(flat >= 0) & (flat < self.vocab_size)]
        unique, counts = torch.unique(valid, return_counts=True)
        self.counts[unique] += counts.to(self.counts.dtype)

    def get_multipliers(self, mode_ids: torch.Tensor) -> torch.Tensor:
        """Return de-biasing multipliers for the specified mode IDs."""
        total = self.counts.sum().clamp(min=1.0)
        freqs = (self.counts / total).float().to(mode_ids.device)

        gathered_freqs = freqs[mode_ids.long()]
        max_f = gathered_freqs.max().clamp(min=1e-8)
        norm_f = gathered_freqs / max_f
        multipliers = (norm_f + self.eps) ** (-self.gamma)
        return multipliers.clamp(0.1, 5.0)

    def __call__(self, dist: torch.Tensor) -> torch.Tensor:
        """Apply frequency de-biasing to a distribution over vocabulary."""
        v = dist.shape[-1]
        mode_ids = torch.arange(v, device=dist.device)
        mults = self.get_multipliers(mode_ids)
        adjusted = dist * mults
        return adjusted / adjusted.sum(dim=-1, keepdim=True).clamp(min=1e-12)


class BackgroundSuppression:
    """Detects and suppresses redundant background modes on detailed patches."""

    def __init__(self, bg_token_id: int = 0, penalty: float = 0.5):
        self.bg_token_id = bg_token_id
        self.penalty = penalty

    def suppress(
        self,
        mode_ids: torch.Tensor,
        weights: torch.Tensor,
        spatial_variance: torch.Tensor,
        var_threshold: float = 0.1,
    ) -> torch.Tensor:
        """Downweight background token when local spatial variance indicates active structure."""
        is_bg = (mode_ids == self.bg_token_id)
        is_active = (spatial_variance > var_threshold).unsqueeze(-1)  # (B, L, 1)

        adjusted = weights.clone()
        suppression_mask = is_bg & is_active
        adjusted[suppression_mask] *= self.penalty
        return adjusted


class RoPECarrierDebiaser(nn.Module):
    """Fits and subtracts prompt-invariant 1D RoPE carrier from decoder hidden states.

    In DiffusionGemma 26B-A4B, the raw decoder hidden states across the 256 canvas slots
    are >90% dominated by a static 1D RoPE positional wave mu_pos[0..255].
    Without subtracting mu_pos, readout heads learn to output the mean-regressed carrier,
    causing visual collapse.

    mu_pos[i] = (1 / N) * sum_{n=1}^N h_n[i]
    """

    def __init__(self, canvas_len: int = 256, embed_dim: int = 2816):
        super().__init__()
        self.canvas_len = canvas_len
        self.embed_dim = embed_dim
        self.register_buffer("carrier", torch.zeros(canvas_len, embed_dim))
        self.register_buffer("samples_seen", torch.tensor(0, dtype=torch.long))

    def fit_batch(self, h_dec: torch.Tensor) -> None:
        """Accumulate running mean of 256-slot hidden states across training batches."""
        # h_dec: (B, 256, D)
        with torch.no_grad():
            b = h_dec.shape[0]
            batch_mean = h_dec.detach().float().mean(dim=0)  # (256, D)

            new_total = self.samples_seen + b
            w_old = self.samples_seen.float() / new_total.float()
            w_new = float(b) / new_total.float()

            self.carrier.copy_(w_old * self.carrier + w_new * batch_mean.to(self.carrier.device))
            self.samples_seen.copy_(new_total)

    def subtract(self, h_dec: torch.Tensor) -> torch.Tensor:
        """Subtract fitted 1D RoPE carrier from decoder hidden states."""
        # h_dec: (B, 256, D)
        return h_dec - self.carrier.to(h_dec.device, dtype=h_dec.dtype).unsqueeze(0)

    def save(self, path: str) -> None:
        """Save fitted carrier to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "carrier": self.carrier.cpu(),
                "samples_seen": self.samples_seen.cpu(),
                "canvas_len": self.canvas_len,
                "embed_dim": self.embed_dim,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load fitted carrier from disk."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.carrier.copy_(state["carrier"])
        self.samples_seen.copy_(state["samples_seen"])
