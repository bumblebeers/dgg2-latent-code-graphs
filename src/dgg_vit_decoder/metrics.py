#!/usr/bin/env python3
"""DGG ViT Decoder: Image Reconstruction Metrics (PSNR and SSIM)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels for RGB in [0, 1].

    Args:
        pred: (B, 3, H, W) predicted image tensor in [0, 1].
        target: (B, 3, H, W) target image tensor in [0, 1].

    Returns:
        Per-image PSNR tensor of shape (B,).
    """
    mse = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=(-3, -2, -1))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def _gaussian_window(win: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    """Generate 2D Gaussian kernel window for SSIM."""
    coords = torch.arange(win, device=device, dtype=torch.float32) - win // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum()).to(dtype=dtype)
    return g[:, None] * g[None, :]


def ssim(pred: torch.Tensor, target: torch.Tensor, win: int = 11) -> torch.Tensor:
    """Compute Structural Similarity Index Measure (SSIM) over RGB channels.

    Args:
        pred: (B, 3, H, W) predicted image tensor in [0, 1].
        target: (B, 3, H, W) target image tensor in [0, 1].
        win: Gaussian window size (default 11).

    Returns:
        Per-image SSIM tensor of shape (B,).
    """
    p = pred.float()
    t = target.float()

    c1 = 0.01**2
    c2 = 0.03**2
    w = _gaussian_window(win, device=p.device, dtype=p.dtype)[None, None]
    w = w.repeat(3, 1, 1, 1)
    pad = win // 2

    mu_p = F.conv2d(F.pad(p, (pad,) * 4, mode="reflect"), w, groups=3)
    mu_t = F.conv2d(F.pad(t, (pad,) * 4, mode="reflect"), w, groups=3)

    sp = F.conv2d(F.pad(p * p, (pad,) * 4, mode="reflect"), w, groups=3) - mu_p**2
    st = F.conv2d(F.pad(t * t, (pad,) * 4, mode="reflect"), w, groups=3) - mu_t**2
    spt = F.conv2d(F.pad(p * t, (pad,) * 4, mode="reflect"), w, groups=3) - mu_p * mu_t

    score = ((2 * mu_p * mu_t + c1) * (2 * spt + c2)) / (
        (mu_p**2 + mu_t**2 + c1) * (sp + st + c2)
    )
    return score.mean(dim=(-3, -2, -1))
