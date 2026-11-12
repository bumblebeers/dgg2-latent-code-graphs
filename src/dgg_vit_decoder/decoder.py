#!/usr/bin/env python3
"""DGG ViT Decoder: 34.15M ViT+Conv Image Decoder Architecture.

Maps continuous latent visual tokens h_v in R^{B x 256 x 1152} back to
768x768x3 RGB images using a hybrid Vision Transformer token mixer and
5-stage PixelShuffle convolutional upsampler.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 1152
CANVAS_GRID = 16


class PreLNBlock(nn.Module):
    """Pre-LayerNorm Transformer Block with zero-initialized output projections."""

    def __init__(self, d: int, heads: int, mlp: int, zero_init_out: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp)
        self.fc2 = nn.Linear(mlp, d)
        self.heads = heads
        self.d = d

        if zero_init_out:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.ln1(x)
        q, k, v = (
            self.qkv(h)
            .reshape(b, n, 3, self.heads, self.d // self.heads)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, self.d))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TokenMixer(nn.Module):
    """ViT token mixer: (B, 256, 1152) -> feature map (B, d, 16, 16)."""

    def __init__(
        self,
        d_in: int = EMBED_DIM,
        d: int = 768,
        depth: int = 4,
        heads: int = 12,
        mlp: int = 3072,
        grid: int = CANVAS_GRID,
        input_ln: bool = True,
        zero_init_out: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d = d
        self.grid = grid
        self.input_ln = nn.LayerNorm(d_in) if input_ln else nn.Identity()
        self.proj = nn.Conv2d(d_in, d, kernel_size=1)
        self.pos = nn.Parameter(torch.zeros(1, d, grid, grid))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [PreLNBlock(d, heads, mlp, zero_init_out) for _ in range(depth)]
        )

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        b = h_v.shape[0]
        x = self.input_ln(h_v.float()).reshape(b, self.grid, self.grid, self.d_in)
        x = x.permute(0, 3, 1, 2)  # (B, d_in, 16, 16)
        x = self.proj(x) + self.pos  # (B, d, 16, 16)
        t = x.flatten(2).transpose(1, 2)  # (B, 256, d)
        for blk in self.blocks:
            t = blk(t)
        return t.transpose(1, 2).reshape(b, self.d, self.grid, self.grid)


class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> 3x3 conv -> GroupNorm -> SiLU -> 3x3 conv + skip."""

    def __init__(self, c: int, groups: int = 32):
        super().__init__()
        g = math.gcd(groups, c)
        self.net = nn.Sequential(
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.GroupNorm(g, c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UpsampleTo(nn.Module):
    """Conv -> PixelShuffle: (B, c_in, H, W) -> (B, c_out, H*scale, W*scale)."""

    def __init__(self, c_in: int, c_out: int, scale: int, kernel: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(
            c_in, c_out * scale * scale, kernel, padding=kernel // 2, bias=False
        )
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ps(self.conv(x))


class ConvDecoder(nn.Module):
    """Convolutional decoder upsampling (B, c_in, 16, 16) -> (B, 3, 768, 768).

    Stages:
        16x16 -> 48x48 (x3 pixel shuffle, c=256)
        48x48 -> 96x96 (x2 pixel shuffle, c=128)
        96x96 -> 192x192 (x2 pixel shuffle, c=64)
        192x192 -> 384x384 (x2 pixel shuffle, c=32)
        384x384 -> 768x768 (x2 pixel shuffle, c=16)
        Final 3x3 conv to RGB (zero init -> sigmoid(0) = 0.5 mid-grey).
    """

    def __init__(self, c_in: int = 768):
        super().__init__()
        self.pre = nn.Conv2d(c_in, 512, 1)
        self.up1 = UpsampleTo(512, 256, scale=3, kernel=1)  # 48^2
        self.res1 = ResBlock(256)
        self.up2 = UpsampleTo(256, 128, scale=2)  # 96^2
        self.res2 = ResBlock(128)
        self.up3 = UpsampleTo(128, 64, scale=2)  # 192^2
        self.res3 = ResBlock(64)
        self.up4 = UpsampleTo(64, 32, scale=2)  # 384^2
        self.res4 = ResBlock(32)
        self.up5 = UpsampleTo(32, 16, scale=2)  # 768^2
        self.res5 = ResBlock(16)
        self.out = nn.Conv2d(16, 3, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.res4(self.up4(x))
        x = self.res5(self.up5(x))
        return torch.sigmoid(self.out(x))


class ImageDecoder(nn.Module):
    """34.15M parameter hybrid ViT+Conv ImageDecoder mapping h_v -> 768x768 RGB."""

    def __init__(
        self,
        in_dim: int = EMBED_DIM,
        d: int = 768,
        depth: int = 4,
        heads: int = 12,
        mlp: int = 3072,
        input_ln: bool = True,
        zero_init_out: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.mix = TokenMixer(
            d_in=in_dim,
            d=d,
            depth=depth,
            heads=heads,
            mlp=mlp,
            input_ln=input_ln,
            zero_init_out=zero_init_out,
        )
        self.dec = ConvDecoder(d)

    def forward(self, h_v: torch.Tensor) -> torch.Tensor:
        """Forward pass mapping (B, 256, 1152) -> (B, 3, 768, 768) in [0, 1]."""
        return self.dec(self.mix(h_v))

    def num_params(self) -> int:
        """Return total parameter count (~34.15M)."""
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, os.PathLike],
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        strict: bool = True,
    ) -> "ImageDecoder":
        """Instantiate and load weights from a saved checkpoint with strict=True by default."""
        model = cls()
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            if "model" in state:
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        clean_state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(clean_state, strict=strict)
        if device is not None:
            model.to(device=device, dtype=dtype)
        model.eval()
        return model

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> "ImageDecoder":
        """Instantiate and load weights from a saved checkpoint."""
        return cls.from_checkpoint(
            path=checkpoint_path,
            device=device if device is not None else "cpu",
            dtype=dtype,
            strict=True,
        )


def pred_to_grids(
    pred_rgb: torch.Tensor,
    palette_rgb: Union[np.ndarray, torch.Tensor],
    grid_dim: int = 16,
) -> np.ndarray:
    """Map decoded continuous (B, 3, 768, 768) RGB to discrete (B, grid_dim, grid_dim) palette grid.

    Averages RGB values across each cell (48x48 px for 16x16 grid) and assigns
    nearest palette class by Euclidean distance.
    """
    pred = pred_rgb.detach().float().clamp(0.0, 1.0)
    b, c, h, w = pred.shape
    cell_h = h // grid_dim
    cell_w = w // grid_dim

    # (B, grid_dim, cell_h, grid_dim, cell_w, 3)
    p = (
        pred.permute(0, 2, 3, 1)
        .reshape(b, grid_dim, cell_h, grid_dim, cell_w, c)
        .mean(dim=(2, 4))
    )
    p_np = (p * 255.0).cpu().numpy()

    if isinstance(palette_rgb, torch.Tensor):
        pal_np = palette_rgb.cpu().numpy().astype(np.float32)
    else:
        pal_np = np.asarray(palette_rgb, dtype=np.float32)

    # Euclidean distance to palette classes: (B, grid_dim, grid_dim, n_classes)
    diff = p_np[:, :, :, None, :] - pal_np[None, None, None, :, :]
    dist_sq = (diff**2).sum(axis=-1)
    return dist_sq.argmin(axis=-1).astype(np.uint8)
