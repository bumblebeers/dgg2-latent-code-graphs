#!/usr/bin/env python3
"""cg_nondg_model.py -- Genuine PyTorch Non-DG Spatial Transformer Baseline.

Faithful implementation of NonDGTransformBaseline (~49.9M-53.6M parameters at d_model=768)
ported from parent arc-agi-3 (scripts/dgg2/colab_runner_phase2_transforms.py and
colab_runner_sec27_part0_1_2.py).

Features:
- 2D spatial self-attention with AdaLN modulation from instruction conditioning
- Unpooled cross-attention from 256 image patches to instruction tokens
- Pre-LN FFN with AdaLN modulation
- DualReadout-compatible linear head (head_primary)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NonDGTransformBlock(nn.Module):
    """Spatial Transformer block with AdaLN modulation and cross-attention."""

    def __init__(self, d_model: int = 768, n_heads: int = 12, d_ff: int = 3072):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ca_q = nn.LayerNorm(d_model)
        self.norm_ca_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ada_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 4 * d_model),
        )
        # Initialize AdaLN modulation to zero (identity initialization)
        nn.init.zeros_(self.ada_mod[-1].weight)
        nn.init.zeros_(self.ada_mod[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        txt_tokens: torch.Tensor,
        txt_key_padding_mask: torch.Tensor,
        txt_cond: torch.Tensor,
    ) -> torch.Tensor:
        # AdaLN modulation chunked into (shift_sa, scale_sa, shift_ff, scale_ff)
        shift_sa, scale_sa, shift_ff, scale_ff = self.ada_mod(txt_cond).chunk(4, dim=-1)

        # 1. Self-Attention with AdaLN
        q_sa = self.norm_sa(x) * (1.0 + scale_sa.unsqueeze(1)) + shift_sa.unsqueeze(1)
        sa_out, _ = self.self_attn(q_sa, q_sa, q_sa, need_weights=False)
        x = x + sa_out

        # 2. Cross-Attention from patches to instruction tokens
        q_ca = self.norm_ca_q(x)
        kv_ca = self.norm_ca_kv(txt_tokens)
        ca_out, _ = self.cross_attn(
            q_ca, kv_ca, kv_ca, key_padding_mask=txt_key_padding_mask, need_weights=False
        )
        x = x + ca_out

        # 3. Feed-Forward Network with AdaLN
        h_ff = self.norm_ff(x) * (1.0 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
        x = x + self.ffn(h_ff)
        return x


class NonDGTransformBaseline(nn.Module):
    """Full 4-layer Spatial Transformer Baseline (~49.9M-53.6M params at d_model=768)."""

    def __init__(
        self,
        hv_dim: int = 1152,
        txt_dim: int = 2816,
        d_model: int = 768,
        n_heads: int = 12,
        d_ff: int = 3072,
        n_layers: int = 4,
        max_txt_len: int = 160,
    ):
        super().__init__()
        self.hv_dim = hv_dim
        self.txt_dim = txt_dim
        self.d_model = d_model

        # Patch feature projection
        self.in_proj = nn.Sequential(
            nn.LayerNorm(hv_dim),
            nn.Linear(hv_dim, d_model),
        )
        self.pos_2d = nn.Parameter(torch.randn(1, 256, d_model) * 0.02)

        # Text conditioning projection
        self.txt_proj = nn.Sequential(
            nn.LayerNorm(txt_dim),
            nn.Linear(txt_dim, d_model),
        )
        self.txt_pos = nn.Parameter(torch.randn(1, max_txt_len, d_model) * 0.02)
        self.txt_attn_score = nn.Linear(d_model, 1)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            NonDGTransformBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

        # Primary prediction head -> predicted h_v (256, hv_dim)
        self.head_primary = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, hv_dim),
        )

    def forward(
        self,
        src_hv: torch.Tensor,
        txt_tokens_std: torch.Tensor,
        txt_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        Args:
            src_hv: (B, 256, hv_dim) continuous source latent features
            txt_tokens_std: (B, L_t, txt_dim) Z-standardized instruction tokens
            txt_valid_mask: (B, L_t) boolean mask, True for valid tokens
        Returns:
            pred_hv: (B, 256, hv_dim) predicted target latent features
        """
        b, l_t, _ = txt_tokens_std.shape
        x = self.in_proj(src_hv.float()) + self.pos_2d

        # Project instruction tokens
        if l_t > self.txt_pos.shape[1]:
            pos = F.interpolate(self.txt_pos.transpose(1, 2), size=l_t, mode="linear", align_corners=False).transpose(1, 2)
        else:
            pos = self.txt_pos[:, :l_t, :]
        t_h = self.txt_proj(txt_tokens_std.float()) + pos

        # Attention-pooled global instruction conditioning for AdaLN
        scores = self.txt_attn_score(t_h).squeeze(-1)
        scores = scores.masked_fill(~txt_valid_mask, -1e9)
        attn_w = torch.softmax(scores, dim=-1).unsqueeze(-1)
        txt_cond = (t_h * attn_w).sum(dim=1)

        key_pad_mask = ~txt_valid_mask
        for blk in self.blocks:
            x = blk(x, t_h, key_pad_mask, txt_cond)

        h = self.out_norm(x)
        return self.head_primary(h)
