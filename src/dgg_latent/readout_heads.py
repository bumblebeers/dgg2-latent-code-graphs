#!/usr/bin/env python3
"""DGG Latent: Single-Canvas, Dual-Gated, and Cross-Canvas Readout Heads.

Provides pointwise MLP, gated vocabulary chord residual, and multi-canvas cross-attention
readout architectures mapping decoder hidden representations to latent code spaces.
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

CANVAS_LEN = 256
DEFAULT_SOFT_DIM = 2816
DEFAULT_HV_DIM = 1152
DEFAULT_BOT_DIM = 768


class SoftCanvasHead(nn.Module):
    """Pointwise MLP readout head for single-canvas representations.

    Maps decoder hidden states H in R^{B x 256 x d_in} to continuous latent
    vectors in R^{B x 256 x d_out} via a 2-hidden-layer bottleneck MLP.

    Architecture:
        x in R^{B x 256 x d_in}
        -> LayerNorm(d_in)
        -> Linear(d_in, d_bot) + GELU
        -> Linear(d_bot, d_bot) + GELU
        -> Linear(d_bot, d_out)
    """

    arch = "single_canvas_mlp"

    def __init__(
        self,
        d_in: int = DEFAULT_SOFT_DIM,
        d_bot: int = DEFAULT_BOT_DIM,
        d_out: int = DEFAULT_HV_DIM,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_bot = d_bot
        self.d_out = d_out

        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, d_bot)
        self.fc2 = nn.Linear(d_bot, d_bot)
        self.fc3 = nn.Linear(d_bot, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Tensor of shape (B, L, d_in) or (B, d_in).

        Returns:
            Projected latent tensor of shape (B, L, d_out) or (B, d_out).
        """
        x = self.norm(h.float())
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        return self.fc3(x)

    def init_cold_start(self) -> None:
        """Cold-start Xavier/Kaiming initialization routine."""
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class DualReadoutSoftCanvasHead(nn.Module):
    """Dual-norm gated residual readout head.

    Fuses unconstrained decoder hidden state h_dec in R^{B x L x d_in} with
    expected vocabulary embedding chord e_soft in R^{B x L x d_in} and detached
    logit confidence statistics c_z in R^{B x L x d_conf}:

        u_h = GELU(W_h * LayerNorm_h(h_dec) + b_h)
        u_z = GELU(W_z * LayerNorm_z(e_soft) + b_z)
        g   = sigmoid(W_g * [u_h ; u_z ; c_z] + b_g)
        u_fused = m_drop * u_h + g * u_z
        pred_latent = W_3 * GELU(W_2 * LayerNorm_fuse(u_fused) + b_2) + b_3
    """

    arch = "dual_gated_canvas_mlp"

    def __init__(
        self,
        d_in: int = DEFAULT_SOFT_DIM,
        d_bot: int = DEFAULT_BOT_DIM,
        d_out: int = DEFAULT_HV_DIM,
        d_conf: int = 2,
        p_drop_h: float = 0.15,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_bot = d_bot
        self.d_out = d_out
        self.d_conf = d_conf
        self.p_drop_h = float(p_drop_h)

        self.norm_h = nn.LayerNorm(d_in)
        self.fc_h = nn.Linear(d_in, d_bot)
        self.norm_z = nn.LayerNorm(d_in)
        self.fc_z = nn.Linear(d_in, d_bot)

        self.gate_proj = nn.Linear(2 * d_bot + d_conf, d_bot)

        self.norm_fuse = nn.LayerNorm(d_bot)
        self.fc2 = nn.Linear(d_bot, d_bot)
        self.fc3 = nn.Linear(d_bot, d_out)
        self.last_gate_mean: float = 0.5

    def init_cold_start(self) -> None:
        """Cold-start initialization with zero gate bias so initial g ≈ 0.5."""
        for norm in (self.norm_h, self.norm_z, self.norm_fuse):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for layer in (self.fc_h, self.fc_z, self.fc2, self.fc3):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.xavier_normal_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        h_or_pack: Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[torch.Tensor, ...], list],
        e_soft: Optional[torch.Tensor] = None,
        c_z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass supporting dict pack, tuple, or explicit tensors."""
        if isinstance(h_or_pack, dict):
            h_dec = h_or_pack["h_dec"]
            e_soft = h_or_pack["e_soft"]
            c_z = h_or_pack.get("c_z")
        elif isinstance(h_or_pack, (tuple, list)):
            h_dec = h_or_pack[0]
            e_soft = h_or_pack[1]
            c_z = h_or_pack[2] if len(h_or_pack) > 2 else None
        else:
            h_dec = h_or_pack
            if e_soft is None:
                raise ValueError("DualReadoutSoftCanvasHead requires both h_dec and e_soft")

        h_f = h_dec.float()
        z_f = e_soft.float()
        b, l, _ = h_f.shape

        if c_z is None:
            c_f = h_f.new_zeros((b, l, self.d_conf))
        else:
            c_f = c_z.float().to(h_f.device)

        u_h = F.gelu(self.fc_h(self.norm_h(h_f)))
        u_z = F.gelu(self.fc_z(self.norm_z(z_f)))

        gate_in = torch.cat([u_h, u_z, c_f], dim=-1)
        g = torch.sigmoid(self.gate_proj(gate_in))
        self.last_gate_mean = float(g.detach().mean().item())

        if self.training and self.p_drop_h > 0.0:
            keep_mask = (torch.rand((b, l, 1), device=u_h.device, dtype=u_h.dtype) >= self.p_drop_h).to(u_h.dtype)
            u_fused = keep_mask * u_h + g * u_z
        else:
            u_fused = u_h + g * u_z

        x = F.gelu(self.fc2(self.norm_fuse(u_fused)))
        return self.fc3(x)

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


class CrossSoftCanvasHead(nn.Module):
    """Two-canvas composite readout head with cross-attention.

    Attends across Canvas 1 (macro-structure) and Canvas 2 (detail) hidden representations,
    fusing them into target continuous latent features.
    """

    arch = "cross_canvas_attn"

    def __init__(
        self,
        d_in: int = DEFAULT_SOFT_DIM,
        d_hidden: int = DEFAULT_BOT_DIM,
        d_out: int = DEFAULT_HV_DIM,
        heads: int = 8,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self.heads = heads

        self.norm1 = nn.LayerNorm(d_in)
        self.norm2 = nn.LayerNorm(d_in)
        self.c1_proj = nn.Linear(d_in, d_hidden)
        self.c2_proj = nn.Linear(d_in, d_hidden)
        self.q_proj = nn.Linear(2 * d_hidden, d_hidden)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_hidden, num_heads=heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def init_cold_start(self) -> None:
        """Cold-start Xavier/Kaiming initialization routine."""
        for norm in (self.norm1, self.norm2):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        for proj in (self.c1_proj, self.c2_proj, self.q_proj):
            nn.init.kaiming_normal_(proj.weight, nonlinearity="relu")
            nn.init.zeros_(proj.bias)

    def forward(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Cross-attention forward across canvas 1 and canvas 2 hidden states."""
        c1_f = self.norm1(c1.float())
        c2_f = self.norm2(c2.float())

        h1 = F.gelu(self.c1_proj(c1_f))
        h2 = F.gelu(self.c2_proj(c2_f))

        q_in = torch.cat([h1, h2], dim=-1)
        q = self.q_proj(q_in)
        kv = torch.cat([h1, h2], dim=1)

        attended, _ = self.cross_attn(q, kv, kv)
        return self.mlp(attended)

    def num_params(self) -> int:
        """Return total parameter count."""
        return sum(p.numel() for p in self.parameters())


def compute_readout_input(
    h_dec: Optional[torch.Tensor],
    logits: Optional[torch.Tensor],
    readout_mode: str = "hidden",
    lm_weight: Optional[torch.Tensor] = None,
    tau: float = 1.0,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the input structure for SoftCanvasHead or DualReadoutSoftCanvasHead.

    Args:
        h_dec: Decoder hidden states of shape (B, L, D).
        logits: Decoder vocabulary logits of shape (B, L, V).
        readout_mode: 'hidden', 'logits', or 'dual_gated'.
        lm_weight: Vocabulary embedding matrix (V, D) for computing continuous chords.
        tau: Softmax temperature scaling.

    Returns:
        Tensor or dict suitable for passing to the corresponding readout head.
    """
    if readout_mode == "hidden":
        if h_dec is None:
            raise ValueError("h_dec is required when readout_mode == 'hidden'")
        return h_dec
    elif readout_mode == "logits":
        if logits is None or lm_weight is None:
            raise ValueError("logits and lm_weight are required when readout_mode == 'logits'")
        probs = F.softmax(logits.float() / float(tau), dim=-1)
        e_soft = torch.matmul(probs.to(lm_weight.dtype), lm_weight)
        return e_soft.float()
    elif readout_mode == "dual_gated":
        if h_dec is None or logits is None or lm_weight is None:
            raise ValueError("h_dec, logits, and lm_weight are required when readout_mode == 'dual_gated'")
        probs = F.softmax(logits.float() / float(tau), dim=-1)
        e_soft = torch.matmul(probs.to(lm_weight.dtype), lm_weight).float()
        with torch.no_grad():
            p_max = probs.max(dim=-1, keepdim=True).values
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1, keepdim=True)
            log_v = float(math.log(max(2, probs.shape[-1])))
            conf_ent = (1.0 - ent / log_v).clamp(0.0, 1.0)
            c_z = torch.cat([p_max, conf_ent], dim=-1).float()
        return {"h_dec": h_dec, "e_soft": e_soft, "c_z": c_z}
    else:
        raise ValueError(
            f"Unknown readout_mode: {readout_mode!r}; expected 'hidden', 'logits', or 'dual_gated'"
        )


def build_single_canvas_head(readout_mode: str = "hidden", **kwargs: Any) -> nn.Module:
    """Factory returning the appropriate single-canvas readout head."""
    if readout_mode == "dual_gated":
        return DualReadoutSoftCanvasHead(**kwargs)
    return SoftCanvasHead(**kwargs)


# copied from arc-agi-3@5396d0d:scripts/dgg2/colab_runner_sec27_part0_1_2.py
class SymbolVectorInputAdapter(nn.Module):
    """Input adapter mapping standardised symbol vectors (d=384) to DiffusionGemma soft tokens (2816)
    with RMS normalization matching per HANDOFF.md §6.5."""

    arch = "symbol_vector_input_adapter"

    def __init__(self, in_dim: int = 384, out_dim: int = DEFAULT_SOFT_DIM, rms_scale: float = 1.28):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ln = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.rms_scale = float(rms_scale)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.ln(x.float()))
        rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (out / rms) * self.rms_scale

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, os.PathLike],
        device: Union[str, torch.device] = "cpu",
        strict: bool = True,
    ) -> "SymbolVectorInputAdapter":
        """Instantiate and load weights from checkpoint with strict=True."""
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict):
            if "adapter_state_dict" in ckpt:
                state_dict = ckpt["adapter_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        clean_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}
        if "ln.weight" in clean_sd and "linear.weight" in clean_sd:
            in_dim = clean_sd["ln.weight"].shape[0]
            out_dim = clean_sd["linear.weight"].shape[0]
            model = cls(in_dim=in_dim, out_dim=out_dim)
        else:
            model = cls()
        model.load_state_dict(clean_sd, strict=strict)
        model.to(device)
        model.eval()
        return model


ExpBInputAdapter = SymbolVectorInputAdapter
