#!/usr/bin/env python3
"""DGG ViT Decoder: Standalone Vision-Stack-Only Loader.

Extracts and runs the frozen 572M SigLIP vision tower and embed_vision projector
from DiffusionGemma without loading the 49 GB / 26B language model.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

N_TOKENS = 256
EMBED_DIM = 1152
SOFT_DIM = 2816


def _load_sub_state_dict(ckpt_dir: str, prefixes: Tuple[str, ...]) -> Dict[str, torch.Tensor]:
    """Extract tensors matching prefixes from safetensors or pytorch bin files."""
    out: Dict[str, torch.Tensor] = {}
    st_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if st_files:
        from safetensors import safe_open

        for fn in st_files:
            with safe_open(fn, framework="pt", device="cpu") as fh:
                for key in fh.keys():
                    for p in prefixes:
                        if key.startswith(p):
                            out[key[len(p) :]] = fh.get_tensor(key)
        return out

    bin_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.bin")))
    for fn in bin_files:
        data = torch.load(fn, map_location="cpu", weights_only=True)
        for key, tensor in data.items():
            for p in prefixes:
                if key.startswith(p):
                    out[key[len(p) :]] = tensor
    return out


class VisionStack:
    """Frozen DG image-encoding stack: Gemma4VisionModel + embed_vision projector (572M params).

    Reproduces DiffusionGemmaEncoderModel.get_image_features(pixel_values, image_position_ids)
    exactly without loading the language model.
    """

    VISION_PREFIX = "model.encoder.vision_tower."
    EMBED_PREFIX = "model.encoder.embed_vision."

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        config: Optional[object] = None,
    ):
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.device = device
        self.dtype = dtype
        self.config = config

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_dir: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "VisionStack":
        """Load vision stack directly from DiffusionGemma checkpoint directory."""
        from transformers import AutoModel, DiffusionGemmaConfig
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaMultimodalEmbedder,
        )

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        cfg = DiffusionGemmaConfig.from_pretrained(ckpt_dir)
        vision_tower = AutoModel.from_config(cfg.vision_config)
        embed_vision = DiffusionGemmaMultimodalEmbedder(cfg.vision_config, cfg.text_config)

        v_sd = _load_sub_state_dict(ckpt_dir, (cls.VISION_PREFIX,))
        e_sd = _load_sub_state_dict(ckpt_dir, (cls.EMBED_PREFIX,))

        if v_sd:
            vision_tower.load_state_dict(v_sd, strict=True, assign=True)
        if e_sd:
            embed_vision.load_state_dict(e_sd, strict=True, assign=True)

        vision_tower.to(device=device, dtype=dtype).eval()
        embed_vision.to(device=device, dtype=dtype).eval()

        for p in list(vision_tower.parameters()) + list(embed_vision.parameters()):
            p.requires_grad_(False)

        return cls(vision_tower, embed_vision, device, dtype, cfg)

    def num_params(self) -> int:
        """Return total parameter count of the vision stack (~572M)."""
        return sum(p.numel() for p in self.vision_tower.parameters()) + sum(
            p.numel() for p in self.embed_vision.parameters()
        )

    @torch.inference_mode()
    def encode(
        self,
        pixel_values: torch.Tensor,
        image_position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode pixel_values (B, 2520, 768) -> h_v (B, 256, 1152) and softs (B, 256, 2816)."""
        pv = pixel_values.to(self.device, non_blocking=True)
        kwargs = {"pixel_values": pv}
        if image_position_ids is not None:
            kwargs["pixel_position_ids"] = image_position_ids.to(self.device, non_blocking=True)

        out = self.vision_tower(**kwargs)
        h_flat = out.last_hidden_state  # (B*256, 1152)
        s_flat = self.embed_vision(inputs_embeds=h_flat)  # (B*256, 2816)
        b = pv.shape[0]

        return {
            "h_v": h_flat.reshape(b, N_TOKENS, EMBED_DIM),
            "softs": s_flat.reshape(b, N_TOKENS, SOFT_DIM),
            "h_v_flat": h_flat,
        }


def load_embed_vision(
    ckpt_dir: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Load only the frozen embed_vision projector (RMSNorm -> Linear 1152->2816)."""
    from transformers import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaMultimodalEmbedder,
    )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = DiffusionGemmaConfig.from_pretrained(ckpt_dir)
    embed_vision = DiffusionGemmaMultimodalEmbedder(cfg.vision_config, cfg.text_config)
    e_sd = _load_sub_state_dict(ckpt_dir, (VisionStack.EMBED_PREFIX,))
    if e_sd:
        embed_vision.load_state_dict(e_sd, strict=True, assign=True)

    embed_vision.to(device=device, dtype=dtype).eval()
    for p in embed_vision.parameters():
        p.requires_grad_(False)
    return embed_vision
