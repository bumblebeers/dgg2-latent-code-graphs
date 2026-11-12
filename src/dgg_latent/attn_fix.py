# copied from arc-agi-3@5396d0d:scripts/dg_attn_fix.py
#!/usr/bin/env python3
"""Workaround for transformers 5.19.0 DiffusionGemma multimodal-encoder mask bugs.

WHY THIS FILE EXISTS (2026-10-22)
---------------------------------
Two upstream bugs make every direct `model(...)` forward with images in this project
encode image tokens through *causal* attention instead of DG's intended bidirectional
image-block mask, contrary to the model contract in DG-STOCK-REFERENCE.md §3.6.

Bug 1 -- the mask is built and thrown away (transformers 5.19.0, also 5.20.2 and main;
        present since the first DG release v5.11.0):
    transformers/models/diffusion_gemma/modeling_diffusion_gemma.py, encoder forward:
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            self.create_masks_for_generate(...)      # <-- return value never assigned
        outputs = self.language_model(attention_mask=causal_mask_mapping, ...)
    With a 2-D attention_mask the language model therefore builds plain causal masks.
    Stock `model.generate()` is unaffected only because it precomputes the dict mask
    (generation_diffusion_gemma._prepare_encoder_inputs) and passes it in.

Bug 2 -- DG's own mask helper uses stale semantics: its `create_masks_for_generate`
        applies bidirectional image blocks to *both* sliding and full-attention layers
        and does not clip them to the sliding window. The Gemma team verified the
        intended Gemma 3/4 behaviour (transformers PR #46850): full attention stays
        causal; sliding attention is (causal OR image-blockwise) AND sliding_window.
        DiffusionGemma is initialized from Gemma 4 and inherits its multimodal path,
        so Gemma-4 semantics are the best-supported interpretation; DG's modular file
        even has a TODO to consolidate its helper into the Gemma 4 class. The variant
        switch below exists so the A/B can decide empirically.

WHAT THIS MODULE DOES
---------------------
- builds the encoder mask dict from *padded* input_ids (image positions are derived as
  `input_ids == config.image_token_id`; verified identical to the processor's
  mm_token_type_ids), so no collator padding change is needed;
- patches `DiffusionGemmaEncoderModel.forward` so any direct forward carrying images
  receives that dict mask (the upstream one-line fix, without editing site-packages);
- patches `DiffusionGemmaEncoderModel.create_masks_for_generate` so `model.generate()`
  uses the *same* mask variant (otherwise train/eval would still disagree);
- is a no-op for text-only forwards (no pixel_values / mm_encoder_outputs).

USAGE
-----
Importing this module installs the fix automatically (default variant "gemma4"); that is
why every DG-on-image script in scripts/ has a one-line `import dg_attn_fix` after its
imports. The install is idempotent, process-global, and can be switched for A/B via
`set_mask_variant()` or the environment:

    DG_ATTN_FIX_AUTO=0        do not auto-install on import (for tests that need the
                              pre-fix behaviour; dg_mask_check/dg_mask_ab set this)
    DG_ATTN_FIX_VARIANT=...   "gemma4" (default), "dg_generate" (A/B), "causal" (A/B)
    DG_ATTN_FIX_VERBOSE=0     silence the one-line install message (stderr)

Explicit usage when you load a model before doing anything else:

    from dg_attn_fix import install_dg_attn_fix
    install_dg_attn_fix(model)                      # variant="gemma4" (default)
    install_dg_attn_fix(model, variant="dg_generate")  # A/B: HF generate semantics
    install_dg_attn_fix(model, variant="causal")       # A/B: pre-fix behaviour

This is a version-pinned shim (transformers==5.19.0). When upstream fixes the encoder
forward and DG's mask helper, delete the imports/install calls; otherwise nothing else
in the project needs to change. `python scripts/dg_mask_check.py` is the offline unit
check; `python scripts/dg_mask_ab.py` is the real-model A/B (run on a GPU).
"""
from __future__ import annotations

import functools
import os
import sys

import torch

VARIANTS = ("gemma4", "dg_generate", "causal")
_DEFAULT_VARIANT = "gemma4"

# module state ---------------------------------------------------------------
_MODEL_VARIANT = None       # None = not installed
_PATCHED = False
_ORIG_FORWARD = None
_ORIG_CREATE_MASKS = None


# --------------------------------------------------------------------------- masks
def _text_config(config):
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _plain_causal_masks(config, inputs_embeds, attention_mask, past_key_values, position_ids):
    """Stock text-only masks: causal full + causal sliding-window (no image blocks)."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    kwargs = dict(config=_text_config(config), inputs_embeds=inputs_embeds,
                  attention_mask=attention_mask, past_key_values=past_key_values,
                  position_ids=position_ids)
    return {"full_attention": create_causal_mask(**kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**kwargs)}


def _dg_generate_masks(config, inputs_embeds, attention_mask, past_key_values,
                       position_ids, mm_token_type_ids):
    """The mask HF's stock `generate` would build for this prompt (pre-#46850 semantics)."""
    if _ORIG_CREATE_MASKS is None:
        from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
            DiffusionGemmaEncoderModel)
        return DiffusionGemmaEncoderModel.create_masks_for_generate(
            config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            past_key_values=past_key_values, position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids)
    return _ORIG_CREATE_MASKS(config=config, inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask, past_key_values=past_key_values,
                              position_ids=position_ids, mm_token_type_ids=mm_token_type_ids)


def _gemma4_masks(config, inputs_embeds, attention_mask, past_key_values,
                  position_ids, mm_token_type_ids):
    """Gemma-4 verified semantics (PR #46850): full causal; sliding (causal OR block) AND window."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        get_block_sequence_ids_for_mask)
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_vision_model

    if mm_token_type_ids is None:
        return _dg_generate_masks(config, inputs_embeds, attention_mask, past_key_values,
                                  position_ids, None)
    want = inputs_embeds.shape[1]
    got = mm_token_type_ids.shape[-1]
    if got != want:
        raise ValueError(
            f"mm_token_type_ids length {got} != encoder length {want}: the mask would be "
            "misaligned. Pass mm ids padded in lock-step with input_ids (or use the "
            "model-based helper, which derives them from padded input_ids).")
    block_ids = get_block_sequence_ids_for_mask(mm_token_type_ids,
                                                device=inputs_embeds.device)
    return create_masks_for_vision_model(_text_config(config), inputs_embeds,
                                         attention_mask, past_key_values, position_ids,
                                         block_ids)


def build_masks(config, inputs_embeds, attention_mask=None, past_key_values=None,
                position_ids=None, mm_token_type_ids=None, variant=None):
    """Pure mask builder. `inputs_embeds` may be metadata-only: (B, L, 0)."""
    variant = variant or _MODEL_VARIANT or os.environ.get("DG_ATTN_FIX_VARIANT",
                                                           _DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(f"unknown mask variant {variant!r}; choose from {VARIANTS}")
    if variant == "causal":
        return _plain_causal_masks(config, inputs_embeds, attention_mask,
                                   past_key_values, position_ids)
    if variant == "dg_generate":
        return _dg_generate_masks(config, inputs_embeds, attention_mask, past_key_values,
                                  position_ids, mm_token_type_ids)
    if mm_token_type_ids is None or int(mm_token_type_ids.sum()) == 0:
        # text-only prompt: stock causal masks
        return _plain_causal_masks(config, inputs_embeds, attention_mask,
                                   past_key_values, position_ids)
    return _gemma4_masks(config, inputs_embeds, attention_mask, past_key_values,
                         position_ids, mm_token_type_ids)


def build_encoder_mask(config, input_ids, attention_mask, position_ids=None,
                       past_key_values=None, variant=None, dtype=None, device=None):
    """Build the encoder mask dict for a direct forward with a padded prompt.

    Image positions are derived from `input_ids == config.image_token_id`, which equals
    the processor's mm_token_type_ids for image tokens and is already aligned with any
    left padding applied to input_ids.
    """
    input_ids = torch.as_tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, L = input_ids.shape
    device = device if device is not None else input_ids.device
    dtype = dtype if dtype is not None else torch.bfloat16
    dummy = torch.empty((B, L, 0), dtype=dtype, device=device)
    if position_ids is None:
        past_seen = (past_key_values.get_seq_length()
                     if past_key_values is not None else 0)
        position_ids = (torch.arange(L, device=device) + past_seen).unsqueeze(0)
    mm = (input_ids.to(device) == int(config.image_token_id)).long()
    return build_masks(config, dummy, attention_mask, past_key_values, position_ids,
                       mm, variant=variant)


# --------------------------------------------------------------------------- forward patch
def _masks_for_encoder_forward(enc, input_ids, attention_mask, position_ids,
                               past_key_values, mm_token_type_ids, inputs_embeds):
    config = enc.config
    device = (input_ids.device if input_ids is not None
              else next(enc.parameters()).device)
    dtype = enc.get_input_embeddings().weight.dtype
    if input_ids is not None:
        B, L = input_ids.shape
        mm = (input_ids == int(config.image_token_id)).long()
    elif inputs_embeds is not None and mm_token_type_ids is not None:
        B, L = inputs_embeds.shape[0], inputs_embeds.shape[1]
        if mm_token_type_ids.shape[-1] != L:
            raise ValueError("mm_token_type_ids length mismatch with inputs_embeds")
        mm = mm_token_type_ids
    else:  # cannot tell where the image blocks are -> leave the caller's mask untouched
        return attention_mask
    dummy = torch.empty((B, L, 0), dtype=dtype, device=device)
    if position_ids is None:
        past_seen = (past_key_values.get_seq_length()
                     if past_key_values is not None else 0)
        position_ids = (torch.arange(L, device=device) + past_seen).unsqueeze(0)
    return build_masks(config, dummy, attention_mask, past_key_values, position_ids, mm)


def _make_forward(orig_forward):
    @functools.wraps(orig_forward)
    def forward(self, input_ids=None, pixel_values=None, attention_mask=None,
                position_ids=None, past_key_values=None, mm_token_type_ids=None,
                inputs_embeds=None, image_position_ids=None, mm_encoder_outputs=None,
                **kwargs):
        has_images = pixel_values is not None or mm_encoder_outputs is not None
        if (has_images and not isinstance(attention_mask, dict)
                and (_MODEL_VARIANT or "gemma4") != "causal"):
            attention_mask = _masks_for_encoder_forward(
                self, input_ids, attention_mask, position_ids, past_key_values,
                mm_token_type_ids, inputs_embeds)
        return orig_forward(
            self, input_ids=input_ids, pixel_values=pixel_values,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, mm_token_type_ids=mm_token_type_ids,
            inputs_embeds=inputs_embeds, image_position_ids=image_position_ids,
            mm_encoder_outputs=mm_encoder_outputs, **kwargs)
    return forward


def _make_create_masks(orig_create):
    def create_masks_for_generate(config, inputs_embeds, attention_mask,
                                  past_key_values, position_ids,
                                  mm_token_type_ids=None, **kwargs):
        variant = _MODEL_VARIANT or "gemma4"
        if variant == "causal":
            return _plain_causal_masks(config, inputs_embeds, attention_mask,
                                       past_key_values, position_ids)
        if variant == "dg_generate":
            return orig_create(config=config, inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               past_key_values=past_key_values,
                               position_ids=position_ids,
                               mm_token_type_ids=mm_token_type_ids)
        # gemma4: only text-only prompts fall back to the stock builder
        if mm_token_type_ids is None or int(mm_token_type_ids.sum()) == 0:
            return orig_create(config=config, inputs_embeds=inputs_embeds,
                               attention_mask=attention_mask,
                               past_key_values=past_key_values,
                               position_ids=position_ids,
                               mm_token_type_ids=mm_token_type_ids)
        return _gemma4_masks(config, inputs_embeds, attention_mask, past_key_values,
                             position_ids, mm_token_type_ids)
    return create_masks_for_generate


# --------------------------------------------------------------------------- install
def install_dg_attn_fix(model=None, variant=None, verbose=True):
    """Patch DG's encoder so direct forwards (and generate) use a vision-aware mask.

    Call once per process after the model is loaded, BEFORE install_reinjection()
    (the injection wrapper captures the encoder forward, so the mask fix must be in
    place first). Safe to call repeatedly; changing `variant` switches the A/B mode.
    `variant` defaults to $DG_ATTN_FIX_VARIANT or "gemma4".
    """
    global _PATCHED, _ORIG_FORWARD, _ORIG_CREATE_MASKS, _MODEL_VARIANT
    variant = variant or os.environ.get("DG_ATTN_FIX_VARIANT", _DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(f"unknown mask variant {variant!r}; choose from {VARIANTS}")
    _MODEL_VARIANT = variant
    if _PATCHED:
        if verbose:
            print(f"[dg_attn_fix] variant -> {variant}", flush=True)
        return model

    import transformers
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderModel)

    _ORIG_FORWARD = DiffusionGemmaEncoderModel.forward
    _ORIG_CREATE_MASKS = DiffusionGemmaEncoderModel.create_masks_for_generate
    DiffusionGemmaEncoderModel.forward = _make_forward(_ORIG_FORWARD)
    DiffusionGemmaEncoderModel.create_masks_for_generate = staticmethod(
        _make_create_masks(_ORIG_CREATE_MASKS))
    _PATCHED = True
    if verbose:
        print(f"[dg_attn_fix] installed (transformers {transformers.__version__}, "
              f"variant={variant}); upstream bugs: DG encoder drops the built mask for 2-D "
              f"attention_mask, and its mask helper predates Gemma-4 sliding-only semantics.",
              flush=True)
    if transformers.__version__ != "5.19.0" and verbose:
        print("[dg_attn_fix] WARNING: shim is written against transformers 5.19.0; "
              "re-verify against upstream before trusting it.", flush=True)
    return model


def set_mask_variant(variant):
    """Change the mask variant after install (A/B testing)."""
    global _MODEL_VARIANT
    if variant not in VARIANTS:
        raise ValueError(f"unknown mask variant {variant!r}; choose from {VARIANTS}")
    if not _PATCHED:
        raise RuntimeError("call install_dg_attn_fix(model) first")
    _MODEL_VARIANT = variant


def is_installed():
    return _PATCHED


def _auto_install():
    """Importing this module installs the fix, so no script can silently miss it."""
    if os.environ.get("DG_ATTN_FIX_AUTO", "1").lower() in ("0", "false", "no", "off"):
        return
    verbose = os.environ.get("DG_ATTN_FIX_VERBOSE", "1").lower() not in ("0", "false", "no")
    try:
        install_dg_attn_fix(None, verbose=verbose)
    except Exception as exc:  # pragma: no cover - depends on the installed transformers
        print(f"[dg_attn_fix] WARNING: auto-install failed ({type(exc).__name__}: {exc}); "
              "DG image forwards are running with the upstream causal-mask bug.",
              file=sys.stderr)


_auto_install()
