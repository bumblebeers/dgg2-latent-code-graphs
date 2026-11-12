#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cg_colab_runner.py -- DiffusionGemma 26B-A4B Code-Graph Runner (Colab G4/A100).

Implements genuine DiffusionGemma 26B-A4B training and evaluation on:
  - Experiment A: Call-graph 16x16 adjacency grids (48px/cell -> 768x768 RGB -> 256 soft tokens)
                  via frozen vision tower and DualReadoutSoftCanvasHead (h_v: 256 x 1152)
  - Experiment B: 256-symbol AST repository representations (frozen text encoder d=384 ->
                  trainable LayerNorm+Linear adapter -> 2816 soft tokens) and DualReadout head (d=384)

Strict DiffusionGemma Stock Contracts:
  1. 256 canvas length (16x16 grid or 256 latent symbol tokens).
  2. Text sequence length max_txt_len >= 160 (never truncate after 80-token preamble).
  3. Import and apply dg_attn_fix.py before any DG forward pass.
  4. Multinomial corruption with full-vocab CE and self-conditioning p=0.5.
  5. Readout strictly via stock_generate_with_readout at final_sc_denoising_step.
  6. Exactly 4,824 optimizer steps across 3 epochs (12,864 rows/epoch, bs=4, accum=2).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import contextlib
import gc
import gzip
import io
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Path setup
for p in [
    "/content/dgg2/bundle",
    "/content/dgg2",
    "/content/dgg2/cg",
    str(Path(__file__).resolve().parent),
    str(Path(__file__).resolve().parent.parent / "cg"),
    str(Path(__file__).resolve().parent.parent),
    str(Path.home() / "workspaces" / "arc-agi-3" / "scripts"),
    str(Path.home() / "workspaces" / "arc-agi-3" / "scripts" / "dgg2"),
]:
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)
    elif p.startswith("/content") and p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

import dg_attn_fix  # noqa: F401 -- mandatory upstream fix
import cg_sec27_spec as SP
import cg_instructions as CI
import cg_ops as O

from dg_decode_train import ImageDecoder
from dgg2_canvas_heads import (
    CANVAS_LEN_SINGLE,
    DG_VOCAB_SIZE,
    HV_DIM,
    DualReadoutSoftCanvasHead,
    build_single_canvas_head,
)
from dgg2_loss import (
    DEFAULT_COS_WEIGHT,
    DEFAULT_SMOOTH_L1_BETA,
    DEFAULT_W_THRESH,
    compute_position_ids,
    compute_readout_input,
    distribution_ce,
    loss_hv_distance,
    sample_x0_from_modes,
)
from dgg2_sample_stock import (
    _GenRecorder,
    load_stock_generation_config,
    stock_generate_with_readout,
)
from colab_runner_m2 import (
    create_cold_start_lora,
    get_dg_modules,
    supervised_forward,
    corrupt_canvas_multinomial,
)
from colab_runner_dual_scale import (
    load_ckpt_any,
    push_file_to_drive,
    save_slim_ckpt_gz,
)
from colab_runner_phase1_identity import (
    IMAGE_TOKEN_ID,
    build_image_instruction_prompt,
    make_mm_encoder_outputs,
)
from colab_runner_phase2_transforms import (
    encode_pil_images_vision_tower,
    extract_teacher_modes_from_hv_batched,
    tokenize_instruction_batch,
)


# ==============================================================================
# Helper: Drive Sync
# ==============================================================================
def try_pull_from_drive(remote_name: str, local_path: Path) -> bool:
    if local_path.exists() and local_path.stat().st_size > 100:
        return True
    drive_util = Path("/content/dgg2/drive_util.py")
    token_json = Path("/content/dgg2/token.json")
    if not (drive_util.exists() and token_json.exists()):
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    try:
        subprocess.check_call(
            [sys.executable, str(drive_util), "download", str(token_json), remote_name, str(local_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=600,
        )
        if local_path.exists() and local_path.stat().st_size > 100:
            print(f"[Drive] Pulled {remote_name} -> {local_path} ({local_path.stat().st_size / 1024**2:.1f} MB)", flush=True)
            return True
    except Exception:
        pass
    return False


# ==============================================================================
# Int8 Quantization for h_v Cache
# ==============================================================================
def quantize_hv_per_token_int8(hv_fp16: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    hv_f = hv_fp16.float()
    amax = hv_f.abs().amax(dim=-1).clamp_min(1e-6)
    scale = (amax / 127.0).to(torch.float16)
    q = torch.round(hv_f / scale.float().unsqueeze(-1)).clamp(-127.0, 127.0).to(torch.int8)
    return q.cpu(), scale.cpu()


def dequantize_hv_per_token_int8(q_i8: torch.Tensor, scale_fp16: torch.Tensor, device: torch.device) -> torch.Tensor:
    return q_i8.to(device=device, dtype=torch.float32) * scale_fp16.to(device=device, dtype=torch.float32).unsqueeze(-1)


# ==============================================================================
# In-place LoRA Weight Reset
# ==============================================================================
def reset_cg_lora_weights_inplace(model_s: nn.Module, seed: int = 42, enable_lora: bool = True) -> List[nn.Parameter]:
    torch.manual_seed(seed)
    trainable: List[nn.Parameter] = []
    for n, p in model_s.named_parameters():
        if "lora_" in n:
            p.requires_grad = enable_lora
            if "lora_B" in n:
                nn.init.zeros_(p)
            elif "lora_A" in n and enable_lora:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            if enable_lora:
                trainable.append(p)
        else:
            p.requires_grad = False
    return trainable


# ==============================================================================
# Experiment A: Cache Builders (Vision-Tower Encoding)
# ==============================================================================
def prepare_cg_test360_cache(
    pack_path: str,
    processor: Any,
    base_model: nn.Module,
    device: torch.device,
    ctx_ckpt: Any,
    cache_path: str = "/content/dgg2/cg_test360_cache_33states.pt",
) -> Dict[str, Any]:
    if cache_path == "/content/dgg2/cg_test360_cache_33states.pt" and "click" in str(pack_path):
        cache_path = "/content/dgg2/cg_test360_click_cache_33states.pt"
    c_p = Path(cache_path)
    try_pull_from_drive(c_p.name, c_p)
    if c_p.exists():
        print(f"[ExpA Cache] Loading pre-encoded Test-360 cache from {c_p}...", flush=True)
        cached_dict = torch.load(c_p, map_location="cpu", weights_only=False)
        cached_dict["pack_path"] = str(pack_path)
        return cached_dict

    pack = SP.load_pack(pack_path)
    n_sources = len(pack["src_grid"])  # 360
    n_states = len(SP.CG_ALL_STATES)  # 33

    print(f"\n[ExpA TestPrep] Encoding N={n_sources} x {n_states} states with Vision Tower...", flush=True)
    all_tgt_hv = torch.zeros((n_states, n_sources, 256, 1152), dtype=torch.float16)
    all_mode_ids = torch.zeros((n_states, n_sources, 256, 64), dtype=torch.int32)
    all_mode_weights = torch.zeros((n_states, n_sources, 256, 64), dtype=torch.float16)

    for k_idx, st_name in enumerate(SP.CG_ALL_STATES):
        st_pils: List[Image.Image] = []
        valid_indices: List[int] = []
        for i in range(n_sources):
            if SP.state_valid(pack, i, st_name):
                st_pils.append(SP.render_state_pil(pack, i, st_name))
                valid_indices.append(i)

        if st_pils:
            hv_valid = encode_pil_images_vision_tower(
                st_pils, processor, base_model, device, batch_size=36, desc=f"Test360-{k_idx:02d}-{st_name}"
            )
            for local_idx, src_idx in enumerate(valid_indices):
                all_tgt_hv[k_idx, src_idx] = hv_valid[local_idx]

            if ctx_ckpt is not None:
                ids_k, w_k = extract_teacher_modes_from_hv_batched(
                    hv_valid, ctx_ckpt, processor.tokenizer, device, batch_size=75
                )
                for local_idx, src_idx in enumerate(valid_indices):
                    all_mode_ids[k_idx, src_idx] = ids_k[local_idx]
                    all_mode_weights[k_idx, src_idx] = w_k[local_idx]
                del ids_k, w_k
            del st_pils, hv_valid

    hv_mean = np.load("/content/dgg2/data/hv/hv_mean.npy")
    cache_obj = {
        "all_tgt_hv_33x360": all_tgt_hv,
        "all_mode_ids_33x360": all_mode_ids,
        "all_mode_weights_33x360": all_mode_weights,
        "hv_mean": hv_mean,
        "pack": pack,
        "pack_path": str(pack_path),
    }
    torch.save(cache_obj, c_p)
    push_file_to_drive(c_p, c_p.name)
    return cache_obj


@torch.no_grad()
def prepare_cg_train1608_cache(
    pack_path: str,
    processor: Any,
    base_model: nn.Module,
    device: torch.device,
    ctx_ckpt: Any,
    cache_path: str = "/content/dgg2/cg_train1608_cache_23states.pt",
) -> Dict[str, Any]:
    c_p = Path(cache_path)
    try_pull_from_drive(c_p.name, c_p)
    if c_p.exists():
        print(f"[ExpA Cache] Loading pre-encoded Train-1608 cache from {c_p}...", flush=True)
        return torch.load(c_p, map_location="cpu", weights_only=False)

    pack = SP.load_pack(pack_path)
    n_sources = len(pack["src_grid"])  # 1,608
    n_states = len(SP.CG_TRAIN_STATES)  # 23

    print(f"\n[ExpA TrainPrep] Encoding N={n_sources} x {n_states} states with Vision Tower...", flush=True)
    train_hv_i8 = torch.zeros((n_states, n_sources, 256, 1152), dtype=torch.int8)
    train_hv_scale = torch.zeros((n_states, n_sources, 256), dtype=torch.float16)
    train_mode_ids = torch.zeros((n_states, n_sources, 256, 64), dtype=torch.int32)
    train_mode_weights = torch.zeros((n_states, n_sources, 256, 64), dtype=torch.float16)

    for k_idx, st_name in enumerate(SP.CG_TRAIN_STATES):
        st_pils: List[Image.Image] = []
        valid_indices: List[int] = []
        for i in range(n_sources):
            if SP.state_valid(pack, i, st_name):
                st_pils.append(SP.render_state_pil(pack, i, st_name))
                valid_indices.append(i)

        if st_pils:
            hv_valid = encode_pil_images_vision_tower(
                st_pils, processor, base_model, device, batch_size=36, desc=f"Train1608-{k_idx:02d}-{st_name}"
            )
            q_i8, s_fp16 = quantize_hv_per_token_int8(hv_valid)
            for local_idx, src_idx in enumerate(valid_indices):
                train_hv_i8[k_idx, src_idx] = q_i8[local_idx]
                train_hv_scale[k_idx, src_idx] = s_fp16[local_idx]

            ids_k, w_k = extract_teacher_modes_from_hv_batched(
                hv_valid, ctx_ckpt, processor.tokenizer, device, batch_size=75
            )
            for local_idx, src_idx in enumerate(valid_indices):
                train_mode_ids[k_idx, src_idx] = ids_k[local_idx]
                train_mode_weights[k_idx, src_idx] = w_k[local_idx]
            del st_pils, hv_valid, ids_k, w_k, q_i8, s_fp16

    cache_obj = {
        "train_hv_i8_23x1608": train_hv_i8,
        "train_hv_scale_23x1608": train_hv_scale,
        "train_mode_ids_23x1608": train_mode_ids,
        "train_mode_weights_23x1608": train_mode_weights,
        "pack": pack,
    }
    torch.save(cache_obj, c_p)
    push_file_to_drive(c_p, c_p.name)
    return cache_obj


# ==============================================================================
# Stock Readout Generation Helper
# ==============================================================================
def generate_and_readout_cg(
    model_s: nn.Module,
    embed_vision: nn.Module,
    tokenizer: Any,
    lm_weight: torch.Tensor,
    src_hv_per_item_np: np.ndarray,
    instructions: List[str],
    head: nn.Module,
    device: torch.device,
    batch_size: int = 16,
    seed_base: int = 42,
) -> np.ndarray:
    """Executes stock_generate_with_readout capturing final_sc_denoising_step readout."""
    if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 90 * (1024 ** 3):
        cur_bs = min(batch_size, 10)
    else:
        cur_bs = min(batch_size, 6)
    m_items = len(instructions)
    model_s.eval()
    head.eval()
    base_m = model_s.get_base_model() if hasattr(model_s, "get_base_model") else model_s
    heads_dict = {"primary": head}

    preds_all = np.zeros((m_items, CANVAS_LEN_SINGLE, HV_DIM), dtype=np.float32)

    # Group by tokenized length to minimize padding overhead
    len_to_idxs: Dict[int, List[int]] = {}
    item_ids: List[torch.Tensor] = []
    for idx_m, inst_str in enumerate(instructions):
        prompt_str = build_image_instruction_prompt(instruction=inst_str)
        ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids
        item_ids.append(ids)
        len_to_idxs.setdefault(int(ids.shape[1]), []).append(idx_m)
    for seq_len, idxs_group in sorted(len_to_idxs.items()):
        pos = 0
        while pos < len(idxs_group):
            b_idxs = idxs_group[pos : pos + cur_bs]
            b_cur = len(b_idxs)
            torch.manual_seed(seed_base + b_idxs[0])
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_base + b_idxs[0])

            input_ids = torch.cat([item_ids[i] for i in b_idxs], dim=0).to(device).contiguous()
            attention_mask = torch.ones_like(input_ids)
            mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).to(torch.int32)

            src_hv_b = torch.from_numpy(np.asarray(src_hv_per_item_np[b_idxs], dtype=np.float32)).to(device)
            mm_out = make_mm_encoder_outputs(embed_vision, src_hv_b, device)

            kw_gen = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mm_token_type_ids": mm_token_type_ids,
                "mm_encoder_outputs": mm_out,
            }
            res_g = stock_generate_with_readout(
                base_m,
                kw_gen,
                heads=heads_dict,
                readout_mode="dual_gated",
                lm_weight=lm_weight,
                readout_tau=1.0,
                max_new_tokens=CANVAS_LEN_SINGLE,
                recorder=_GenRecorder(milestone_steps=(), record_step_telemetry=False),
            )
            p_b = res_g["preds_by_head"]["primary"].detach().cpu().float().numpy()
            for j, orig_m in enumerate(b_idxs):
                preds_all[orig_m] = p_b[j]

            pos += b_cur
            del res_g, kw_gen, mm_out, src_hv_b, input_ids, attention_mask, mm_token_type_ids
            torch.cuda.empty_cache()

    return preds_all


# ==============================================================================
# Experiment A: Training & Evaluation
# ==============================================================================
def train_and_evaluate_cg_dg_arm(
    arm_name: str,
    seed: int,
    enable_lora: bool,
    out_dir: Path,
    device: torch.device,
    base_model: nn.Module,
    model_s: nn.Module,
    dec_mod: nn.Module,
    embed_vision: nn.Module,
    tokenizer: Any,
    lm_weight: torch.Tensor,
    train_ds: Optional[Dict[str, Any]],
    test_ds: Dict[str, Any],
    num_epochs: int = 3,
    only_manifests: Optional[List[str]] = None,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_json = out_dir / "cg_dg_EPOCH_COMPLETE.json"
    if only_manifests is None and done_json.exists():
        print(f"[{arm_name}] Loading cached summary from {done_json}", flush=True)
        return json.loads(done_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 96)
    print(f"EXPERIMENT A DG ARM: {arm_name} (seed={seed}, enable_lora={enable_lora}, epochs={num_epochs})")
    print("=" * 96, flush=True)

    trainable_lora = reset_cg_lora_weights_inplace(model_s, seed=seed, enable_lora=enable_lora)
    model_s.generation_config = base_model.generation_config
    model_s.to(device)
    _, dec_mod, _ = get_dg_modules(model_s)

    head_primary = build_single_canvas_head("dual_gated").to(device)
    head_primary.init_cold_start()

    slim_ckpt = out_dir / f"{arm_name}_slim_ckpt.pt.gz"
    try_pull_from_drive(slim_ckpt.name, slim_ckpt)
    if slim_ckpt.exists():
        print(f"[{arm_name}] Restoring checkpoint from {slim_ckpt}...", flush=True)
        ckpt = load_ckpt_any(slim_ckpt, map_location="cpu")
        if enable_lora and "lora_state_dict" in ckpt:
            for k_p, p_m in model_s.named_parameters():
                if k_p in ckpt["lora_state_dict"]:
                    p_m.data.copy_(ckpt["lora_state_dict"][k_p].to(device=device, dtype=p_m.dtype))
        head_primary.load_state_dict(ckpt["head_state_dict"], strict=True)
        del ckpt
        torch.cuda.empty_cache()
    else:
        if train_ds is None:
            raise RuntimeError(f"[{arm_name}] Checkpoint not found and train_ds is None!")
        train_hv_i8 = train_ds["train_hv_i8_23x1608"]
        train_hv_scale = train_ds["train_hv_scale_23x1608"]
        train_mode_ids = train_ds["train_mode_ids_23x1608"]
        train_mode_weights = train_ds["train_mode_weights_23x1608"]
        train_pack = train_ds["pack"]
        preamble_str = SP.preamble(train_pack)

        n_sources = int(train_hv_i8.shape[1])  # 1608
        n_examples_per_ep = n_sources * 8  # 12,864
        bs = 4 if enable_lora else 8
        accum = 2 if enable_lora else 1
        steps_per_ep = math.ceil((math.ceil(n_examples_per_ep / bs)) / accum)  # 1,608
        total_steps = num_epochs * steps_per_ep  # 4,824

        param_groups = []
        if enable_lora and trainable_lora:
            param_groups.append({"params": trainable_lora, "lr": 1.0e-4, "name": "lora"})
        param_groups.append({"params": list(head_primary.parameters()), "lr": 2.0e-4, "name": "head"})
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)

        step_idx = 0
        t0_train = time.time()
        for ep_num in range(1, num_epochs + 1):
            ep_seed = seed + (ep_num - 1)
            schedule = SP.build_cg_contrastive_schedule(pack=train_pack, epoch_seed=ep_seed)

            for batch_idx, b_start in enumerate(range(0, len(schedule), bs)):
                b_end = min(b_start + bs, len(schedule))
                batch_items = schedule[b_start:b_end]
                b_cur = len(batch_items)

                s_idxs = torch.tensor([x[0] for x in batch_items], dtype=torch.long)
                t23_idxs = torch.tensor([x[1] for x in batch_items], dtype=torch.long)
                b_insts = [f"{preamble_str} {x[3]}" for x in batch_items]

                src_hv_b = dequantize_hv_per_token_int8(train_hv_i8[0, s_idxs], train_hv_scale[0, s_idxs], device)
                tgt_hv_b = dequantize_hv_per_token_int8(train_hv_i8[t23_idxs, s_idxs], train_hv_scale[t23_idxs, s_idxs], device)
                mode_ids_b = train_mode_ids[t23_idxs, s_idxs].to(device)
                mode_weights_b = train_mode_weights[t23_idxs, s_idxs].to(device)

                x0 = sample_x0_from_modes(mode_ids_b, mode_weights_b, w_thresh=DEFAULT_W_THRESH)
                t_vec = torch.rand((b_cur,), device=device)
                xt, _ = corrupt_canvas_multinomial(x0, vocab_size=DG_VOCAB_SIZE, t=t_vec)

                input_ids, attention_mask, mm_token_type_ids, pos_ids = tokenize_instruction_batch(
                    tokenizer, b_insts, device
                )
                mm_out = make_mm_encoder_outputs(embed_vision, src_hv_b, device)
                dec_attn = torch.cat([attention_mask, torch.ones_like(xt)], dim=1)

                kw = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "mm_token_type_ids": mm_token_type_ids,
                    "position_ids": pos_ids,
                    "mm_encoder_outputs": mm_out,
                    "decoder_input_ids": xt,
                    "decoder_attention_mask": dec_attn,
                }

                # Self-conditioning (p=0.5)
                sc_active = bool((torch.rand((), device=device) < 0.5).item())
                sc_logits = None
                sc_mask = None
                if sc_active:
                    with torch.no_grad():
                        res_first = model_s(**kw)
                        sc_logits = res_first.logits.detach().to(torch.bfloat16)
                    sc_mask = torch.ones((b_cur,), device=device, dtype=torch.bool)

                if enable_lora:
                    canvas_logits, h_dec = supervised_forward(
                        model_s, dec_mod, kw, sc_logits=sc_logits, sc_mask=sc_mask, n_tokens=CANVAS_LEN_SINGLE
                    )
                    loss_ce = distribution_ce(
                        canvas_logits, mode_ids_b, mode_weights_b, w_thresh=DEFAULT_W_THRESH, reduction="none"
                    ).mean()
                else:
                    with torch.no_grad():
                        canvas_logits, h_dec = supervised_forward(
                            model_s, dec_mod, kw, sc_logits=sc_logits, sc_mask=sc_mask, n_tokens=CANVAS_LEN_SINGLE
                        )
                    loss_ce = torch.zeros((), device=device)

                ro_in = compute_readout_input(
                    h_dec=h_dec, logits=canvas_logits, readout_mode="dual_gated", lm_weight=lm_weight
                )
                pred_hv = head_primary(ro_in)
                loss_hv = loss_hv_distance(
                    pred_hv, tgt_hv_b, smooth_l1_beta=DEFAULT_SMOOTH_L1_BETA, cos_weight=DEFAULT_COS_WEIGHT
                )

                total_loss = (loss_ce if enable_lora else torch.zeros_like(loss_hv)) + loss_hv
                (total_loss / accum).backward()

                if (batch_idx + 1) % accum == 0 or b_end == len(schedule):
                    clip_params = [p for pg in opt.param_groups for p in pg["params"] if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    opt.step()
                    opt.zero_grad()
                    step_idx += 1

                if step_idx <= 2 or step_idx % 100 == 0:
                    print(
                        f"  [{arm_name} Ep {ep_num}/{num_epochs} Step {step_idx:04d}/{total_steps:04d}] "
                        f"CE={loss_ce.item():.4f} | Hv={loss_hv.item():.4f} | {time.time() - t0_train:.1f}s",
                        flush=True,
                    )

        # Save trained slim checkpoint
        ckpt_payload: Dict[str, Any] = {
            "epoch": num_epochs,
            "seed": seed,
            "enable_lora": enable_lora,
            "head_state_dict": {k: v.detach().cpu() for k, v in head_primary.state_dict().items()},
        }
        if enable_lora:
            ckpt_payload["lora_state_dict"] = {
                k: v.detach().cpu() for k, v in model_s.named_parameters() if v.requires_grad
            }
        save_slim_ckpt_gz(ckpt_payload, slim_ckpt)
        push_file_to_drive(slim_ckpt, slim_ckpt.name)
        model_s.zero_grad(set_to_none=True)
        if 'opt' in locals() and opt is not None:
            del opt
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================================================
    # Evaluation Suite on Experiment A Test Set
    # ==========================================================================
    model_s.zero_grad(set_to_none=True)
    if 'opt' in locals() and opt is not None:
        del opt
    gc.collect()
    torch.cuda.empty_cache()
    model_s.eval()
    head_primary.eval()

    test_pack = test_ds["pack"]
    test_preamble = SP.preamble(test_pack)
    all_tgt_hv = test_ds["all_tgt_hv_33x360"]
    src_hv_test_np = all_tgt_hv[0].float().numpy()
    manifests = SP.build_cg_eval_manifests(test_pack)

    target_manifests = only_manifests or list(manifests.keys())

    import cg_score_preds as CS
    from cg_hv_probe import load_model, read_grids
    reader_path = Path("/content/dgg2/reader.npz")
    if not reader_path.exists():
        reader_path = Path(__file__).resolve().parent.parent / "outputs" / "m1" / "reader.npz"
    reader = load_model(str(reader_path))

    is_click = "click" in str(test_ds.get("pack_path", "")) or (isinstance(test_pack, dict) and "click" in str(test_pack.get("repo", "")))
    suffix = "_click" if is_click else ""
    summary_file = out_dir / f"{arm_name}{suffix}_summary.json"
    eval_results: Dict[str, Any] = {}
    if summary_file.exists():
        try:
            cached_sum = json.loads(summary_file.read_text(encoding="utf-8"))
            if "manifests" in cached_sum and isinstance(cached_sum["manifests"], dict):
                eval_results.update(cached_sum["manifests"])
            else:
                eval_results.update({k: v for k, v in cached_sum.items() if k not in ("arm_name", "manifests")})
        except Exception:
            pass
    elif try_pull_from_drive(summary_file.name, summary_file):
        try:
            cached_sum = json.loads(summary_file.read_text(encoding="utf-8"))
            if "manifests" in cached_sum and isinstance(cached_sum["manifests"], dict):
                eval_results.update(cached_sum["manifests"])
            else:
                eval_results.update({k: v for k, v in cached_sum.items() if k not in ("arm_name", "manifests")})
        except Exception:
            pass

    for m_key in target_manifests:
        if m_key not in manifests:
            continue
        man = manifests[m_key]
        src_idxs = man["img_indices"]
        src_items = src_hv_test_np[src_idxs]

        pred_file = out_dir / f"preds_{arm_name}_{m_key}{suffix}.npy"

        # Check existing cached predictions
        if m_key.startswith("open_"):
            pfiles = [out_dir / f"preds_{arm_name}_{m_key}{suffix}_gen{g}.npy" for g in range(4)]
            for pf in pfiles:
                if not (pf.exists() and pf.stat().st_size > 1_000_000):
                    try_pull_from_drive(pf.name, pf)
            all_open_exist = all(pf.exists() and pf.stat().st_size > 1_000_000 for pf in pfiles)
            if all_open_exist:
                print(f"[Eval {m_key}] Found existing valid prediction {pred_file.name}, loading from disk...", flush=True)
                pred_grids_list = [read_grids(reader, np.load(pf)) for pf in pfiles]
                score = CS.score_open(test_pack, man, pred_grids_list)
            else:
                insts = [f"{test_preamble} {inst}" for inst in man["instructions"]]
                pred_grids_list = []
                for g in range(4):
                    pfile_g = pfiles[g]
                    pred_hv_g = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, src_items, insts, head_primary, device, seed_base=seed + 1000 * g
                    )
                    np.save(pfile_g, pred_hv_g)
                    push_file_to_drive(pfile_g, pfile_g.name)
                    if g == 0:
                        np.save(pred_file, pred_hv_g)
                        push_file_to_drive(pred_file, pred_file.name)
                    pred_grids_list.append(read_grids(reader, pred_hv_g))
                score = CS.score_open(test_pack, man, pred_grids_list)
        else:
            if not (pred_file.exists() and pred_file.stat().st_size > 1_000_000):
                try_pull_from_drive(pred_file.name, pred_file)
            if pred_file.exists() and pred_file.stat().st_size > 1_000_000:
                print(f"[Eval {m_key}] Found existing valid prediction {pred_file.name}, loading from disk...", flush=True)
                pred_hv = np.load(pred_file)
                pred_grids = read_grids(reader, pred_hv)
                score = CS.score_fixed(test_pack, man, pred_grids)
            else:
                if m_key == "refeed_chain_heldout_ops_360":
                    # 3-step chain re-feed: step 1 -> step 2 -> step 3
                    chain_insts = man["chain_instructions"]
                    s1_insts = [f"{test_preamble} {ci[0]}" for ci in chain_insts]
                    s2_insts = [f"{test_preamble} {ci[1]}" for ci in chain_insts]
                    s3_insts = [f"{test_preamble} {ci[2]}" for ci in chain_insts]
                    pred_s1 = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, src_items, s1_insts, head_primary, device, seed_base=seed + 100
                    )
                    pred_s2 = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, pred_s1, s2_insts, head_primary, device, seed_base=seed + 200
                    )
                    pred_hv = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, pred_s2, s3_insts, head_primary, device, seed_base=seed + 300
                    )
                elif m_key.startswith("refeed_"):
                    # 2-step re-feed: step 1 -> step 2
                    s1_insts = [f"{test_preamble} {inst}" for inst in man["step1_instructions"]]
                    s2_insts = [f"{test_preamble} {inst}" for inst in man["step2_instructions"]]
                    pred_s1 = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, src_items, s1_insts, head_primary, device, seed_base=seed + 100
                    )
                    pred_hv = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, pred_s1, s2_insts, head_primary, device, seed_base=seed + 200
                    )
                else:
                    insts = [f"{test_preamble} {inst}" for inst in man["instructions"]]
                    pred_hv = generate_and_readout_cg(
                        model_s, embed_vision, tokenizer, lm_weight, src_items, insts, head_primary, device, seed_base=seed
                    )
                np.save(pred_file, pred_hv)
                push_file_to_drive(pred_file, pred_file.name)
                pred_grids = read_grids(reader, pred_hv)
                score = CS.score_fixed(test_pack, man, pred_grids)

        eval_results[m_key] = score
        acc_str = f"{score['acc_kway']:.4f}" if "acc_kway" in score else f"open(valid={np.mean([v['valid'] for v in score['per_generation'][0].values()]):.3f})"
        print(f"  [{arm_name} Eval {m_key}] Result={acc_str}", flush=True)

        summary_payload = {
            "arm_name": arm_name,
            "manifests": eval_results,
            **eval_results,
        }
        summary_file.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        push_file_to_drive(summary_file, summary_file.name)

    summary_payload = {
        "arm_name": arm_name,
        "manifests": eval_results,
        **eval_results,
    }
    done_json = out_dir / (f"{arm_name}{suffix}_completed.json" if is_click else "eval_completed.json")
    done_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    push_file_to_drive(done_json, done_json.name)
    return eval_results


# ==============================================================================
# Experiment B: Symbol Vector Input Adapter & Readout Head
# ==============================================================================
class ExpBInputAdapter(nn.Module):
    """Modified model: trained input adapter mapping standardised symbol vectors
    (d=384) to DiffusionGemma soft tokens (2816) per HANDOFF.md §6.5."""

    def __init__(self, in_dim: int = 384, out_dim: int = 2816, rms_scale: float = 1.28):
        super().__init__()
        self.ln = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.rms_scale = float(rms_scale)
        nn.init.normal_(self.linear.weight, mean=0.0, std=1.0 / math.sqrt(in_dim))
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(self.ln(x))
        rms = out.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return (out / rms) * self.rms_scale


def resolve_expB_path(filename: str) -> Path:
    candidates = [
        Path(f"/content/dgg2/data/{filename}"),
        Path(f"/content/dgg2/{filename}"),
        Path(f"data/{filename}"),
        Path(filename),
    ]
    for c in candidates:
        if c.exists():
            return c
    local_p = Path(f"/content/dgg2/data/{filename}")
    if try_pull_from_drive(filename, local_p):
        return local_p
    p = Path(filename)
    if p.exists():
        return p
    return candidates[0]


def generate_and_readout_expB(
    base_model: nn.Module,
    adapter: nn.Module,
    tokenizer: Any,
    lm_weight: torch.Tensor,
    src_vecs_np: np.ndarray,
    instructions: List[str],
    head_node: nn.Module,
    device: torch.device,
    c_toks: Optional[torch.Tensor] = None,
    seed_base: int = 42,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates predictions and discrete canvas tokens for Experiment B symbol vectors."""
    if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory >= 90 * (1024 ** 3):
        cur_bs = min(batch_size, 10)
    else:
        cur_bs = min(batch_size, 6)
    n_items = len(instructions)
    base_m = base_model.get_base_model() if hasattr(base_model, "get_base_model") else base_model
    base_m.eval()
    adapter.eval()
    head_node.eval()

    preds = np.zeros((n_items, 256, 384), dtype=np.float32)
    gen_canvases = np.zeros((n_items, 256), dtype=np.int64)
    canvas_3logits_all = np.zeros((n_items, 256, 3), dtype=np.float32)

    # Group by tokenized length to eliminate padding overhead
    len_to_idxs: Dict[int, List[int]] = {}
    item_ids: List[torch.Tensor] = []
    for idx_m, inst_str in enumerate(instructions):
        prompt_str = build_image_instruction_prompt(instruction=inst_str)
        ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids
        item_ids.append(ids)
        len_to_idxs.setdefault(int(ids.shape[1]), []).append(idx_m)
    with torch.no_grad():
        for seq_len, idxs_group in sorted(len_to_idxs.items()):
            pos = 0
            while pos < len(idxs_group):
                b_idxs = idxs_group[pos : pos + cur_bs]
                torch.manual_seed(seed_base + b_idxs[0])
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_base + b_idxs[0])

                input_ids = torch.cat([item_ids[i] for i in b_idxs], dim=0).to(device).contiguous()
                attention_mask = torch.ones_like(input_ids)
                mm_token_type_ids = (input_ids == IMAGE_TOKEN_ID).to(torch.int32)

                b_srcs = torch.from_numpy(np.asarray(src_vecs_np[b_idxs], dtype=np.float32)).to(device)
                softs = adapter(b_srcs).to(dtype=torch.bfloat16).reshape(-1, 2816)
                from transformers.modeling_outputs import BaseModelOutputWithPooling
                mm_out = {"image": BaseModelOutputWithPooling(pooler_output=softs)}
                kw_gen = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "mm_token_type_ids": mm_token_type_ids,
                    "mm_encoder_outputs": mm_out,
                }
                res_g = stock_generate_with_readout(
                    base_m,
                    kw_gen,
                    heads={"primary": head_node},
                    readout_mode="dual_gated",
                    lm_weight=lm_weight,
                    readout_tau=1.0,
                    max_new_tokens=256,
                    recorder=_GenRecorder(milestone_steps=(), record_step_telemetry=False),
                )
                p_b = res_g["preds_by_head"]["primary"].detach().cpu().float().numpy()
                for j, orig_m in enumerate(b_idxs):
                    preds[orig_m] = p_b[j]

                c_gen = res_g.get("canvas")
                if c_gen is not None:
                    c_b = c_gen.detach().cpu().numpy()
                    for j, orig_m in enumerate(b_idxs):
                        gen_canvases[orig_m] = c_b[j]

                lg = res_g.get("logits")
                if lg is not None and c_toks is not None:
                    lg_3 = lg[:, :, c_toks].detach().cpu().float().numpy()
                    for j, orig_m in enumerate(b_idxs):
                        canvas_3logits_all[orig_m] = lg_3[j]

                pos += cur_bs

    return preds, gen_canvases, canvas_3logits_all


def aggregate_per_primary_symbol(
    pack: dict,
    items: Sequence[Tuple[int, Any]],
    per_item_rows: List[dict],
) -> Dict[str, Any]:
    """Computes macro-average metrics across unique primary symbols (156 symbols in test360_httpx)."""
    if not per_item_rows or len(items) != len(per_item_rows):
        return {}
    by_symbol: Dict[Any, List[dict]] = defaultdict(list)
    for (i, _), row in zip(items, per_item_rows):
        src_obj = pack["sources"][i]
        if "primary_id" in src_obj:
            sym_id = src_obj["primary_id"]
        elif "primary_slot" in src_obj:
            sym_id = src_obj["slot_text"][src_obj["primary_slot"]]
        elif "sites" in src_obj:
            sym_id = tuple(st.get("primary_id", st.get("primary_slot", idx)) for idx, st in enumerate(src_obj["sites"]))
        else:
            sym_id = i
        by_symbol[sym_id].append(row)

    keys = ("site_hit", "all_sites_hit", "top_is_site", "changed_recall", "secondary_hit", "drift")
    sym_aggs: Dict[str, List[float]] = {k: [] for k in keys}
    for sym_id, s_rows in by_symbol.items():
        for k in keys:
            vals = [r[k] for r in s_rows if k in r and not np.isnan(r[k])]
            if vals:
                sym_aggs[k].append(float(np.mean(vals)))

    res = {
        "n_symbols": len(by_symbol),
        **{f"{k}_per_symbol": float(np.mean(sym_aggs[k])) if sym_aggs[k] else float("nan") for k in keys},
    }
    return res


def compute_expB_canvas_metrics(
    gen_canvases: np.ndarray,
    canvas_3logits: np.ndarray,
    true_plans: np.ndarray,
    c_toks: torch.Tensor,
) -> Dict[str, float]:
    """Computes discrete 3-class canvas accuracy (0=unchanged, 1=edited, 2=deleted)."""
    c_toks_arr = c_toks.detach().cpu().numpy()
    target_canvas = c_toks_arr[true_plans]
    tok_acc = float((gen_canvases == target_canvas).mean())

    pred_3class = canvas_3logits.argmax(axis=-1)
    class_acc = float((pred_3class == true_plans).mean())

    changed_mask = (true_plans > 0)
    changed_acc = float((pred_3class[changed_mask] == true_plans[changed_mask]).mean()) if changed_mask.any() else 1.0

    return {
        "canvas_tok_match_acc": tok_acc,
        "canvas_3class_acc": class_acc,
        "canvas_changed_acc": changed_acc,
    }


def _compute_vocab_readout_and_ce(
    canvas_logits: torch.Tensor,
    h_dec: torch.Tensor,
    lm_weight: torch.Tensor,
    target_canvas_ids: torch.Tensor,
    enable_lora: bool,
    tau: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Activation-checkpointed helper computing vocab readout chords, confidence stats, and CE loss.

    Frees >3.0 GiB of (B, 256, 262144) autograd buffers during the backward pass.
    """
    b_cur, n_tok, v_size = canvas_logits.shape
    chunk_size = 1
    e_soft_list = []
    c_z_list = []
    loss_ce_list = []

    for c_start in range(0, b_cur, chunk_size):
        c_end = min(c_start + chunk_size, b_cur)
        c_logits = canvas_logits[c_start:c_end]
        probs = F.softmax(c_logits.float() / float(tau), dim=-1)
        e_soft_chunk = torch.matmul(probs.to(lm_weight.dtype), lm_weight).float()
        e_soft_list.append(e_soft_chunk)

        with torch.no_grad():
            p_max = probs.max(dim=-1, keepdim=True).values
            ent = -(probs * probs.clamp(min=1e-12).log()).sum(dim=-1, keepdim=True)
            log_v = float(math.log(max(2, probs.shape[-1])))
            conf_ent = (1.0 - ent / log_v).clamp(0.0, 1.0)
            c_z_chunk = torch.cat([p_max, conf_ent], dim=-1).float()
            c_z_list.append(c_z_chunk)

        if enable_lora:
            c_targets = target_canvas_ids[c_start:c_end]
            ce_chunk = F.cross_entropy(c_logits.reshape(-1, v_size), c_targets.flatten(), reduction="sum")
            loss_ce_list.append(ce_chunk)

    e_soft = torch.cat(e_soft_list, dim=0)
    c_z = torch.cat(c_z_list, dim=0)
    if enable_lora:
        loss_ce = torch.stack(loss_ce_list).sum() / float(b_cur * n_tok)
    else:
        loss_ce = torch.zeros((), device=canvas_logits.device, dtype=torch.float32)

    return h_dec, e_soft, c_z, loss_ce


def train_and_evaluate_expB_dg_arm(
    arm_name: str,
    seed: int,
    enable_lora: bool,
    out_dir: Path,
    device: torch.device,
    base_model: nn.Module,
    model_s: nn.Module,
    dec_mod: nn.Module,
    tokenizer: Any,
    lm_weight: torch.Tensor,
    train_ds_path: Optional[str] = None,
    test_ds_path: Optional[str] = None,
    emb_path: Optional[str] = None,
    num_epochs: int = 3,
    eval_only: bool = False,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    done_json = out_dir / "cg_expB_EPOCH_COMPLETE.json"
    if done_json.exists() and not eval_only:
        print(f"[{arm_name}] Loading cached summary from {done_json}", flush=True)
        return json.loads(done_json.read_text(encoding="utf-8"))

    print("\n" + "=" * 96)
    print(f"EXPERIMENT B DG ARM: {arm_name} (seed={seed}, enable_lora={enable_lora}, epochs={num_epochs}, eval_only={eval_only})")
    print("=" * 96, flush=True)

    import cg_node_metrics as NM
    import cg_node_gate as NG

    p_train = Path(train_ds_path) if train_ds_path else resolve_expB_path("nodeB_train1608.json.gz")
    p_test = Path(test_ds_path) if test_ds_path else resolve_expB_path("nodeB_test360_httpx.json.gz")
    p_emb = Path(emb_path) if emb_path else resolve_expB_path("emb.npz")

    with gzip.open(p_train, "rt") as f:
        train_pack = json.load(f)
    with gzip.open(p_test, "rt") as f:
        test_pack = json.load(f)

    emb = NG.Emb(str(p_emb))
    train_ev = NM.Evaluator(train_pack, emb)
    test_ev = NM.Evaluator(test_pack, emb, stats_vec=train_ev.emb.vec)

    trainable_lora = reset_cg_lora_weights_inplace(model_s, seed=seed, enable_lora=enable_lora)
    adapter = ExpBInputAdapter(in_dim=384, out_dim=2816).to(device)
    head_node = DualReadoutSoftCanvasHead(d_out=384).to(device)
    head_node.init_cold_start()

    c_toks = torch.tensor([
        tokenizer.encode("_", add_special_tokens=False)[0],
        tokenizer.encode("+", add_special_tokens=False)[0],
        tokenizer.encode("-", add_special_tokens=False)[0],
    ], device=device, dtype=torch.long)

    slim_ckpt = out_dir / f"{arm_name}_slim_ckpt.pt.gz"
    try_pull_from_drive(slim_ckpt.name, slim_ckpt)

    if slim_ckpt.exists():
        print(f"[{arm_name}] Loading trained weights from {slim_ckpt}", flush=True)
        ckpt_payload = load_ckpt_any(slim_ckpt, map_location="cpu")
        if "adapter_state_dict" in ckpt_payload:
            adapter.load_state_dict(ckpt_payload["adapter_state_dict"])
        if "head_state_dict" in ckpt_payload:
            head_node.load_state_dict(ckpt_payload["head_state_dict"])
        if enable_lora and "lora_state_dict" in ckpt_payload:
            for k_p, p_m in model_s.named_parameters():
                if k_p in ckpt_payload["lora_state_dict"]:
                    p_m.data.copy_(ckpt_payload["lora_state_dict"][k_p].to(device=device, dtype=p_m.dtype))
        del ckpt_payload
        torch.cuda.empty_cache()
    elif eval_only:
        print(f"[{arm_name}][WARN] No checkpoint found at {slim_ckpt}; evaluating cold weights", flush=True)
    else:
        param_groups = []
        if enable_lora and trainable_lora:
            param_groups.append({"params": trainable_lora, "lr": 1.0e-4, "name": "lora"})
        param_groups.append({"params": list(adapter.parameters()), "lr": 2.0e-4, "name": "adapter"})
        param_groups.append({"params": list(head_node.parameters()), "lr": 2.0e-4, "name": "head"})
        opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.0)

        n_sources = len(train_pack["sources"])  # 1608
        n_cands_per_source = 8
        n_examples_per_ep = n_sources * n_cands_per_source  # 12864
        if device.type == "cuda" and torch.cuda.get_device_properties(device).total_memory < 90 * (1024 ** 3):
            bs = 4
            accum = 2
        else:
            bs = 4
            accum = 2
        steps_per_ep = n_examples_per_ep // (bs * accum)    # exactly 1608 steps
        total_steps = num_epochs * steps_per_ep             # exactly 4824 steps

        step_idx = 0
        t0_train = time.time()
        first_backward_verified = False

        for ep_num in range(1, num_epochs + 1):
            pairs = [(i, k) for i in range(n_sources) for k in range(n_cands_per_source)]
            rng = np.random.RandomState(seed + ep_num)
            rng.shuffle(pairs)

            for b_start in range(0, len(pairs), bs):
                b_pairs = pairs[b_start : min(b_start + bs, len(pairs))]
                b_cur = len(b_pairs)
                b_src_idxs = [p[0] for p in b_pairs]
                b_k = [p[1] for p in b_pairs]

                b_src_vecs = torch.from_numpy(np.stack([train_ev.source(i) for i in b_src_idxs])).to(device, dtype=torch.float32)
                b_tgt_vecs = torch.from_numpy(np.stack([train_ev.target(i, k) for i, k in zip(b_src_idxs, b_k)])).to(device, dtype=torch.float32)
                b_weights = torch.from_numpy(np.stack([train_ev.loss_weights(i, k) for i, k in zip(b_src_idxs, b_k)])).to(device, dtype=torch.float32)
                b_plan = torch.from_numpy(np.stack([train_ev.canvas_labels(i, k) for i, k in zip(b_src_idxs, b_k)])).to(device, dtype=torch.long)

                target_canvas_ids = c_toks[b_plan]
                xt, _ = corrupt_canvas_multinomial(target_canvas_ids, vocab_size=DG_VOCAB_SIZE, t=torch.rand((b_cur,), device=device))

                b_insts = [
                    train_pack["sources"][i]["cands"][k]["instr_seen"][
                        rng.randint(0, len(train_pack["sources"][i]["cands"][k]["instr_seen"]))
                    ]
                    for i, k in zip(b_src_idxs, b_k)
                ]
                input_ids, attention_mask, mm_token_type_ids, pos_ids = tokenize_instruction_batch(tokenizer, b_insts, device)

                soft_tokens = adapter(b_src_vecs).to(dtype=torch.bfloat16).reshape(-1, 2816)
                from transformers.modeling_outputs import BaseModelOutputWithPooling
                mm_out = {"image": BaseModelOutputWithPooling(pooler_output=soft_tokens)}
                dec_attn = torch.cat([attention_mask, torch.ones_like(xt)], dim=1)

                kw = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "mm_token_type_ids": mm_token_type_ids,
                    "position_ids": pos_ids,
                    "mm_encoder_outputs": mm_out,
                    "decoder_input_ids": xt,
                    "decoder_attention_mask": dec_attn,
                }

                sc_active = bool((torch.rand((), device=device) < 0.5).item())
                sc_logits = None
                sc_mask = None
                if sc_active:
                    with torch.no_grad():
                        res_first = model_s(**kw)
                        sc_logits = res_first.logits.detach().to(torch.bfloat16)
                        del res_first
                    sc_mask = torch.ones((b_cur,), device=device, dtype=torch.bool)

                # Both LoRA and Frozen DG run supervised_forward with grad enabled so gradients flow to adapter
                canvas_logits, h_dec = supervised_forward(
                    model_s, dec_mod, kw, sc_logits=sc_logits, sc_mask=sc_mask, n_tokens=256
                )

                # Checkpoint vocab-space readout chords and CE loss (frees >3.0 GiB of autograd buffers)
                h_d, e_s, c_z, loss_ce = cp.checkpoint(
                    _compute_vocab_readout_and_ce,
                    canvas_logits,
                    h_dec,
                    lm_weight,
                    target_canvas_ids,
                    enable_lora,
                    use_reentrant=False,
                )

                ro_in = (h_d, e_s, c_z)
                pred_vecs = head_node(ro_in)  # (B, 256, 384)
                diff = F.smooth_l1_loss(pred_vecs, b_tgt_vecs, beta=0.01, reduction="none").mean(dim=-1)
                cos_dist = 1.0 - F.cosine_similarity(pred_vecs, b_tgt_vecs, dim=-1)
                loss_vec = ((diff + cos_dist) * b_weights).sum(dim=-1).mean()

                total_loss = (loss_ce if enable_lora else torch.zeros_like(loss_vec)) + 5.0 * loss_vec
                (total_loss / accum).backward()

                if not first_backward_verified:
                    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()), "Adapter received zero gradient!"
                    print(f"  [{arm_name}] Verified: adapter received non-zero gradients on first backward pass", flush=True)
                    first_backward_verified = True

                if (b_start // bs + 1) % accum == 0 or (b_start + bs) >= len(pairs):
                    clip_params = [p for pg in opt.param_groups for p in pg["params"] if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
                    opt.step()
                    opt.zero_grad()
                    step_idx += 1

                if step_idx <= 2 or step_idx % 100 == 0:
                    print(f"  [{arm_name} Ep {ep_num}/{num_epochs} Step {step_idx:04d}/{total_steps:04d}] CE={loss_ce.item():.4f} | VecLoss={loss_vec.item():.4f} | {time.time() - t0_train:.1f}s", flush=True)

        # Save checkpoint
        ckpt_payload = {
            "epoch": num_epochs,
            "seed": seed,
            "enable_lora": enable_lora,
            "adapter_state_dict": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
            "head_state_dict": {k: v.detach().cpu() for k, v in head_node.state_dict().items()},
        }
        if enable_lora:
            ckpt_payload["lora_state_dict"] = {k: v.detach().cpu() for k, v in model_s.named_parameters() if v.requires_grad}
        save_slim_ckpt_gz(ckpt_payload, slim_ckpt)
        push_file_to_drive(slim_ckpt, slim_ckpt.name)
        model_s.zero_grad(set_to_none=True)
        if 'opt' in locals() and opt is not None:
            del opt
        gc.collect()
        torch.cuda.empty_cache()

    # ==========================================================================
    # Full Experiment B Evaluation Suite
    # Covers: instr_seen, instr_unseen, unseenargs, renameclass, multi_k2, multi_k4
    # ==========================================================================
    model_s.zero_grad(set_to_none=True)
    if 'opt' in locals() and opt is not None:
        del opt
    gc.collect()
    torch.cuda.empty_cache()
    model_s.eval()
    adapter.eval()
    head_node.eval()

    eval_results: Dict[str, Any] = {}

    # 1. instr_seen on test360 httpx
    items_seen = NM.default_items(test_pack)
    src_vecs_seen = np.stack([test_ev.source(i) for i, _ in items_seen])
    insts_seen = [test_pack["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items_seen]
    preds_seen, canvases_seen, lg_seen = generate_and_readout_expB(
        base_model, adapter, tokenizer, lm_weight, src_vecs_seen, insts_seen, head_node, device, c_toks=c_toks, seed_base=seed + 10
    )
    score_seen = test_ev.score(preds_seen, items_seen)
    plans_seen = np.stack([test_ev.canvas_labels(i, k) for i, k in items_seen])
    score_seen.update(compute_expB_canvas_metrics(canvases_seen, lg_seen, plans_seen, c_toks))
    score_seen.update(aggregate_per_primary_symbol(test_pack, items_seen, score_seen.get("per_item", [])))
    pfile_seen = out_dir / f"preds_{arm_name}_instr_seen.npy"
    np.save(pfile_seen, preds_seen)
    push_file_to_drive(pfile_seen, pfile_seen.name)
    eval_results["instr_seen"] = score_seen
    print(f"  [{arm_name} Eval instr_seen] Site Hit={score_seen.get('site_hit', 0.0):.4f} (per-sym={score_seen.get('site_hit_per_symbol', 0.0):.4f}) | Top-is-site={score_seen.get('top_is_site', 0.0):.4f} | Canvas Acc={score_seen.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 2. instr_unseen on test360 httpx
    insts_unseen = [test_pack["sources"][i]["cands"][k]["instr_unseen"][0] for i, k in items_seen]
    preds_unseen, canvases_unseen, lg_unseen = generate_and_readout_expB(
        base_model, adapter, tokenizer, lm_weight, src_vecs_seen, insts_unseen, head_node, device, c_toks=c_toks, seed_base=seed + 20
    )
    score_unseen = test_ev.score(preds_unseen, items_seen)
    score_unseen.update(compute_expB_canvas_metrics(canvases_unseen, lg_unseen, plans_seen, c_toks))
    score_unseen.update(aggregate_per_primary_symbol(test_pack, items_seen, score_unseen.get("per_item", [])))
    pfile_unseen = out_dir / f"preds_{arm_name}_instr_unseen.npy"
    np.save(pfile_unseen, preds_unseen)
    push_file_to_drive(pfile_unseen, pfile_unseen.name)
    eval_results["instr_unseen"] = score_unseen
    print(f"  [{arm_name} Eval instr_unseen] Site Hit={score_unseen.get('site_hit', 0.0):.4f} (per-sym={score_unseen.get('site_hit_per_symbol', 0.0):.4f}) | Top-is-site={score_unseen.get('top_is_site', 0.0):.4f} | Canvas Acc={score_unseen.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 2b. Shuffled instruction control (instr_shuf) on test360 httpx
    rng_shuf = np.random.RandomState(seed + 999)
    insts_shuf = list(insts_seen)
    rng_shuf.shuffle(insts_shuf)
    preds_shuf, canvases_shuf, lg_shuf = generate_and_readout_expB(
        base_model, adapter, tokenizer, lm_weight, src_vecs_seen, insts_shuf, head_node, device, c_toks=c_toks, seed_base=seed + 50
    )
    score_shuf = test_ev.score(preds_shuf, items_seen)
    score_shuf.update(compute_expB_canvas_metrics(canvases_shuf, lg_shuf, plans_seen, c_toks))
    score_shuf.update(aggregate_per_primary_symbol(test_pack, items_seen, score_shuf.get("per_item", [])))
    pfile_shuf = out_dir / f"preds_{arm_name}_instr_shuf.npy"
    np.save(pfile_shuf, preds_shuf)
    push_file_to_drive(pfile_shuf, pfile_shuf.name)
    eval_results["instr_shuf"] = score_shuf
    print(f"  [{arm_name} Eval instr_shuf] Site Hit={score_shuf.get('site_hit', 0.0):.4f} (per-sym={score_shuf.get('site_hit_per_symbol', 0.0):.4f}) | Top-is-site={score_shuf.get('top_is_site', 0.0):.4f} | Canvas Acc={score_shuf.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 2c. Floors and ceilings (source copy, training-mean vector, oracle target)
    tgt_vecs_seen = np.stack([test_ev.target(i, k) for i, k in items_seen])
    zero_vecs_seen = np.zeros_like(src_vecs_seen)

    score_floor_src = test_ev.score(src_vecs_seen, items_seen)
    score_floor_src.update(aggregate_per_primary_symbol(test_pack, items_seen, score_floor_src.get("per_item", [])))
    eval_results["floor_source_copy"] = score_floor_src

    score_floor_mean = test_ev.score(zero_vecs_seen, items_seen)
    score_floor_mean.update(aggregate_per_primary_symbol(test_pack, items_seen, score_floor_mean.get("per_item", [])))
    eval_results["floor_training_mean"] = score_floor_mean

    score_ceil_oracle = test_ev.score(tgt_vecs_seen, items_seen)
    score_ceil_oracle.update(aggregate_per_primary_symbol(test_pack, items_seen, score_ceil_oracle.get("per_item", [])))
    eval_results["ceiling_oracle_target"] = score_ceil_oracle
    print(f"  [{arm_name} Baselines] Source Copy: site_hit={score_floor_src.get('site_hit', 0.0):.4f} | Mean Vec: site_hit={score_floor_mean.get('site_hit', 0.0):.4f} | Oracle: site_hit={score_ceil_oracle.get('site_hit', 0.0):.4f}", flush=True)

    # 3. nodeB_test360_httpx_unseenargs
    p_unseenargs = resolve_expB_path("nodeB_test360_httpx_unseenargs.json.gz")
    if p_unseenargs.exists():
        with gzip.open(p_unseenargs, "rt") as f:
            unseenargs_pack = json.load(f)
        unseenargs_ev = NM.Evaluator(unseenargs_pack, emb, stats_vec=train_ev.emb.vec)
        items_uargs = NM.default_items(unseenargs_pack)
        src_vecs_uargs = np.stack([unseenargs_ev.source(i) for i, _ in items_uargs])
        insts_uargs = [unseenargs_pack["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items_uargs]
        preds_uargs, canvases_uargs, lg_uargs = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, src_vecs_uargs, insts_uargs, head_node, device, c_toks=c_toks, seed_base=seed + 30
        )
        score_uargs = unseenargs_ev.score(preds_uargs, items_uargs)
        plans_uargs = np.stack([unseenargs_ev.canvas_labels(i, k) for i, k in items_uargs])
        score_uargs.update(compute_expB_canvas_metrics(canvases_uargs, lg_uargs, plans_uargs, c_toks))
        score_uargs.update(aggregate_per_primary_symbol(unseenargs_pack, items_uargs, score_uargs.get("per_item", [])))
        pfile_uargs = out_dir / f"preds_{arm_name}_nodeB_test360_httpx_unseenargs.npy"
        np.save(pfile_uargs, preds_uargs)
        push_file_to_drive(pfile_uargs, pfile_uargs.name)
        eval_results["nodeB_test360_httpx_unseenargs"] = score_uargs
        print(f"  [{arm_name} Eval unseenargs] Site Hit={score_uargs.get('site_hit', 0.0):.4f} (per-sym={score_uargs.get('site_hit_per_symbol', 0.0):.4f}) | Top-is-site={score_uargs.get('top_is_site', 0.0):.4f} | Canvas Acc={score_uargs.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 4. nodeB_test180_httpx_renameclass
    p_renameclass = resolve_expB_path("nodeB_test180_httpx_renameclass.json.gz")
    if p_renameclass.exists():
        with gzip.open(p_renameclass, "rt") as f:
            renameclass_pack = json.load(f)
        rc_ev = NM.Evaluator(renameclass_pack, emb, stats_vec=train_ev.emb.vec)
        items_rc = NM.default_items(renameclass_pack)
        src_vecs_rc = np.stack([rc_ev.source(i) for i, _ in items_rc])
        insts_rc = [renameclass_pack["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items_rc]
        preds_rc, canvases_rc, lg_rc = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, src_vecs_rc, insts_rc, head_node, device, c_toks=c_toks, seed_base=seed + 40
        )
        score_rc = rc_ev.score(preds_rc, items_rc)
        plans_rc = np.stack([rc_ev.canvas_labels(i, k) for i, k in items_rc])
        score_rc.update(compute_expB_canvas_metrics(canvases_rc, lg_rc, plans_rc, c_toks))
        score_rc.update(aggregate_per_primary_symbol(renameclass_pack, items_rc, score_rc.get("per_item", [])))
        pfile_rc = out_dir / f"preds_{arm_name}_nodeB_test180_httpx_renameclass.npy"
        np.save(pfile_rc, preds_rc)
        push_file_to_drive(pfile_rc, pfile_rc.name)
        eval_results["nodeB_test180_httpx_renameclass"] = score_rc
        print(f"  [{arm_name} Eval renameclass] Site Hit={score_rc.get('site_hit', 0.0):.4f} (per-sym={score_rc.get('site_hit_per_symbol', 0.0):.4f}) | Top-is-site={score_rc.get('top_is_site', 0.0):.4f} | Canvas Acc={score_rc.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 5. nodeB_multi_test360_httpx_k2 (2-step re-feed through adapter)
    p_k2 = resolve_expB_path("nodeB_multi_test360_httpx_k2.json.gz")
    if p_k2.exists():
        with gzip.open(p_k2, "rt") as f:
            k2_pack = json.load(f)
        k2_ev = NM.Evaluator(k2_pack, emb, stats_vec=train_ev.emb.vec)
        items_k2 = NM.default_items(k2_pack)
        s_vecs_k2 = np.stack([k2_ev.source(i) for i, _ in items_k2])
        # 1-pass: direct multi-symbol instruction
        s_insts_k2_1p = [k2_pack["sources"][i]["instr_seen"][0] for i in range(len(items_k2))]
        pred_k2_1p, canvases_k2_1p, lg_k2_1p = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, s_vecs_k2, s_insts_k2_1p, head_node, device, c_toks=c_toks, seed_base=seed + 150
        )
        score_k2_1p = k2_ev.score(pred_k2_1p, items_k2)
        plans_k2 = np.stack([k2_ev.canvas_labels(i) for i, _ in items_k2])
        score_k2_1p.update(compute_expB_canvas_metrics(canvases_k2_1p, lg_k2_1p, plans_k2, c_toks))
        score_k2_1p.update(aggregate_per_primary_symbol(k2_pack, items_k2, score_k2_1p.get("per_item", [])))
        pfile_k2_1p = out_dir / f"preds_{arm_name}_nodeB_multi_test360_httpx_k2_1pass.npy"
        np.save(pfile_k2_1p, pred_k2_1p)
        push_file_to_drive(pfile_k2_1p, pfile_k2_1p.name)
        eval_results["multi_k2_1pass"] = score_k2_1p
        eval_results["nodeB_multi_test360_httpx_k2_1pass"] = score_k2_1p
        print(f"  [{arm_name} Eval multi_k2_1pass] All Sites Hit={score_k2_1p.get('all_sites_hit', 0.0):.4f} | Site Hit={score_k2_1p.get('site_hit', 0.0):.4f} (per-sym={score_k2_1p.get('site_hit_per_symbol', 0.0):.4f}) | Canvas Acc={score_k2_1p.get('canvas_3class_acc', 0.0):.4f}", flush=True)

        # 2-step re-feeding through adapter
        s1_insts_k2 = [k2_pack["sources"][i]["sites"][0]["instr_seen"][0] for i in range(len(items_k2))]
        pred_k2_s1, _, _ = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, s_vecs_k2, s1_insts_k2, head_node, device, c_toks=c_toks, seed_base=seed + 100
        )
        s2_insts_k2 = [k2_pack["sources"][i]["sites"][1]["instr_seen"][0] for i in range(len(items_k2))]
        pred_k2_s2, canvases_k2, lg_k2 = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, pred_k2_s1, s2_insts_k2, head_node, device, c_toks=c_toks, seed_base=seed + 200
        )
        score_k2 = k2_ev.score(pred_k2_s2, items_k2)
        score_k2.update(compute_expB_canvas_metrics(canvases_k2, lg_k2, plans_k2, c_toks))
        score_k2.update(aggregate_per_primary_symbol(k2_pack, items_k2, score_k2.get("per_item", [])))
        pfile_k2 = out_dir / f"preds_{arm_name}_nodeB_multi_test360_httpx_k2.npy"
        np.save(pfile_k2, pred_k2_s2)
        push_file_to_drive(pfile_k2, pfile_k2.name)
        eval_results["multi_k2_refeed"] = score_k2
        eval_results["nodeB_multi_test360_httpx_k2"] = score_k2
        print(f"  [{arm_name} Eval multi_k2_refeed] All Sites Hit={score_k2.get('all_sites_hit', 0.0):.4f} | Site Hit={score_k2.get('site_hit', 0.0):.4f} (per-sym={score_k2.get('site_hit_per_symbol', 0.0):.4f}) | Canvas Acc={score_k2.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    # 6. nodeB_multi_test360_httpx_k4 (1-pass and 4-step re-feed through adapter)
    p_k4 = resolve_expB_path("nodeB_multi_test360_httpx_k4.json.gz")
    if p_k4.exists():
        with gzip.open(p_k4, "rt") as f:
            k4_pack = json.load(f)
        k4_ev = NM.Evaluator(k4_pack, emb, stats_vec=train_ev.emb.vec)
        items_k4 = NM.default_items(k4_pack)
        s_vecs_k4 = np.stack([k4_ev.source(i) for i, _ in items_k4])
        plans_k4 = np.stack([k4_ev.canvas_labels(i) for i, _ in items_k4])

        # 1-pass: direct multi-symbol instruction
        s_insts_k4_1p = [k4_pack["sources"][i]["instr_seen"][0] for i in range(len(items_k4))]
        pred_k4_1p, canvases_k4_1p, lg_k4_1p = generate_and_readout_expB(
            base_model, adapter, tokenizer, lm_weight, s_vecs_k4, s_insts_k4_1p, head_node, device, c_toks=c_toks, seed_base=seed + 250
        )
        score_k4_1p = k4_ev.score(pred_k4_1p, items_k4)
        score_k4_1p.update(compute_expB_canvas_metrics(canvases_k4_1p, lg_k4_1p, plans_k4, c_toks))
        score_k4_1p.update(aggregate_per_primary_symbol(k4_pack, items_k4, score_k4_1p.get("per_item", [])))
        pfile_k4_1p = out_dir / f"preds_{arm_name}_nodeB_multi_test360_httpx_k4_1pass.npy"
        np.save(pfile_k4_1p, pred_k4_1p)
        push_file_to_drive(pfile_k4_1p, pfile_k4_1p.name)
        eval_results["multi_k4_1pass"] = score_k4_1p
        eval_results["nodeB_multi_test360_httpx_k4_1pass"] = score_k4_1p
        print(f"  [{arm_name} Eval multi_k4_1pass] All Sites Hit={score_k4_1p.get('all_sites_hit', 0.0):.4f} | Site Hit={score_k4_1p.get('site_hit', 0.0):.4f} (per-sym={score_k4_1p.get('site_hit_per_symbol', 0.0):.4f}) | Canvas Acc={score_k4_1p.get('canvas_3class_acc', 0.0):.4f}", flush=True)

        # 4-step re-feeding through adapter
        cur_vecs_k4 = s_vecs_k4.copy()
        canvases_k4 = None
        lg_k4 = None
        for step_i in range(4):
            insts_k4_step = [k4_pack["sources"][i]["sites"][step_i]["instr_seen"][0] for i in range(len(items_k4))]
            cur_vecs_k4, canvases_k4, lg_k4 = generate_and_readout_expB(
                base_model, adapter, tokenizer, lm_weight, cur_vecs_k4, insts_k4_step, head_node, device, c_toks=c_toks, seed_base=seed + 100 * (step_i + 1)
            )
        score_k4 = k4_ev.score(cur_vecs_k4, items_k4)
        score_k4.update(compute_expB_canvas_metrics(canvases_k4, lg_k4, plans_k4, c_toks))
        score_k4.update(aggregate_per_primary_symbol(k4_pack, items_k4, score_k4.get("per_item", [])))
        pfile_k4 = out_dir / f"preds_{arm_name}_nodeB_multi_test360_httpx_k4.npy"
        np.save(pfile_k4, cur_vecs_k4)
        push_file_to_drive(pfile_k4, pfile_k4.name)
        eval_results["multi_k4_refeed"] = score_k4
        eval_results["nodeB_multi_test360_httpx_k4"] = score_k4
        print(f"  [{arm_name} Eval multi_k4_refeed] All Sites Hit={score_k4.get('all_sites_hit', 0.0):.4f} | Site Hit={score_k4.get('site_hit', 0.0):.4f} (per-sym={score_k4.get('site_hit_per_symbol', 0.0):.4f}) | Canvas Acc={score_k4.get('canvas_3class_acc', 0.0):.4f}", flush=True)

    summary_file = out_dir / f"{arm_name}_summary.json"
    summary_file.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    push_file_to_drive(summary_file, summary_file.name)
    done_json.write_text(json.dumps(eval_results, indent=2), encoding="utf-8")
    return eval_results


# ==============================================================================
# Main Entry Point
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "train_lora_dg_expA",
            "train_frozen_dg_expA",
            "eval_dg_expA",
            "train_lora_dg_expB",
            "train_frozen_dg_expB",
            "eval_dg_expB",
        ],
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--arm-name", default=None, help="Explicit arm name override")
    p.add_argument("--manifests", nargs="+", default=None)
    p.add_argument("--test-pack", default="/content/dgg2/data/gridA_test360_httpx.npz", help="Path to test pack npz")
    p.add_argument("--model-id", default="google/diffusiongemma-26B-A4B-it", help="Model repo or local dir")
    p.add_argument("--base-output-dir", default="/content/dgg2/output_cg")
    return p


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.base_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 96)
    print(f"DIFFUSIONGEMMA CODE-GRAPH RUNNER (MODE: {args.mode}, DEVICE: {device})")
    print("=" * 96, flush=True)

    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from dg_mask_ab import _load_model_streaming
    import torch.utils.checkpoint as cp

    model_dir = getattr(args, "model_id", "google/diffusiongemma-26B-A4B-it")
    if not Path(model_dir).exists():
        model_dir = snapshot_download("google/diffusiongemma-26B-A4B-it", token=False)
    print(f"[Model] Using model snapshot at {model_dir}", flush=True)

    # Load 26B streaming
    base_model = _load_model_streaming(model_dir, log=print).to(device)
    gc_stock, _ = load_stock_generation_config(model_dir)
    base_model.generation_config = gc_stock
    base_model.eval()

    # Clip linear replace & grouped_mm MoE
    for _, parent in base_model.named_modules():
        for cn, ch in list(parent.named_children()):
            if isinstance(ch, Gemma4ClippableLinear):
                setattr(parent, cn, ch.linear)

    for mod in base_model.modules():
        if hasattr(mod, "config") and hasattr(mod, "gate_up_proj") and hasattr(mod, "num_experts"):
            mod.config._experts_implementation = "grouped_mm"

    # Activation checkpointing (decoder only; encoder writes shared KV cache and must not be checkpointed)
    mm = getattr(base_model, "model", base_model)
    dec = getattr(mm, "decoder", None)
    if dec is not None:
        layers = getattr(dec, "layers", None)
        if layers is None and hasattr(dec, "language_model"):
            layers = getattr(dec.language_model, "layers", None)
        if layers is not None:
            for layer in layers:
                orig = layer.forward
                if not getattr(orig, "_act_ckpt", False):
                    def _fwd(*a, __orig=orig, **kw):
                        if not torch.is_grad_enabled():
                            return __orig(*a, **kw)
                        return cp.checkpoint(__orig, *a, use_reentrant=False, preserve_rng_state=False, **kw)
                    _fwd._act_ckpt = True
                    layer.forward = _fwd

    processor = AutoProcessor.from_pretrained(model_dir)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
    lm_weight = base_model.get_input_embeddings().weight.detach()

    enc_mod = getattr(mm, "encoder", None)
    embed_vision = getattr(enc_mod, "embed_vision", getattr(mm, "embed_vision", None))
    embed_vision.eval()
    for p_ev in embed_vision.parameters():
        p_ev.requires_grad_(False)

    if args.mode in ("train_lora_dg_expA", "train_frozen_dg_expA", "eval_dg_expA"):
        # Load ctx teacher checkpoint for teacher modes (optional if pre-encoded caches exist)
        ctx_ckpt = None
        p2_path = Path("/content/dgg2/phase2_train1608_and_ctx_pack.pt")
        if p2_path.exists():
            p2_pack = torch.load(p2_path, map_location="cpu", weights_only=False)
            ctx_ckpt = p2_pack.get("ctx_ckpt")
            del p2_pack

        # Prepare caches
        test_ds = prepare_cg_test360_cache(
            pack_path=args.test_pack,
            processor=processor,
            base_model=base_model,
            device=device,
            ctx_ckpt=ctx_ckpt,
        )
        if args.mode == "eval_dg_expA":
            train_ds = {}
        else:
            train_ds = prepare_cg_train1608_cache(
                pack_path="/content/dgg2/data/gridA_train1608.npz",
                processor=processor,
                base_model=base_model,
                device=device,
                ctx_ckpt=ctx_ckpt,
            )

        enable_lora = (args.mode in ("train_lora_dg_expA", "eval_dg_expA") if args.arm_name is None else "lora" in args.arm_name)
        arm_name = args.arm_name or (
            f"cg_lora_dg_seed{args.seed}_ep{args.num_epochs}"
            if enable_lora
            else f"cg_frozen_dg_ep{args.num_epochs}"
        )

        model_s = create_cold_start_lora(base_model)
        model_s.generation_config = base_model.generation_config
        model_s.to(device)
        _, dec_mod, _ = get_dg_modules(model_s)

        train_and_evaluate_cg_dg_arm(
            arm_name=arm_name,
            seed=args.seed,
            enable_lora=enable_lora,
            out_dir=out_dir / arm_name,
            device=device,
            base_model=base_model,
            model_s=model_s,
            dec_mod=dec_mod,
            embed_vision=embed_vision,
            tokenizer=processor.tokenizer,
            lm_weight=lm_weight,
            train_ds=train_ds,
            test_ds=test_ds,
            num_epochs=args.num_epochs,
            only_manifests=args.manifests,
        )

    elif args.mode in ("train_lora_dg_expB", "train_frozen_dg_expB", "eval_dg_expB"):
        enable_lora = (args.mode in ("train_lora_dg_expB", "eval_dg_expB") if args.arm_name is None else "lora" in args.arm_name)
        eval_only = (args.mode == "eval_dg_expB")
        arm_name = args.arm_name or (
            f"cg_expB_lora_dg_seed{args.seed}_ep{args.num_epochs}"
            if enable_lora
            else f"cg_expB_frozen_dg_ep{args.num_epochs}"
        )

        model_s = create_cold_start_lora(base_model)
        model_s.generation_config = base_model.generation_config
        model_s.to(device)
        _, dec_mod, _ = get_dg_modules(model_s)

        train_and_evaluate_expB_dg_arm(
            arm_name=arm_name,
            seed=args.seed,
            enable_lora=enable_lora,
            out_dir=out_dir / arm_name,
            device=device,
            base_model=base_model,
            model_s=model_s,
            dec_mod=dec_mod,
            tokenizer=processor.tokenizer,
            lm_weight=lm_weight,
            num_epochs=args.num_epochs,
            eval_only=eval_only,
        )

    print("\n[cg_colab_runner] All requested tasks completed successfully!", flush=True)


if __name__ == "__main__":
    main()
