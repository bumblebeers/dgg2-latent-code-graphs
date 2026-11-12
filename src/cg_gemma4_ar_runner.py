#!/usr/bin/env python3
"""src/cg_gemma4_ar_runner.py -- Gemma 4 Autoregressive Baseline Runner.

Implements two matched Gemma 4 (26B total / 4B active MoE) autoregressive baselines:
1. Arm 1 (--mode prompted):
   - Gemma4ForConditionalGeneration + AutoProcessor in bfloat16.
   - 8-class palette legend (0=empty, 1=mod_header, 2=function, 3=class, 4=method, 5=test, 6=call_edge, 7=marked).
   - 2 deterministic in-context worked examples from data/gridA_train1608.npz (zero test leakage).
   - Test item's rendered 768x768 RGB image + text source grid + instruction.
   - Single-digit tokenization invariance check.
   - Robust parser parse_16x16_grid(text, fallback_src_grid) tracking exact_256_parse_rate.
   - Batched KV-cached evaluation (bs=32/36) across 9 Exp A splits:
     * seen_bal360
     * seen_shuf360
     * unseen_para_bal360
     * unseen_infer_360
     * heldout_reverse_order+delete_selected_360
     * heldout_reverse_edges+keep_callees_360
     * refeed_heldout_comp_bal360 (2-step visual refeed)
     * refeed_chain_heldout_ops_360 (3-step chain refeed)
     * open_unseen_360 (4 stochastic seeds for open-ended diversity)
   - Scoring with cg/cg_score_preds.py into outputs/m2/arms/gemma4_ar_prompted/.

2. Arm 2 (--mode lora_train_and_eval):
   - Gemma4ForConditionalGeneration in bfloat16 with matched LoRA (r=64, alpha=128, dropout=0.0).
   - target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"].
   - Strips LoRA from vision/audio/multimodal projectors and freezes MoE routers.
   - Records exact trainable and frozen parameter counts.
   - Perceptual input matched to Exp A: ONLY rendered 768x768 RGB image of src_grid + text instruction.
     (NO cheat text copy of src_grid!).
   - Causal CE only over 16x16 target palette grid tokens. Prompt/image masked with -100.
   - 3 epochs = 4,824 optimizer steps at effective bs=8 (e.g. micro_bs=4, accum=2).
   - Benchmark Steps 1–20 speed and ETA.
   - Checkpoints after each epoch.
   - Evaluation across all 9 splits into outputs/m2/arms/gemma4_ar_lora_r64_s42/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure cg modules are importable
for _d in [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../cg")),
    "/content/dgg2",
    "/content/dgg2/cg",
    "/content/dgg2/bundle",
]:
    if os.path.exists(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

import cg_ops as O  # noqa: E402
import cg_sec27_spec as SP  # noqa: E402
import cg_score_preds as CS  # noqa: E402
from cg_metrics import open_outcome, open_summary, open_diversity  # noqa: E402


# -----------------------------------------------------------------------------
# Security & Token Handling
# -----------------------------------------------------------------------------
def find_hf_token() -> Optional[str]:
    """Find Hugging Face access token securely without printing or logging."""
    # 1. Environment variable
    env_tok = os.environ.get("HF_TOKEN")
    if env_tok and len(env_tok.strip()) > 8:
        return env_tok.strip()

    # 2. ~/.tokens/HF_TOKEN
    p1 = os.path.expanduser("~/.tokens/HF_TOKEN")
    if os.path.exists(p1):
        try:
            with open(p1, "r") as f:
                t = f.read().strip()
                if t:
                    return t
        except Exception:
            pass

    # 3. ~/.cache/huggingface/token
    p2 = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(p2):
        try:
            with open(p2, "r") as f:
                t = f.read().strip()
                if t:
                    return t
        except Exception:
            pass

    return None


# -----------------------------------------------------------------------------
# Palette Legend & Grid Text Formatting
# -----------------------------------------------------------------------------
PALETTE_LEGEND = (
    "8-class palette legend:\n"
    "0 = background (no symbol on diagonal / no call edge off-diagonal)\n"
    "1 = call_edge (off-diagonal cell (r, c) == 1 means symbol r calls symbol c)\n"
    "2 = function (diagonal (i, i))\n"
    "3 = method (diagonal (i, i))\n"
    "4 = class (diagonal (i, i))\n"
    "5 = test (diagonal (i, i))\n"
    "6 = selected (diagonal (i, i): highlighted target symbol for selected-node / reachability edits)\n"
    "7 = marked (diagonal (i, i): symbol highlighted by mark_reachable / mark_ancestors)\n"
    "Each call graph is represented on a 16x16 grid of integers from 0 to 7."
)


def format_grid_text(grid: np.ndarray) -> str:
    """Format a 16x16 numpy array into 16 rows of 16 single-digit characters."""
    assert grid.shape == (16, 16), f"Expected (16, 16), got {grid.shape}"
    lines = []
    for r in range(16):
        lines.append("".join(str(int(grid[r, c])) for c in range(16)))
    return "\n".join(lines)


def parse_16x16_grid(text: str, fallback_src_grid: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Robustly parse a 16x16 grid of digits (0-7) from model output.

    Returns:
        (parsed_grid, is_exact_256)
    """
    # 1. Strip thought channel if present
    cleaned = text
    if "<|channel>thought" in cleaned and "<channel|>" in cleaned:
        cleaned = cleaned.split("<channel|>")[-1]
    elif "<thought>" in cleaned and "</thought>" in cleaned:
        cleaned = cleaned.split("</thought>")[-1]

    # 2. Strip code fences if present
    cleaned = re.sub(r"```[a-zA-Z]*", "", cleaned)
    cleaned = cleaned.replace("```", "")

    # 3. Try line-by-line parsing: look for lines containing 16 digits
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    grid_rows = []
    for line in lines:
        row_digits = re.findall(r"[0-7]", line)
        if len(row_digits) == 16:
            grid_rows.append([int(d) for d in row_digits])
        elif len(row_digits) > 16:
            grid_rows.append([int(d) for d in row_digits[:16]])
        if len(grid_rows) == 16:
            break

    if len(grid_rows) == 16:
        out = np.array(grid_rows, dtype=np.uint8)
        return out, True

    # 4. Fallback: extract all digits [0-7] across the whole text
    all_digits = [int(d) for d in re.findall(r"[0-7]", cleaned)]
    if len(all_digits) >= 256:
        out = np.array(all_digits[:256], dtype=np.uint8).reshape(16, 16)
        return out, (len(all_digits) == 256)

    # 5. Partial fallback: populate available digits, fill remainder from fallback_src_grid
    out = fallback_src_grid.copy().astype(np.uint8)
    if len(all_digits) > 0:
        flat = out.flatten()
        n = min(len(all_digits), 256)
        flat[:n] = all_digits[:n]
        out = flat.reshape(16, 16)

    return out, False


def build_in_context_examples(train_pack: Dict[str, np.ndarray]) -> str:
    """Build 2 deterministic in-context worked examples from data/gridA_train1608.npz."""
    # Example 1: train item 0 with reverse_edges
    ex1_idx = 0
    ex1_src = train_pack["src_grid"][ex1_idx]
    ex1_op = "reverse_edges"
    ex1_tgt_idx = SP.CG_CANDIDATE_STATES.index(ex1_op)
    ex1_tgt = train_pack["tgt_grid"][ex1_idx, ex1_tgt_idx]
    ex1_inst = "Reverse the direction of all call edges in the graph."

    # Example 2: train item 1 with delete_tests
    ex2_idx = 1
    ex2_src = train_pack["src_grid"][ex2_idx]
    ex2_op = "delete_tests"
    ex2_tgt_idx = SP.CG_CANDIDATE_STATES.index(ex2_op)
    ex2_tgt = train_pack["tgt_grid"][ex2_idx, ex2_tgt_idx]
    ex2_inst = "Delete all test functions from the call graph."

    ex_text = (
        "--- In-Context Example 1 ---\n"
        f"Instruction: {ex1_inst}\n"
        "Source Grid:\n"
        f"{format_grid_text(ex1_src)}\n"
        "Target Grid:\n"
        f"{format_grid_text(ex1_tgt)}\n\n"
        "--- In-Context Example 2 ---\n"
        f"Instruction: {ex2_inst}\n"
        "Source Grid:\n"
        f"{format_grid_text(ex2_src)}\n"
        "Target Grid:\n"
        f"{format_grid_text(ex2_tgt)}\n\n"
    )
    return ex_text


def build_prompted_user_text(
    legend: str,
    in_context_text: str,
    src_grid: np.ndarray,
    instruction: str,
) -> str:
    """Build user turn text for Arm 1 (Prompted)."""
    return (
        f"{legend}\n\n"
        "You are given a 768x768 RGB image and text representation of a 16x16 call graph source grid, along with an edit instruction.\n"
        "Apply the edit instruction to the call graph and output ONLY the resulting 16x16 target grid as 16 rows of 16 digits (0-7).\n"
        "Do not output thought reasoning, explanations, or markdown fences. Output ONLY the 16x16 digit grid.\n\n"
        f"{in_context_text}"
        "--- Target Task ---\n"
        f"Instruction: {instruction}\n"
        "Source Grid:\n"
        f"{format_grid_text(src_grid)}\n"
        "Target Grid:\n"
    )


# -----------------------------------------------------------------------------
# Model & LoRA Loading
# -----------------------------------------------------------------------------
def load_gemma4_processor(model_id: str, token: Optional[str] = None):
    """Load Gemma 4 AutoProcessor and verify single-digit tokenization."""
    from transformers import AutoProcessor

    print(f"[cg_gemma4_ar] Loading AutoProcessor for {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Verify single-digit tokenization invariance
    digit_tokens = tokenizer.encode("01234567", add_special_tokens=False)
    print(f"[cg_gemma4_ar] Encoded '01234567': {digit_tokens} (len={len(digit_tokens)})")
    assert len(digit_tokens) == 8, f"Expected 8 single-digit tokens, got {len(digit_tokens)}: {digit_tokens}"
    for ch in "01234567":
        ch_tok = tokenizer.encode(ch, add_special_tokens=False)
        assert len(ch_tok) == 1, f"Digit {ch} did not encode to a single token: {ch_tok}"

    return processor


def load_gemma4_model_prompted(
    model_id: str,
    device: str = "cuda:0",
    token: Optional[str] = None,
):
    """Load Gemma 4 model for Arm 1 (Prompted) in bfloat16."""
    from transformers import Gemma4ForConditionalGeneration

    print(f"[cg_gemma4_ar] Loading Gemma4ForConditionalGeneration in bfloat16 on {device}...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=token,
    )
    model.eval()
    vram = torch.cuda.memory_allocated(device) / (1024**3) if "cuda" in device else 0
    print(f"[cg_gemma4_ar] Model loaded for Arm 1. VRAM allocated: {vram:.2f} GB")
    return model


def load_gemma4_model_lora(
    model_id: str,
    r: int = 64,
    alpha: int = 128,
    dropout: float = 0.0,
    device: str = "cuda:0",
    token: Optional[str] = None,
):
    """Load Gemma 4 model for Arm 2 (LoRA) with matched PEFT configuration."""
    from transformers import Gemma4ForConditionalGeneration
    from peft import LoraConfig, get_peft_model

    print(f"[cg_gemma4_ar] Loading Gemma4ForConditionalGeneration in bfloat16 on {device}...")
    model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=token,
    )

    # Matched LoRA config: r=64, alpha=128, dropout=0.0 on language MoE layers
    # Target only language model linear projections (stripping vision, audio, and multimodal projectors)
    lm_targets = [
        n for n, m in model.named_modules()
        if "language_model.layers" in n
        and type(m).__name__ == "Linear"
        and any(n.endswith("." + t) for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    ]
    print(f"[cg_gemma4_ar] Identified {len(lm_targets)} language model linear target modules.")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=lm_targets,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    # Strip LoRA from vision, audio, multimodal projectors and freeze MoE routers
    all_params = 0
    trainable_params = 0
    stripped_modules = []
    for name, param in peft_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            # Check if parameter is on non-language-model components or MoE router
            if any(bad in name.lower() for bad in ["vision", "audio", "embed_vision", "multi_modal_projector", "router", "gate."]):
                param.requires_grad = False
                stripped_modules.append(name)
            else:
                trainable_params += param.numel()

    frozen_params = all_params - trainable_params
    pct = 100.0 * trainable_params / all_params if all_params > 0 else 0
    print(f"[cg_gemma4_ar] LoRA Trainable Parameters: {trainable_params:,} ({pct:.2f}%)")
    print(f"[cg_gemma4_ar] LoRA Frozen Parameters:    {frozen_params:,} ({100.0 - pct:.2f}%)")
    print(f"[cg_gemma4_ar] Total Parameters:           {all_params:,}")
    if stripped_modules:
        print(f"[cg_gemma4_ar] Stripped non-LM modules: {len(stripped_modules)}")

    if hasattr(peft_model, "gradient_checkpointing_enable"):
        try:
            peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[cg_gemma4_ar] Gradient checkpointing enabled successfully.")
        except Exception as e:
            print("[cg_gemma4_ar] Gradient checkpointing warning:", e)

    return peft_model, trainable_params, frozen_params


# -----------------------------------------------------------------------------
# Arm 1: Prompted Evaluation
# -----------------------------------------------------------------------------
def evaluate_manifest_prompted(
    model,
    processor,
    test_pack: Dict[str, np.ndarray],
    manifest_name: str,
    manifest: Dict[str, Any],
    legend: str,
    in_context_text: str,
    batch_size: int = 32,
    device: str = "cuda:0",
    stochastic_seed: Optional[int] = None,
) -> Tuple[np.ndarray, float]:
    """Evaluate one manifest using batched KV-cached decoding.

    Returns:
        (pred_grids, exact_parse_rate)
    """
    n_items = len(manifest["img_indices"])
    pred_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
    exact_parses = 0

    do_sample = stochastic_seed is not None
    if do_sample:
        torch.manual_seed(stochastic_seed)
    eos_ids = list({int(processor.tokenizer.eos_token_id), 106}) if processor.tokenizer.eos_token_id is not None else [106]

    # Check for multi-step manifests
    is_2step_refeed = "step1_instructions" in manifest and "step2_instructions" in manifest
    is_3step_chain = "chain_instructions" in manifest

    if is_2step_refeed:
        # Step 1
        s1_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
        for b_start in range(0, n_items, batch_size):
            b_end = min(b_start + batch_size, n_items)
            b_imgs = [O.render_pil(test_pack["src_grid"][manifest["img_indices"][i]]) for i in range(b_start, b_end)]
            b_prompts = []
            for i in range(b_start, b_end):
                src_g = test_pack["src_grid"][manifest["img_indices"][i]]
                inst = manifest["step1_instructions"][i]
                u_text = build_prompted_user_text(legend, in_context_text, src_g, inst)
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

            inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=288,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    eos_token_id=eos_ids,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for j, text in enumerate(decoded):
                idx = b_start + j
                fallback = test_pack["src_grid"][manifest["img_indices"][idx]]
                g, _ = parse_16x16_grid(text, fallback)
                s1_grids[idx] = g
            del inputs, outputs, gen_ids, decoded
            torch.cuda.empty_cache()

        # Step 2: Feed s1_grids as input image & text
        for b_start in range(0, n_items, batch_size):
            b_end = min(b_start + batch_size, n_items)
            b_imgs = [O.render_pil(s1_grids[i]) for i in range(b_start, b_end)]
            b_prompts = []
            for i in range(b_start, b_end):
                src_g = s1_grids[i]
                inst = manifest["step2_instructions"][i]
                u_text = build_prompted_user_text(legend, in_context_text, src_g, inst)
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

            inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=288,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    eos_token_id=eos_ids,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for j, text in enumerate(decoded):
                idx = b_start + j
                fallback = s1_grids[idx]
                g, exact = parse_16x16_grid(text, fallback)
                pred_grids[idx] = g
                if exact:
                    exact_parses += 1
            del inputs, outputs, gen_ids, decoded
            torch.cuda.empty_cache()
            if (b_start // batch_size) % 2 == 0 or b_end == n_items:
                print(f"[{manifest_name}] Step 2 Batch {b_start//batch_size + 1}/{(n_items + batch_size - 1)//batch_size} | Exact parses: {exact_parses}/{b_end} ({(exact_parses/b_end)*100:.1f}%)", flush=True)

    elif is_3step_chain:
        curr_grids = np.array([test_pack["src_grid"][manifest["img_indices"][i]] for i in range(n_items)], dtype=np.uint8)
        chain_len = len(manifest["chain_instructions"][0])  # 3
        for step_idx in range(chain_len):
            next_grids = np.zeros_like(curr_grids)
            for b_start in range(0, n_items, batch_size):
                b_end = min(b_start + batch_size, n_items)
                b_imgs = [O.render_pil(curr_grids[i]) for i in range(b_start, b_end)]
                b_prompts = []
                for i in range(b_start, b_end):
                    src_g = curr_grids[i]
                    inst = manifest["chain_instructions"][i][step_idx]
                    u_text = build_prompted_user_text(legend, in_context_text, src_g, inst)
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                    b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=288,
                        do_sample=do_sample,
                        temperature=0.7 if do_sample else None,
                        top_p=0.95 if do_sample else None,
                        eos_token_id=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for j, text in enumerate(decoded):
                    idx = b_start + j
                    fallback = curr_grids[idx]
                    g, exact = parse_16x16_grid(text, fallback)
                    next_grids[idx] = g
                    if step_idx == chain_len - 1 and exact:
                        exact_parses += 1
                del inputs, outputs, gen_ids, decoded
                torch.cuda.empty_cache()
            curr_grids = next_grids
            print(f"[{manifest_name}] Completed Chain Step {step_idx + 1}/{chain_len}", flush=True)
        pred_grids = curr_grids

    else:
        # Standard 1-pass manifests
        for b_start in range(0, n_items, batch_size):
            b_end = min(b_start + batch_size, n_items)
            b_imgs = [O.render_pil(test_pack["src_grid"][manifest["img_indices"][i]]) for i in range(b_start, b_end)]
            b_prompts = []
            for i in range(b_start, b_end):
                src_g = test_pack["src_grid"][manifest["img_indices"][i]]
                inst = manifest["instructions"][i]
                u_text = build_prompted_user_text(legend, in_context_text, src_g, inst)
                msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

            inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=288,
                    do_sample=do_sample,
                    temperature=0.7 if do_sample else None,
                    top_p=0.95 if do_sample else None,
                    eos_token_id=eos_ids,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
            decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            for j, text in enumerate(decoded):
                idx = b_start + j
                fallback = test_pack["src_grid"][manifest["img_indices"][idx]]
                g, exact = parse_16x16_grid(text, fallback)
                pred_grids[idx] = g
                if exact:
                    exact_parses += 1
            del inputs, outputs, gen_ids, decoded
            torch.cuda.empty_cache()
            if (b_start // batch_size) % 2 == 0 or b_end == n_items:
                print(f"[{manifest_name}] Batch {b_start//batch_size + 1}/{(n_items + batch_size - 1)//batch_size} ({b_end}/{n_items} items) | Exact parses: {exact_parses}/{b_end} ({(exact_parses/b_end)*100:.1f}%)", flush=True)

    exact_rate = float(exact_parses) / float(n_items)
    return pred_grids, exact_rate


def run_prompted_arm(
    model_id: str,
    train_pack_path: str,
    test_pack_path: str,
    output_dir: str,
    batch_size: int = 32,
    device: str = "cuda:0",
    token: Optional[str] = None,
    splits: Optional[List[str]] = None,
):
    """Run full evaluation suite for Arm 1 (Prompted) across Exp A splits."""
    os.makedirs(output_dir, exist_ok=True)
    preds_dir = os.path.join(output_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    print(f"[cg_gemma4_ar] Loading data packs: {train_pack_path}, {test_pack_path}")
    train_pack = dict(np.load(train_pack_path))
    test_pack = dict(np.load(test_pack_path))
    manifests = SP.build_cg_eval_manifests(test_pack)

    processor = load_gemma4_processor(model_id, token=token)
    model = load_gemma4_model_prompted(model_id, device=device, token=token)

    legend = PALETTE_LEGEND
    in_context_text = build_in_context_examples(train_pack)

    target_splits = splits if splits else [
        "seen_bal360",
        "unseen_para_bal360",
        "unseen_infer_360",
        "refeed_heldout_comp_bal360",
        "seen_shuf360",
        "heldout_reverse_order+delete_selected_360",
        "heldout_reverse_edges+keep_callees_360",
        "refeed_chain_heldout_ops_360",
        "open_unseen_360",
    ]

    all_summaries: Dict[str, Any] = {}

    eval_batch_size = min(batch_size, 36)
    if batch_size > 36:
        print(f"[cg_gemma4_ar] Note: Clamping eval batch size from {batch_size} to safe KV cache limit 36")

    for split in target_splits:
        if split not in manifests:
            print(f"[cg_gemma4_ar] Warning: split {split} not found in manifests; skipping.")
            continue
        split_out_file = os.path.join(output_dir, f"{split}_summary.json")
        if os.path.exists(split_out_file):
            print(f"[cg_gemma4_ar] Split {split} already exists at {split_out_file}; loading and skipping.")
            with open(split_out_file) as f:
                all_summaries[split] = json.load(f)
            continue

        print(f"\n{'='*70}\n[cg_gemma4_ar] Evaluating Arm 1 on split: {split}\n{'='*70}")
        man = manifests[split]
        t0 = time.time()

        if split == "open_unseen_360":
            # 4 stochastic seeds for open-ended evaluation
            seeds = [42, 1042, 2042, 3042]
            gen_grids_list = []
            exact_rates = []
            for s_idx, s_val in enumerate(seeds):
                print(f"[cg_gemma4_ar] Open generation {s_idx+1}/4 (seed={s_val})...")
                g_arr, ex_r = evaluate_manifest_prompted(
                    model, processor, test_pack, split, man, legend, in_context_text,
                    batch_size=eval_batch_size, device=device, stochastic_seed=s_val,
                )
                gen_grids_list.append(g_arr)
                exact_rates.append(ex_r)
                np.save(os.path.join(preds_dir, f"{split}_seed{s_val}.npy"), g_arr)

            summary = CS.score_open(test_pack, man, gen_grids_list)
            summary["exact_256_parse_rate"] = float(np.mean(exact_rates))
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_prompted"
            summary["manifest"] = split

            # Print concise summary
            valid_rates = [v.get("valid", 0.0) for v in summary["per_generation"][0].values()] if summary.get("per_generation") else [0.0]
            print(f"[cg_gemma4_ar] {split}: Mean Valid Rate={np.mean(valid_rates):.4f} | Exact Parse={summary['exact_256_parse_rate']:.4f} | Time={summary['elapsed_s']:.1f}s")

        else:
            pred_grids, exact_rate = evaluate_manifest_prompted(
                model, processor, test_pack, split, man, legend, in_context_text,
                batch_size=eval_batch_size, device=device, stochastic_seed=None,
            )
            np.save(os.path.join(preds_dir, f"{split}.npy"), pred_grids)
            summary = CS.score_fixed(test_pack, man, pred_grids)
            summary["exact_256_parse_rate"] = exact_rate
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_prompted"
            summary["manifest"] = split

            print(f"[cg_gemma4_ar] {split}: K-way Acc={summary.get('acc_kway', 0.0):.4f} | Changed Acc={summary.get('changed_acc', 0.0):.4f} | Exact={summary.get('exact', 0.0):.4f} | Parse={exact_rate:.4f} | Time={summary['elapsed_s']:.1f}s")

        # Save individual split summary
        split_out_file = os.path.join(output_dir, f"{split}_summary.json")
        with open(split_out_file, "w") as f:
            json.dump(summary, f, indent=2)
        all_summaries[split] = summary

    # Save master manifest summary
    master_file = os.path.join(output_dir, "manifest_summary.json")
    with open(master_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n[cg_gemma4_ar] Arm 1 Prompted Evaluation Complete. Summary saved to {master_file}")


# -----------------------------------------------------------------------------
# Arm 2: LoRA Train & Eval
# -----------------------------------------------------------------------------
def run_lora_train_and_eval_arm(
    model_id: str,
    train_pack_path: str,
    test_pack_path: str,
    output_dir: str,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 64,
    lora_alpha: int = 128,
    micro_batch_size: int = 4,
    accum_steps: int = 2,
    eval_batch_size: int = 32,
    device: str = "cuda:0",
    seed: int = 42,
    token: Optional[str] = None,
    splits: Optional[List[str]] = None,
):
    """Run LoRA fine-tuning and evaluation for Arm 2 across 9 Exp A splits."""
    os.makedirs(output_dir, exist_ok=True)
    preds_dir = os.path.join(output_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    ckpts_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpts_dir, exist_ok=True)

    print(f"[cg_gemma4_ar] Loading data packs: {train_pack_path}, {test_pack_path}")
    train_pack = dict(np.load(train_pack_path))
    test_pack = dict(np.load(test_pack_path))
    manifests = SP.build_cg_eval_manifests(test_pack)

    processor = load_gemma4_processor(model_id, token=token)
    peft_model, n_trainable, n_frozen = load_gemma4_model_lora(
        model_id, r=lora_r, alpha=lora_alpha, device=device, token=token,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, peft_model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    # Log image token count
    dummy_img = O.render_pil(train_pack["src_grid"][0])
    dummy_inputs = processor(
        text=processor.apply_chat_template([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "test"}]}], add_generation_prompt=True, tokenize=False),
        images=[[dummy_img]],
        return_tensors="pt"
    )
    img_tok_id = getattr(peft_model.config, "image_token_id", 258880)
    image_token_count = int((dummy_inputs["input_ids"] == img_tok_id).sum().item())
    print(f"[cg_gemma4_ar] Visual Soft Token Count per 768x768 image: {image_token_count}")

    # Build Preamble
    inst_preamble = SP.preamble(train_pack)

    global_step = 0
    total_steps = epochs * (len(train_pack["src_grid"]) * 8 // (micro_batch_size * accum_steps))
    print(f"[cg_gemma4_ar] Training Schedule: {epochs} epochs, {total_steps} optimizer steps (effective bs={micro_batch_size * accum_steps})")

    peft_model.train()
    step_times = []
    benchmarked = False
    cutoff_hit = False
    t_train_start = time.time()
    if eval_batch_size > 12:
        print(f"[cg_gemma4_ar] Note: Clamping eval batch size from {eval_batch_size} to safe KV cache limit 12")
    eval_batch_size = min(eval_batch_size, 12)

    for ep in range(epochs):
        ep_seed = seed + ep
        schedule = SP.build_cg_contrastive_schedule(train_pack, epoch_seed=ep_seed, ops_per_source_per_epoch=8)
        n_examples = len(schedule)
        print(f"\n[cg_gemma4_ar] --- Starting Epoch {ep+1}/{epochs} ({n_examples} examples) ---")

        ep_loss = 0.0
        optimizer.zero_grad()

        for step_idx in range(0, n_examples, micro_batch_size * accum_steps):
            t_step_start = time.time()
            accum_loss = 0.0

            for acc in range(accum_steps):
                b_start = step_idx + acc * micro_batch_size
                b_end = min(b_start + micro_batch_size, n_examples)
                if b_start >= b_end:
                    break

                batch_items = schedule[b_start:b_end]
                b_imgs = [O.render_pil(train_pack["src_grid"][item[0]]) for item in batch_items]

                # Perceptual input matched to Exp A: ONLY rendered image + text instruction. NO cheat text copy of src_grid!
                b_prompts_user = []
                b_targets = []
                for item in batch_items:
                    s_idx, train_state_idx, op_name, inst = item
                    tgt_state = SP.CG_TRAIN_STATES[train_state_idx]
                    tgt_g = SP.state_grid(train_pack, s_idx, tgt_state)
                    u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                    b_prompts_user.append(u_text)
                    b_targets.append(format_grid_text(tgt_g))

                # Build full conversational text with target response
                b_full_texts = []
                for u_text, tgt_text in zip(b_prompts_user, b_targets):
                    msg_full = [
                        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]},
                        {"role": "model", "content": tgt_text},
                    ]
                    b_full_texts.append(processor.apply_chat_template(msg_full, add_generation_prompt=False, tokenize=False))

                inputs_full = processor(text=b_full_texts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)

                # Create labels with causal CE ONLY over target grid tokens; prompt/image masked with -100
                labels = inputs_full["input_ids"].clone()
                for b_i in range(len(batch_items)):
                    row = inputs_full["input_ids"][b_i].tolist()
                    found_start = False
                    for idx in range(len(row) - 3):
                        if row[idx:idx+3] == [105, 4368, 107]:
                            labels[b_i, :idx+3] = -100
                            found_start = True
                            break
                    if not found_start:
                        labels[b_i, :] = -100
                    # Also mask padding tokens
                    labels[b_i, inputs_full["attention_mask"][b_i] == 0] = -100

                outputs = peft_model(**inputs_full, labels=labels)
                loss = outputs.loss / accum_steps
                loss.backward()
                accum_loss += loss.item()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            ep_loss += accum_loss

            dt = time.time() - t_step_start
            step_times.append(dt)

            # Benchmark Steps 1–20 speed and report ETA
            if global_step == 20 and not benchmarked:
                mean_s_step = float(np.mean(step_times[:20]))
                rem_steps = total_steps - global_step
                eta_s = rem_steps * mean_s_step
                print(f"\n{'*'*70}\n[cg_gemma4_ar] BENCHMARK STEPS 1–20:\n"
                      f"  Average Speed: {mean_s_step:.4f} s/step\n"
                      f"  Remaining Steps: {rem_steps}\n"
                      f"  Estimated Training ETA: {eta_s / 60:.1f} minutes ({eta_s / 3600:.2f} hours)\n"
                      f"{'*'*70}\n")
                benchmarked = True

            if global_step % 50 == 0 or global_step in [1, 5, 10, 20]:
                mean_recent = np.mean(step_times[-20:]) if len(step_times) >= 20 else np.mean(step_times)
                print(f"[cg_gemma4_ar] Step {global_step:04d}/{total_steps} | Loss: {accum_loss:.4f} | Speed: {mean_recent:.3f} s/step")

            # Periodic checkpointing every 400 optimizer steps
            if global_step % 400 == 0:
                step_ckpt = os.path.join(ckpts_dir, f"step_{global_step}")
                peft_model.save_pretrained(step_ckpt)
                print(f"[cg_gemma4_ar] Saved intermediate checkpoint to {step_ckpt}")

            # Hard wall-clock cutoff check (2026-11-12 21:52:00 UTC / 15:52 CST = 1794520320)
            if time.time() >= 1794520320:
                print(f"\n[cg_gemma4_ar] !!! HARD WALL-CLOCK CUTOFF REACHED (21:52:00 UTC) !!!")
                print(f"[cg_gemma4_ar] Halting training loop cleanly at step {global_step} to begin 9-split evaluation.")
                cutoff_hit = True
                break

        # Save epoch checkpoint
        ckpt_path = os.path.join(ckpts_dir, f"epoch_{ep+1}")
        peft_model.save_pretrained(ckpt_path)
        print(f"[cg_gemma4_ar] Saved checkpoint for Epoch {ep+1} to {ckpt_path}")
        if cutoff_hit:
            break

    # Save final checkpoint and training metadata
    final_ckpt = os.path.join(ckpts_dir, "final")
    peft_model.save_pretrained(final_ckpt)
    training_meta = {
        "completed_steps": global_step,
        "total_planned_steps": total_steps,
        "completed_examples": global_step * micro_batch_size * accum_steps,
        "final_loss": float(accum_loss),
        "total_training_time_s": float(time.time() - t_train_start),
        "cutoff_hit": cutoff_hit,
    }
    with open(os.path.join(output_dir, "training_meta.json"), "w") as meta_f:
        json.dump(training_meta, meta_f, indent=2)
    print(f"[cg_gemma4_ar] Saved final PEFT checkpoint and training_meta.json to {output_dir}")
    print(f"\n[cg_gemma4_ar] Training Complete/Cutoff! Total Steps: {global_step}. Starting Evaluation...")
    return _evaluate_lora_splits(
        peft_model=peft_model,
        processor=processor,
        test_pack=test_pack,
        manifests=manifests,
        inst_preamble=inst_preamble,
        output_dir=output_dir,
        eval_batch_size=eval_batch_size,
        device=device,
        splits=splits,
    )


def _evaluate_lora_splits(
    peft_model,
    processor,
    test_pack: Dict[str, np.ndarray],
    manifests: Dict[str, Any],
    inst_preamble: str,
    output_dir: str,
    eval_batch_size: int = 12,
    device: str = "cuda:0",
    splits: Optional[List[str]] = None,
):
    """Evaluate a trained PEFT Gemma 4 AR model across requested Exp A splits."""
    os.makedirs(output_dir, exist_ok=True)
    preds_dir = os.path.join(output_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    eval_batch_size = min(eval_batch_size, 36)
    peft_model.eval()

    target_splits = splits if splits else [
        "seen_bal360",
        "unseen_para_bal360",
        "unseen_infer_360",
        "refeed_heldout_comp_bal360",
        "seen_shuf360",
        "heldout_reverse_order+delete_selected_360",
        "heldout_reverse_edges+keep_callees_360",
        "refeed_chain_heldout_ops_360",
        "open_unseen_360",
    ]

    eos_ids = list({int(processor.tokenizer.eos_token_id), 106}) if processor.tokenizer.eos_token_id is not None else [106]
    all_summaries: Dict[str, Any] = {}

    for split in target_splits:
        if split not in manifests:
            continue
        split_out_file = os.path.join(output_dir, f"{split}_summary.json")
        if os.path.exists(split_out_file):
            print(f"[cg_gemma4_ar] Split {split} already exists at {split_out_file}; loading and skipping.")
            with open(split_out_file) as f:
                all_summaries[split] = json.load(f)
            continue

        print(f"\n{'='*70}\n[cg_gemma4_ar] Evaluating Arm 2 (LoRA) on split: {split}\n{'='*70}")
        man = manifests[split]
        t0 = time.time()
        n_items = len(man["img_indices"])

        if split == "open_unseen_360":
            seeds = [42, 1042, 2042, 3042]
            gen_grids_list = []
            exact_rates = []
            for s_idx, s_val in enumerate(seeds):
                torch.manual_seed(s_val)
                pred_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
                exact_count = 0
                for b_start in range(0, n_items, eval_batch_size):
                    b_end = min(b_start + eval_batch_size, n_items)
                    b_imgs = [O.render_pil(test_pack["src_grid"][man["img_indices"][i]]) for i in range(b_start, b_end)]
                    b_prompts = []
                    for i in range(b_start, b_end):
                        inst = man["instructions"][i]
                        u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                        b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                    inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                    with torch.inference_mode():
                        outputs = peft_model.generate(
                            **inputs,
                            max_new_tokens=288,
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.95,
                            eos_token_id=eos_ids,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            use_cache=True,
                        )
                    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                    decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    for j, text in enumerate(decoded):
                        idx = b_start + j
                        fallback = test_pack["src_grid"][man["img_indices"][idx]]
                        g, exact = parse_16x16_grid(text, fallback)
                        pred_grids[idx] = g
                        if exact:
                            exact_count += 1

                gen_grids_list.append(pred_grids)
                exact_rates.append(exact_count / n_items)
                np.save(os.path.join(preds_dir, f"{split}_seed{s_val}.npy"), pred_grids)

            summary = CS.score_open(test_pack, man, gen_grids_list)
            summary["exact_256_parse_rate"] = float(np.mean(exact_rates))
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_lora_r64_s42"
            summary["manifest"] = split
            valid_rates = [v.get("valid", 0.0) for v in summary["per_generation"][0].values()] if summary.get("per_generation") else [0.0]
            print(f"[cg_gemma4_ar] {split}: Mean Valid Rate={np.mean(valid_rates):.4f} | Parse={summary['exact_256_parse_rate']:.4f} | Time={summary['elapsed_s']:.1f}s")

        elif "step1_instructions" in man:
            # 2-step visual refeeding
            s1_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
            for b_start in range(0, n_items, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_items)
                b_imgs = [O.render_pil(test_pack["src_grid"][man["img_indices"][i]]) for i in range(b_start, b_end)]
                b_prompts = []
                for i in range(b_start, b_end):
                    inst = man["step1_instructions"][i]
                    u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                    b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    outputs = peft_model.generate(
                        **inputs,
                        max_new_tokens=288,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for j, text in enumerate(decoded):
                    idx = b_start + j
                    fallback = test_pack["src_grid"][man["img_indices"][idx]]
                    g, _ = parse_16x16_grid(text, fallback)
                    s1_grids[idx] = g

            # Step 2
            pred_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
            exact_count = 0
            for b_start in range(0, n_items, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_items)
                b_imgs = [O.render_pil(s1_grids[i]) for i in range(b_start, b_end)]
                b_prompts = []
                for i in range(b_start, b_end):
                    inst = man["step2_instructions"][i]
                    u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                    b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    outputs = peft_model.generate(
                        **inputs,
                        max_new_tokens=288,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for j, text in enumerate(decoded):
                    idx = b_start + j
                    fallback = s1_grids[idx]
                    g, exact = parse_16x16_grid(text, fallback)
                    pred_grids[idx] = g
                    if exact:
                        exact_count += 1

            np.save(os.path.join(preds_dir, f"{split}.npy"), pred_grids)
            summary = CS.score_fixed(test_pack, man, pred_grids)
            summary["exact_256_parse_rate"] = exact_count / n_items
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_lora_r64_s42"
            summary["manifest"] = split
            print(f"[cg_gemma4_ar] {split}: K-way Acc={summary.get('acc_kway', 0.0):.4f} | Changed Acc={summary.get('changed_acc', 0.0):.4f} | Parse={summary['exact_256_parse_rate']:.4f} | Time={summary['elapsed_s']:.1f}s")

        elif "chain_instructions" in man:
            # 3-step chain refeeding
            curr_grids = np.array([test_pack["src_grid"][man["img_indices"][i]] for i in range(n_items)], dtype=np.uint8)
            chain_len = len(man["chain_instructions"][0])
            exact_count = 0
            for step_idx in range(chain_len):
                next_grids = np.zeros_like(curr_grids)
                for b_start in range(0, n_items, eval_batch_size):
                    b_end = min(b_start + eval_batch_size, n_items)
                    b_imgs = [O.render_pil(curr_grids[i]) for i in range(b_start, b_end)]
                    b_prompts = []
                    for i in range(b_start, b_end):
                        inst = man["chain_instructions"][i][step_idx]
                        u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                        msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                        b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                    inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                    with torch.inference_mode():
                        outputs = peft_model.generate(
                            **inputs,
                            max_new_tokens=288,
                            do_sample=False,
                            eos_token_id=eos_ids,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            use_cache=True,
                        )
                    gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                    decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                    for j, text in enumerate(decoded):
                        idx = b_start + j
                        fallback = curr_grids[idx]
                        g, exact = parse_16x16_grid(text, fallback)
                        next_grids[idx] = g
                        if step_idx == chain_len - 1 and exact:
                            exact_count += 1
                curr_grids = next_grids

            pred_grids = curr_grids
            np.save(os.path.join(preds_dir, f"{split}.npy"), pred_grids)
            summary = CS.score_fixed(test_pack, man, pred_grids)
            summary["exact_256_parse_rate"] = exact_count / n_items
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_lora_r64_s42"
            summary["manifest"] = split
            print(f"[cg_gemma4_ar] {split}: K-way Acc={summary.get('acc_kway', 0.0):.4f} | Changed Acc={summary.get('changed_acc', 0.0):.4f} | Parse={summary['exact_256_parse_rate']:.4f} | Time={summary['elapsed_s']:.1f}s")

        else:
            # Standard fixed manifests
            pred_grids = np.zeros((n_items, 16, 16), dtype=np.uint8)
            exact_count = 0
            for b_start in range(0, n_items, eval_batch_size):
                b_end = min(b_start + eval_batch_size, n_items)
                b_imgs = [O.render_pil(test_pack["src_grid"][man["img_indices"][i]]) for i in range(b_start, b_end)]
                b_prompts = []
                for i in range(b_start, b_end):
                    inst = man["instructions"][i]
                    u_text = f"{inst_preamble}\nInstruction: {inst}\nOutput the resulting 16x16 grid as 16 rows of 16 digits (0-7):\n"
                    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": u_text}]}]
                    b_prompts.append(processor.apply_chat_template(msg, add_generation_prompt=True, tokenize=False, enable_thinking=False))

                inputs = processor(text=b_prompts, images=[[im] for im in b_imgs], return_tensors="pt", padding=True).to(device)
                with torch.inference_mode():
                    outputs = peft_model.generate(
                        **inputs,
                        max_new_tokens=288,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        use_cache=True,
                    )
                gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
                decoded = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                for j, text in enumerate(decoded):
                    idx = b_start + j
                    fallback = test_pack["src_grid"][man["img_indices"][idx]]
                    g, exact = parse_16x16_grid(text, fallback)
                    pred_grids[idx] = g
                    if exact:
                        exact_count += 1

            np.save(os.path.join(preds_dir, f"{split}.npy"), pred_grids)
            summary = CS.score_fixed(test_pack, man, pred_grids)
            summary["exact_256_parse_rate"] = exact_count / n_items
            summary["elapsed_s"] = float(time.time() - t0)
            summary["arm"] = "gemma4_ar_lora_r64_s42"
            summary["manifest"] = split
            print(f"[cg_gemma4_ar] {split}: K-way Acc={summary.get('acc_kway', 0.0):.4f} | Changed Acc={summary.get('changed_acc', 0.0):.4f} | Parse={summary['exact_256_parse_rate']:.4f} | Time={summary['elapsed_s']:.1f}s")

        with open(split_out_file, "w") as f:
            json.dump(summary, f, indent=2)
        all_summaries[split] = summary

    master_file = os.path.join(output_dir, "manifest_summary.json")
    merged = {}
    if os.path.exists(master_file):
        try:
            with open(master_file) as f:
                merged = json.load(f)
        except Exception:
            merged = {}
    merged.update(all_summaries)
    with open(master_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[cg_gemma4_ar] Arm 2 LoRA Evaluation Complete ({len(all_summaries)} splits). Summary saved to {master_file}")
    return merged


def run_lora_eval_only_arm(
    model_id: str,
    adapter_dir: str,
    train_pack_path: str,
    test_pack_path: str,
    output_dir: str,
    eval_batch_size: int = 12,
    device: str = "cuda:0",
    token: Optional[str] = None,
    splits: Optional[List[str]] = None,
):
    """Load a saved PEFT Gemma 4 AR adapter and evaluate requested splits."""
    from transformers import Gemma4ForConditionalGeneration
    from peft import PeftModel

    train_pack = dict(np.load(train_pack_path))
    test_pack = dict(np.load(test_pack_path))
    manifests = SP.build_cg_eval_manifests(test_pack)
    inst_preamble = SP.preamble(train_pack)

    processor = load_gemma4_processor(model_id, token=token)
    base_model = Gemma4ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=token,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    peft_model.eval()
    return _evaluate_lora_splits(
        peft_model=peft_model,
        processor=processor,
        test_pack=test_pack,
        manifests=manifests,
        inst_preamble=inst_preamble,
        output_dir=output_dir,
        eval_batch_size=eval_batch_size,
        device=device,
        splits=splits,
    )


# -----------------------------------------------------------------------------
# Main CLI Entrypoint
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Gemma 4 AR Baseline Runner")
    parser.add_argument("--mode", type=str, required=True, choices=["prompted", "lora_train_and_eval", "lora_eval_only"],
                        help="Execution mode: 'prompted' (Arm 1) or 'lora_train_and_eval' (Arm 2)")
    parser.add_argument("--model-id", type=str, default="google/gemma-4-26B-A4B-it",
                        help="HuggingFace model ID")
    parser.add_argument("--train-pack", type=str, default="data/gridA_train1608.npz",
                        help="Path to training data pack")
    parser.add_argument("--test-pack", type=str, default="data/gridA_test360_httpx.npz",
                        help="Path to test data pack")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to outputs/m2/arms/gemma4_ar_<mode>/)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of LoRA training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="AdamW learning rate")
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha")
    parser.add_argument("--micro-batch-size", type=int, default=2,
                        help="Micro batch size for training")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Compute device")
    parser.add_argument("--adapter-dir", type=str, default=None,
                        help="Path to saved PEFT adapter dir (required for --mode lora_eval_only)")
    parser.add_argument("--splits", type=str, default=None,
                        help="Optional comma-separated list of splits to evaluate (for multi-GPU sharding)")

    args = parser.parse_args()
    split_list = [s.strip() for s in args.splits.split(",") if s.strip()] if args.splits else None

    token = find_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token

    if args.output_dir is None:
        if args.mode == "prompted":
            args.output_dir = "outputs/m2/arms/gemma4_ar_prompted"
        else:
            args.output_dir = "outputs/m2/arms/gemma4_ar_lora_r64_s42"

    print(f"[cg_gemma4_ar] Mode: {args.mode}")
    print(f"[cg_gemma4_ar] Model: {args.model_id}")
    print(f"[cg_gemma4_ar] Output Dir: {args.output_dir}")
    print(f"[cg_gemma4_ar] Device: {args.device}")
    if split_list:
        print(f"[cg_gemma4_ar] Sharded Splits: {split_list}")

    if args.mode == "prompted":
        run_prompted_arm(
            model_id=args.model_id,
            train_pack_path=args.train_pack,
            test_pack_path=args.test_pack,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            device=args.device,
            token=token,
            splits=split_list,
        )
    elif args.mode == "lora_eval_only":
        if not args.adapter_dir:
            raise ValueError("--adapter-dir is required when --mode lora_eval_only")
        run_lora_eval_only_arm(
            model_id=args.model_id,
            adapter_dir=args.adapter_dir,
            train_pack_path=args.train_pack,
            test_pack_path=args.test_pack,
            output_dir=args.output_dir,
            eval_batch_size=args.batch_size,
            device=args.device,
            token=token,
            splits=split_list,
        )
    elif args.mode == "lora_train_and_eval":
        run_lora_train_and_eval_arm(
            model_id=args.model_id,
            train_pack_path=args.train_pack,
            test_pack_path=args.test_pack,
            output_dir=args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            micro_batch_size=args.micro_batch_size,
            accum_steps=args.accum_steps,
            eval_batch_size=args.batch_size,
            device=args.device,
            seed=args.seed,
            token=token,
        )


if __name__ == "__main__":
    main()
