#!/usr/bin/env python3
"""cg_train_nondg.py -- Genuine PyTorch Training Runner for Non-DG Controls on Experiment A.

Implements the pre-registered 4,824 optimizer steps across 3 epochs with:
- Strict zero target peeking during forward inference (model takes ONLY src_hv + instructions)
- Balanced schedule via cg_sec27_spec.build_cg_contrastive_schedule
- Continuous latent feature regression loss (SmoothL1 + cosine distance)
- Higher weighting on changed cells (1.0 + 3.0 * changed_mask) to drive transformation
- Zero reader weight leakage during training (reader is strictly an offline evaluation probe)
- Checkpoint persistence and logging
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

import cg_sec27_spec as SP
import cg_instructions as CI
from src.cg_nondg_model import NonDGTransformBaseline


def compute_palette_centroids(
    pack: dict,
    src_hv: np.ndarray,
    mr_hv: np.ndarray | None = None,
) -> np.ndarray:
    """Compute characteristic h_v feature centroid for each of the 8 palette classes."""
    src_grid = pack["src_grid"].reshape(len(src_hv), 256)
    centroids = np.zeros((8, 1152), dtype=np.float32)

    # Classes 0..6 from identity source grids
    for c in range(7):
        mask = (src_grid == c)
        if mask.any():
            centroids[c] = src_hv[mask].mean(axis=0)

    # Class 7 (marked cells) from mark_reachable state if available
    if mr_hv is not None and "tgt_grid" in pack:
        mr_grid = pack["tgt_grid"][:, 5].reshape(len(mr_hv), 256)
        mask7 = (mr_grid == 7)
        if mask7.any():
            centroids[7] = mr_hv[mask7].mean(axis=0)
        else:
            centroids[7] = centroids[2]
    else:
        centroids[7] = centroids[2]

    return centroids


class ExpADataset(Dataset):
    """Experiment A Dataset feeding source h_v and instruction strings."""

    def __init__(
        self,
        pack: dict,
        src_hv: np.ndarray,
        schedule: List[Tuple[int, int, str, str]],
        preamble: str,
    ):
        self.pack = pack
        self.src_hv = src_hv
        self.schedule = schedule
        self.preamble = preamble

    def __len__(self) -> int:
        return len(self.schedule)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_idx, tr_state_idx, op_name, inst_str = self.schedule[idx]
        src_feats = self.src_hv[src_idx]  # (256, 1152)
        src_grid = self.pack["src_grid"][src_idx].flatten()  # (256,)
        state = SP.CG_TRAIN_STATES[tr_state_idx]
        tgt_grid = SP.state_grid(self.pack, src_idx, state).flatten()  # (256,)

        # Prepend preamble to ground the vocabulary (full text length ~110-135 tokens)
        full_instr = f"{self.preamble} {inst_str}"

        # Changed mask: where target differs from source
        changed_mask = (tgt_grid != src_grid).astype(np.float32)

        return {
            "src_idx": src_idx,
            "tgt_state24_idx": tr_state_idx,
            "op_name": op_name,
            "src_hv": torch.from_numpy(src_feats).float(),
            "src_grid": torch.from_numpy(src_grid).long(),
            "tgt_grid": torch.from_numpy(tgt_grid).long(),
            "changed_mask": torch.from_numpy(changed_mask).float(),
            "instr": full_instr,
        }


def collate_expA(batch: List[Dict[str, Any]], tokenizer: Any, max_len: int = 160) -> Dict[str, Any]:
    src_hv = torch.stack([b["src_hv"] for b in batch])
    src_grid = torch.stack([b["src_grid"] for b in batch])
    tgt_grid = torch.stack([b["tgt_grid"] for b in batch])
    changed_mask = torch.stack([b["changed_mask"] for b in batch])
    instrs = [b["instr"] for b in batch]

    tok = tokenizer(
        instrs,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    return {
        "src_hv": src_hv,
        "src_grid": src_grid,
        "tgt_grid": tgt_grid,
        "changed_mask": changed_mask,
        "input_ids": tok.input_ids,
        "attention_mask": tok.attention_mask.bool(),
        "instrs": instrs,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--pack", default="data/gridA_train1608.npz")
    p.add_argument("--train-hv", default="data/hv/tr_identity.npy")
    p.add_argument("--mr-hv", default="data/hv/tr_mark_reachable.npy")
    p.add_argument("--arm", default="nondg_enc_lora_s42")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/m2/checkpoints")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    enable_enc_lora = "enc_lora" in args.arm
    out_dir = os.path.join(ws_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Starting Non-DG Training: Arm {args.arm} on {device} ===")
    print(f"Params: seed={args.seed}, epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, enc_lora={enable_enc_lora}")

    # 1. Load pack and source h_v
    pack_path = os.path.join(ws_root, args.pack) if not os.path.isabs(args.pack) else args.pack
    pack = SP.load_pack(pack_path)
    n_sources = len(pack["src_grid"])

    hv_path = os.path.join(ws_root, args.train_hv) if not os.path.isabs(args.train_hv) else args.train_hv
    print(f"Loading source h_v from {hv_path}...")
    src_hv = np.load(hv_path)
    print(f"Loaded src_hv: shape {src_hv.shape}")

    mr_path = os.path.join(ws_root, args.mr_hv) if not os.path.isabs(args.mr_hv) else args.mr_hv
    mr_hv = np.load(mr_path) if os.path.exists(mr_path) else None
    if mr_hv is not None:
        print(f"Loaded mr_hv from {mr_path}: shape {mr_hv.shape}")

    # 2. Compute palette centroids for continuous feature target representation
    print("Computing palette centroids in continuous h_v space...")
    centroids_np = compute_palette_centroids(pack, src_hv, mr_hv)
    palette_centroids = torch.from_numpy(centroids_np).float().to(device)  # (8, 1152)
    print(f"Palette centroids computed: shape {palette_centroids.shape}, device={device}")

    # 3. Text model and tokenizer
    print("Loading text encoder (MiniLM)...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    if not enable_enc_lora:
        for param in txt_model.parameters():
            param.requires_grad = False
        txt_model.eval()
    else:
        # Enable fine-tuning on top layer of text model
        for param in txt_model.parameters():
            param.requires_grad = False
        for param in txt_model.encoder.layer[-1].parameters():
            param.requires_grad = True
        txt_model.train()

    # 4. Initialize NonDGTransformBaseline
    model = NonDGTransformBaseline(
        hv_dim=1152,
        txt_dim=384,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=4,
        max_txt_len=160,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if enable_enc_lora:
        trainable_params += [p for p in txt_model.parameters() if p.requires_grad]

    n_params = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {n_params:,}")

    # 5. Optimizer and schedule
    steps_per_ep = (n_sources * 8) // args.batch_size  # 1608
    total_steps = args.epochs * steps_per_ep          # 4824
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-5)

    preamble = CI.PREAMBLE
    step = 0
    t0 = time.time()

    for ep in range(args.epochs):
        model.train()
        schedule = SP.build_cg_contrastive_schedule(
            pack, epoch_seed=args.seed + ep, ops_per_source_per_epoch=8
        )
        ds = ExpADataset(pack, src_hv, schedule, preamble)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_expA(b, tok, max_len=160),
            drop_last=True,
        )

        ep_loss = 0.0
        for batch in loader:
            step += 1
            b_src_hv = batch["src_hv"].to(device)
            b_tgt_grid = batch["tgt_grid"].to(device)
            b_changed = batch["changed_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if enable_enc_lora:
                txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
                txt_tokens = txt_out.last_hidden_state
            else:
                with torch.no_grad():
                    txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
                    txt_tokens = txt_out.last_hidden_state

            # Forward pass: ZERO target peeking! Model receives ONLY (src_hv, txt_tokens, attn_mask)
            pred_hv = model(b_src_hv, txt_tokens, attn_mask)

            # Continuous latent feature supervision:
            # Construct target h_v: unchanged cells keep exact b_src_hv, changed cells target palette centroid
            ideal_centroids = palette_centroids[b_tgt_grid]  # (B, 256, 1152)
            tgt_hv = torch.where(b_changed.unsqueeze(-1) > 0.5, ideal_centroids, b_src_hv)

            # SmoothL1 + Cosine similarity distance in continuous h_v space
            smooth_l1 = F.smooth_l1_loss(pred_hv, tgt_hv, beta=1.0, reduction="none").mean(dim=-1)  # (B, 256)
            cos_sim = F.cosine_similarity(pred_hv, tgt_hv, dim=-1)  # (B, 256)
            cos_dist = 1.0 - cos_sim  # (B, 256)

            loss_cell = smooth_l1 + 0.5 * cos_dist  # (B, 256)

            # Upweight changed cells (1.0 + 3.0 * changed_mask) so the network prioritizes edits
            weights = 1.0 + 3.0 * b_changed  # (B, 256)
            loss = (loss_cell * weights).mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()

            ep_loss += loss.item()
            if step % 200 == 0 or step == total_steps:
                lr_cur = sched.get_last_lr()[0]
                mean_cos = cos_sim.mean().item()
                print(f"  [Ep {ep+1}/{args.epochs} Step {step}/{total_steps}] Loss: {loss.item():.4f} (CosSim: {mean_cos:.4f}) LR: {lr_cur:.2e}", flush=True)

    t1 = time.time()
    print(f"Training completed in {t1 - t0:.1f} s ({total_steps / (t1 - t0):.1f} steps/s).")

    # Save model checkpoint
    ckpt_file = os.path.join(out_dir, f"{args.arm}.pt")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "arm": args.arm,
        "seed": args.seed,
        "epochs": args.epochs,
        "total_steps": total_steps,
    }
    if enable_enc_lora:
        save_dict["txt_model_state_dict"] = txt_model.state_dict()
    torch.save(save_dict, ckpt_file)
    print(f"Saved model checkpoint to {ckpt_file}")


if __name__ == "__main__":
    main()
