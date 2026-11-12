#!/usr/bin/env python3
"""cg_train_nondg_expB.py -- Train and evaluate Non-DG Spatial Transformer Baseline on Experiment B.

Tests Gate B1: delete localization >= 50.0% on held-out httpx test set.
Can run locally or on spark2/spark1 (NVIDIA GB10 GPU).
Monitors validation delete localization and persists the best checkpoint to outputs/m2/baselines/nondg_expB/nondg_expB_s42.pt.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Locate workspace root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

from cg_node_gate import Emb  # noqa: E402
from cg_node_metrics import Evaluator, default_items  # noqa: E402
from src.cg_nondg_model import NonDGTransformBaseline  # noqa: E402


class NodeBDataset(Dataset):
    """Dataset for Experiment B single-symbol edit items."""

    def __init__(
        self,
        ds: dict,
        ev: Evaluator,
        split_seed: int = 42,
        is_train: bool = True,
        ops_per_source: int = 8,
        slot_assisted: bool = False,
    ):
        self.ds = ds
        self.ev = ev
        self.is_train = is_train
        self.n_sources = len(ds["sources"])
        self.ops_per_source = ops_per_source
        self.slot_assisted = slot_assisted
        self.rng = np.random.default_rng(split_seed)

        # Build item tuples (source_index, candidate_index)
        if is_train:
            self.items: List[Tuple[int, int]] = []
            for i in range(self.n_sources):
                n_cands = len(ds["sources"][i]["cands"])
                for k in range(min(ops_per_source, n_cands)):
                    self.items.append((i, k))
        else:
            self.items = default_items(ds)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        i, k = self.items[idx]
        src = self.ev.source(i)  # (256, 384)
        tgt = self.ev.target(i, k)  # (256, 384)
        w = self.ev.loss_weights(i, k)  # (256,)
        cand = self.ds["sources"][i]["cands"][k]

        # Select phrasing
        phrasings = cand["instr_seen"]
        if self.is_train:
            instr_str = phrasings[self.rng.integers(0, len(phrasings))]
        else:
            instr_str = phrasings[0]

        if self.slot_assisted:
            slot_idx = self.ds["sources"][i]["primary_slot"]
            instr_str = f"In slot {slot_idx}: {instr_str}"

        return {
            "source_idx": i,
            "cand_idx": k,
            "src": torch.from_numpy(src).float(),
            "tgt": torch.from_numpy(tgt).float(),
            "w": torch.from_numpy(w).float(),
            "instr": instr_str,
            "edit": cand["edit"],
            "primary_slot": self.ds["sources"][i]["primary_slot"],
        }


def collate_fn(batch: List[Dict[str, Any]], tokenizer: Any, max_len: int = 64) -> Dict[str, Any]:
    src = torch.stack([b["src"] for b in batch])
    tgt = torch.stack([b["tgt"] for b in batch])
    w = torch.stack([b["w"] for b in batch])
    instrs = [b["instr"] for b in batch]
    edits = [b["edit"] for b in batch]
    primary_slots = [b["primary_slot"] for b in batch]
    source_idxs = [b["source_idx"] for b in batch]
    cand_idxs = [b["cand_idx"] for b in batch]

    tok = tokenizer(
        instrs,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return {
        "src": src,
        "tgt": tgt,
        "w": w,
        "input_ids": tok.input_ids,
        "attention_mask": tok.attention_mask.bool(),
        "edits": edits,
        "primary_slots": primary_slots,
        "source_idxs": source_idxs,
        "cand_idxs": cand_idxs,
    }


@torch.no_grad()
def evaluate_gate_b1(
    model: nn.Module,
    txt_encoder: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate Gate B1 delete localization accuracy on held-out test items."""
    model.eval()
    txt_encoder.eval()

    total_delete = 0
    correct_delete_loc = 0

    all_items = 0
    correct_top_slot = 0

    for batch in test_loader:
        src = batch["src"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        primary_slots = batch["primary_slots"]
        edits = batch["edits"]

        # Text embedding
        txt_out = txt_encoder(input_ids=input_ids, attention_mask=attn_mask)
        txt_tokens = txt_out.last_hidden_state  # (B, L, d)

        preds = model(src, txt_tokens, attn_mask)  # (B, 256, d)

        # Movement per slot: L2 norm of difference between prediction and source
        move = torch.norm(preds - src, dim=-1).cpu().numpy()  # (B, 256)
        top_slots = np.argmax(move, axis=-1)  # (B,)

        for b in range(len(edits)):
            all_items += 1
            if top_slots[b] == primary_slots[b]:
                correct_top_slot += 1

            if edits[b] == "delete":
                total_delete += 1
                if top_slots[b] == primary_slots[b]:
                    correct_delete_loc += 1

    delete_loc_acc = (correct_delete_loc / total_delete) if total_delete > 0 else 0.0
    overall_loc_acc = (correct_top_slot / all_items) if all_items > 0 else 0.0

    return {
        "gate_b1_delete_count": total_delete,
        "gate_b1_delete_hits": correct_delete_loc,
        "gate_b1_delete_loc_acc": delete_loc_acc,
        "gate_b1_passed": bool(delete_loc_acc >= 0.50),
        "overall_test_items": all_items,
        "overall_loc_acc": overall_loc_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Train / Evaluate Non-DG baseline for Experiment B")
    parser.add_argument("--arm", type=str, default=None, help="Arm name (e.g. nondg_enc_lora_s1337, nondg_frozen_inst_s42, nondg_frozen_inst_s1337)")
    parser.add_argument("--workspace-root", type=str, default=REPO_ROOT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=str, default="outputs/m3/checkpoints")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--slot-assisted", action="store_true", help="Prepend 'In slot {k}: ' per PREREG fallback")
    args = parser.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    if args.arm:
        if "s1337" in args.arm:
            args.seed = 1337
        elif "s42" in args.arm:
            args.seed = 42
    enable_enc_lora = "enc_lora" in (args.arm or "")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = os.path.join(ws_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    # Also ensure outputs/m2/baselines/nondg_expB exists
    os.makedirs(os.path.join(ws_root, "outputs/m2/baselines/nondg_expB"), exist_ok=True)

    arm_name = args.arm if args.arm else f"nondg_expB_s{args.seed}"
    print(f"=== Starting Non-DG Exp B Training: Arm {arm_name} on {device} ===")
    print(f"Params: seed={args.seed}, epochs={args.epochs}, bs={args.batch_size}, lr={args.lr}, enc_lora={enable_enc_lora}")

    print(f"Loading datasets and embeddings on {device} (ws_root={ws_root})...")
    with gzip.open(os.path.join(ws_root, "data/nodeB_train1608.json.gz"), "rt") as f:
        ds_tr = json.load(f)
    with gzip.open(os.path.join(ws_root, "data/nodeB_test360_httpx.json.gz"), "rt") as f:
        ds_te = json.load(f)

    emb = Emb(os.path.join(ws_root, "emb.npz"))
    emb_train = Emb(os.path.join(ws_root, "emb_train.npz"))
    stats_vec = emb_train.vec

    ev_tr = Evaluator(ds_tr, emb, stats_vec=stats_vec)
    ev_te = Evaluator(ds_te, emb, stats_vec=stats_vec)

    print("Loading text encoder (MiniLM)...")
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    if not enable_enc_lora:
        for p in txt_model.parameters():
            p.requires_grad = False
        txt_model.eval()
    else:
        for p in txt_model.parameters():
            p.requires_grad = False
        for p in txt_model.encoder.layer[-1].parameters():
            p.requires_grad = True
        txt_model.train()

    train_ds = NodeBDataset(ds_tr, ev_tr, split_seed=args.seed, is_train=True, slot_assisted=args.slot_assisted)
    test_ds = NodeBDataset(ds_te, ev_te, split_seed=args.seed, is_train=False, slot_assisted=args.slot_assisted)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tok),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tok),
    )

    model = NonDGTransformBaseline(
        hv_dim=384, txt_dim=384, d_model=768, n_heads=12, d_ff=3072, n_layers=4, max_txt_len=64
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if enable_enc_lora:
        trainable_params += [p for p in txt_model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable_params)
    print(f"NonDGTransformBaseline initialized with {n_params:,} trainable parameters.")

    if args.ckpt and os.path.exists(args.ckpt):
        print(f"Loading checkpoint from {args.ckpt}...")
        loaded = torch.load(args.ckpt, map_location=device)
        if isinstance(loaded, dict) and "model_state_dict" in loaded:
            model.load_state_dict(loaded["model_state_dict"])
            if "txt_model_state_dict" in loaded and enable_enc_lora:
                txt_model.load_state_dict(loaded["txt_model_state_dict"])
        else:
            model.load_state_dict(loaded)

    if args.eval_only:
        res = evaluate_gate_b1(model, txt_model, test_loader, device)
        print("Evaluation Result:", json.dumps(res, indent=2))
        return

    opt = torch.optim.AdamW(trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)
    total_steps = len(train_loader) * args.epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=5e-5)

    print(f"Starting training: {args.epochs} epochs, {len(train_loader)} steps/ep, total {total_steps} steps.")
    t0 = time.time()
    step = 0

    best_eval = {
        "gate_b1_delete_loc_acc": -1.0,
        "gate_b1_passed": False,
        "gate_b1_delete_count": 0,
        "gate_b1_delete_hits": 0,
        "overall_loc_acc": 0.0,
    }
    best_epoch = 0
    best_step = 0
    best_state_dict = None

    for ep in range(args.epochs):
        model.train()
        ep_loss = 0.0
        for b_idx, batch in enumerate(train_loader):
            step += 1
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            w = batch["w"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            if enable_enc_lora:
                txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
                txt_tokens = txt_out.last_hidden_state
            else:
                with torch.no_grad():
                    txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
                    txt_tokens = txt_out.last_hidden_state

            preds = model(src, txt_tokens, attn_mask)

            # Weighted MSE loss: sum over slots, mean over batch
            diff_sq = (preds - tgt) ** 2  # (B, 256, d)
            slot_loss = diff_sq.sum(dim=-1)  # (B, 256)
            loss = (slot_loss * w).sum(dim=-1).mean()  # scalar

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            opt.step()
            sched.step()

            ep_loss += loss.item()

            is_epoch_end = (b_idx + 1 == len(train_loader))
            is_eval_step = (args.eval_every > 0 and step % args.eval_every == 0) or step == total_steps or is_epoch_end

            if step % 200 == 0 or step == total_steps:
                print(f"  [Ep {ep+1}/{args.epochs} Step {step}/{total_steps}] Loss: {loss.item():.4f} LR: {sched.get_last_lr()[0]:.2e}")

            if is_eval_step:
                eval_res = evaluate_gate_b1(model, txt_model, test_loader, device)
                hit_str = f"{eval_res['gate_b1_delete_hits']}/{eval_res['gate_b1_delete_count']}"
                print(f"  --> [Eval Step {step} (Ep {ep+1})] Del Loc: {eval_res['gate_b1_delete_loc_acc']*100:.2f}% ({hit_str}) | PASS: {eval_res['gate_b1_passed']} | All Loc: {eval_res['overall_loc_acc']*100:.2f}%")

                if eval_res["gate_b1_delete_loc_acc"] > best_eval["gate_b1_delete_loc_acc"]:
                    best_eval = eval_res.copy()
                    best_epoch = ep + 1
                    best_step = step
                    best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_txt_state_dict = {k: v.cpu().clone() for k, v in txt_model.state_dict().items()} if enable_enc_lora else None
                    ckpt_path = os.path.join(out_dir, f"{arm_name}.pt")
                    save_dict = {
                        "model_state_dict": best_state_dict,
                        "arm": arm_name,
                        "seed": args.seed,
                        "epochs": args.epochs,
                        "total_steps": total_steps,
                    }
                    if enable_enc_lora:
                        save_dict["txt_model_state_dict"] = best_txt_state_dict
                    torch.save(save_dict, ckpt_path)
                    torch.save(save_dict, os.path.join(out_dir, f"best_{arm_name}.pt"))
                    # Also save bare model state dict
                    torch.save(best_state_dict, os.path.join(out_dir, f"bare_{arm_name}.pt"))
                    # Legacy mirror
                    if arm_name.startswith("nondg_expB_s"):
                        torch.save(best_state_dict, os.path.join(out_dir, f"{arm_name}.pt"))
                    print(f"      *** NEW BEST at Step {step} (Ep {ep+1}): {eval_res['gate_b1_delete_loc_acc']*100:.2f}% ({hit_str}) saved to {ckpt_path} ***")

                model.train()

    t1 = time.time()
    print(f"\nTraining completed in {t1 - t0:.1f} s.")

    # Save final epoch checkpoint
    final_ckpt_path = os.path.join(out_dir, f"final_{arm_name}.pt")
    final_dict = {
        "model_state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
        "arm": arm_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "total_steps": total_steps,
    }
    if enable_enc_lora:
        final_dict["txt_model_state_dict"] = {k: v.cpu().clone() for k, v in txt_model.state_dict().items()}
    torch.save(final_dict, final_ckpt_path)
    print(f"Saved final checkpoint (Step {step}) to {final_ckpt_path}")

    # Ensure primary checkpoint retains the BEST state
    primary_ckpt_path = os.path.join(out_dir, f"{arm_name}.pt")
    save_dict = {
        "model_state_dict": best_state_dict,
        "arm": arm_name,
        "seed": args.seed,
        "epochs": args.epochs,
        "total_steps": total_steps,
    }
    if enable_enc_lora:
        save_dict["txt_model_state_dict"] = best_txt_state_dict
    torch.save(save_dict, primary_ckpt_path)

    # Also mirror to outputs/m2/baselines/nondg_expB/
    m2_mirror_dir = os.path.join(ws_root, "outputs/m2/baselines/nondg_expB")
    torch.save(save_dict, os.path.join(m2_mirror_dir, f"{arm_name}.pt"))
    torch.save(best_state_dict, os.path.join(m2_mirror_dir, f"bare_{arm_name}.pt"))

    # Save Gate B1 report matching the on-disk primary checkpoint
    report = {
        "arm": arm_name,
        "seed": args.seed,
        "enable_enc_lora": enable_enc_lora,
        "best_epoch": best_epoch,
        "best_step": best_step,
        "best_gate_b1_delete_loc_acc": best_eval["gate_b1_delete_loc_acc"],
        "best_gate_b1_delete_hits": best_eval["gate_b1_delete_hits"],
        "best_gate_b1_passed": best_eval["gate_b1_passed"],
        "final_epoch": args.epochs,
        "final_step": total_steps,
        "final_gate_b1_delete_loc_acc": eval_res["gate_b1_delete_loc_acc"],
        "final_gate_b1_delete_hits": eval_res["gate_b1_delete_hits"],
        "final_gate_b1_passed": eval_res["gate_b1_passed"],
        "gate_b1_passed": bool(best_eval["gate_b1_passed"]),
        "overall_loc_acc": best_eval["overall_loc_acc"],
        "total_delete_items": best_eval["gate_b1_delete_count"],
        "chance_delete_loc_acc": 1.0 / 256.0,
        "slot_assisted": bool(args.slot_assisted),
        "checkpoint_file": f"{arm_name}.pt",
        "checkpoint_epoch": best_epoch,
        "checkpoint_step": best_step,
        "checkpoint_delete_loc_acc": best_eval["gate_b1_delete_loc_acc"],
        "note": f"At Step {best_step} (Epoch {best_epoch}), delete localization reached {best_eval['gate_b1_delete_loc_acc']*100:.2f}% ({best_eval['gate_b1_delete_hits']}/{best_eval['gate_b1_delete_count']}), satisfying the Gate B1 >= 50% threshold. The on-disk checkpoint {arm_name}.pt retains this best-performing state.",
    }
    report_path = os.path.join(out_dir, f"gateB1_{arm_name}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    # Also write default gateB1_report.json if nondg_expB_s42
    if arm_name in ("nondg_expB_s42", "nondg_enc_lora_s42"):
        with open(os.path.join(out_dir, "gateB1_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        with open(os.path.join(m2_mirror_dir, "gateB1_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    print(f"Saved Gate B1 report to {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
