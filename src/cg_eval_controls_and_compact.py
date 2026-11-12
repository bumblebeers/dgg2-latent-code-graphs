#!/usr/bin/env python3
"""cg_eval_controls_and_compact.py -- Parameter-Free Controls & Compact Pred Grids Generator.

1. Evaluates parameter-free controls across all 16 manifests:
   - source_copy floor (copying src_grid)
   - reader_ceiling (decoding true target h_v from cand_test360_httpx_22.npy via reader.npz)
2. Decodes all float32 pred_hv dumps in outputs/m2/preds/ into compact uint8 pred_grids:
   - pred_grids_{arm}_{manifest}.npy (shape: (360, 16, 16), uint8, 92 KB)
   - item_hit_{arm}_{manifest}.npy (shape: (360,), uint8, 1 for hit, 0 for miss)
   Enables paired McNemar tests against DG.
3. Updates outputs/m2/nondg_baseline_manifest_summary.json with source_copy and reader_ceiling.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

import cg_sec27_spec as SP
import cg_score_preds as CS
from cg_hv_probe import load_model, read_grids
from cg_metrics import nearest_candidate_grid, open_outcome


def eval_source_copy(pack: dict, manifests: Dict[str, Any], preds_dir: str) -> Dict[str, Any]:
    """Evaluate source_copy baseline across all 16 manifests."""
    results: Dict[str, Any] = {}
    print("\n=== Evaluating source_copy Baseline ===")

    for man_name, man in manifests.items():
        src_idxs = man["img_indices"]
        src_grids = pack["src_grid"][src_idxs].astype(np.uint8)  # (360, 16, 16)

        # Save compact pred_grid
        pg_file = os.path.join(preds_dir, f"pred_grids_source_copy_{man_name}.npy")
        np.save(pg_file, src_grids)

        if man_name.startswith("open_"):
            score = CS.score_open(pack, man, [src_grids] * 4)
            outs = [open_outcome(g, pack["src_grid"][src_idxs], pack["open_valid"][src_idxs, man["open_idx"]],
                                 pack["open_count"][src_idxs, man["open_idx"]], pack["open_all"][src_idxs, man["open_idx"]])
                    for g in [src_grids] * 4]
            hit = (outs[0]["label"] == 0).astype(np.uint8)
            mean_valid = np.mean([v['valid'] for v in score['per_generation'][0].values()])
            print(f"  [source_copy Eval {man_name}] open(valid={mean_valid:.3f})")
        else:
            score = CS.score_fixed(pack, man, src_grids)
            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(src_grids, pack["tgt_grid"][src_idxs][:, :K])
            k = np.array(man["tgt_state24_indices"])
            hit = ((idx == k) & ~tie).astype(np.uint8)
            print(f"  [source_copy Eval {man_name}] acc_kway={score['acc_kway']*100:.2f}%, changed_acc={score['changed_acc']*100:.2f}%")

        hit_file = os.path.join(preds_dir, f"item_hit_source_copy_{man_name}.npy")
        np.save(hit_file, hit)
        results[man_name] = score

    return results


def eval_reader_ceiling(pack: dict, manifests: Dict[str, Any], cand_hv_path: str, reader: Any, preds_dir: str) -> Dict[str, Any]:
    """Evaluate reader_ceiling baseline on true target h_v decoded via reader."""
    results: Dict[str, Any] = {}
    print(f"\n=== Evaluating reader_ceiling Baseline (loading cand_hv from {cand_hv_path}) ===")

    if not os.path.exists(cand_hv_path):
        print(f"Warning: candidate h_v not found at {cand_hv_path}, skipping reader_ceiling.")
        return {}

    cand_hv = np.load(cand_hv_path)  # (360, 22, 256, 1152) float32
    print(f"Loaded cand_hv tensor: {cand_hv.shape}")

    # Decode all candidates offline into discrete grids: (360, 22, 16, 16)
    n_items, n_cands = cand_hv.shape[0], cand_hv.shape[1]
    cand_hv_flat = cand_hv.reshape(-1, 256, 1152)
    decoded_cands = read_grids(reader, cand_hv_flat).reshape(n_items, n_cands, 16, 16).astype(np.uint8)
    print(f"Decoded true candidate grids: shape {decoded_cands.shape}")

    for man_name, man in manifests.items():
        src_idxs = man["img_indices"]

        if man_name.startswith("open_"):
            # For open-ended, use first valid answer grid
            k_open = man["open_idx"]
            val_grids = pack["open_valid"][src_idxs, k_open, 0].astype(np.uint8)
            score = CS.score_open(pack, man, [val_grids] * 4)
            pg_file = os.path.join(preds_dir, f"pred_grids_reader_ceiling_{man_name}.npy")
            np.save(pg_file, val_grids)
            outs = [open_outcome(g, pack["src_grid"][src_idxs], pack["open_valid"][src_idxs, k_open],
                                 pack["open_count"][src_idxs, k_open], pack["open_all"][src_idxs, k_open])
                    for g in [val_grids] * 4]
            hit = (outs[0]["label"] == 0).astype(np.uint8)
            hit_file = os.path.join(preds_dir, f"item_hit_reader_ceiling_{man_name}.npy")
            np.save(hit_file, hit)
            mean_valid = np.mean([v['valid'] for v in score['per_generation'][0].values()])
            print(f"  [reader_ceiling Eval {man_name}] open(valid={mean_valid:.3f})")
        else:
            tgt_k = np.array(man["tgt_state24_indices"])
            # True target h_v decoded via reader:
            pred_grids = decoded_cands[src_idxs, tgt_k]  # (360, 16, 16)
            pg_file = os.path.join(preds_dir, f"pred_grids_reader_ceiling_{man_name}.npy")
            np.save(pg_file, pred_grids)

            score = CS.score_fixed(pack, man, pred_grids)
            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(pred_grids, pack["tgt_grid"][src_idxs][:, :K])
            hit = ((idx == tgt_k) & ~tie).astype(np.uint8)

            hit_file = os.path.join(preds_dir, f"item_hit_reader_ceiling_{man_name}.npy")
            np.save(hit_file, hit)
            print(f"  [reader_ceiling Eval {man_name}] acc_kway={score['acc_kway']*100:.2f}%, cell_acc={score['cell_acc']*100:.2f}%, changed_acc={score['changed_acc']*100:.2f}%, edge_f1={score['edge_f1']:.4f}")

        results[man_name] = score

    return results


def compact_existing_preds(pack: dict, manifests: Dict[str, Any], reader: Any, preds_dir: str):
    """Scan preds_dir for float32 .npy pred dumps and convert to compact uint8 pred_grids & item_hit."""
    print(f"\n=== Compacting existing prediction dumps in {preds_dir} ===")
    K = len(SP.CG_CANDIDATE_STATES)

    for man_name, man in manifests.items():
        src_idxs = man["img_indices"]
        tgt_k = np.array(man["tgt_state24_indices"]) if "tgt_state24_indices" in man else None

        # Look for preds files matching *_{man_name}.npy
        pattern = os.path.join(preds_dir, f"*_{man_name}.npy")
        for fpath in glob.glob(pattern):
            fname = os.path.basename(fpath)
            if fname.startswith("pred_grids_") or fname.startswith("item_hit_"):
                continue

            # Determine arm name
            if fname.startswith("preds_"):
                arm = fname[len("preds_") : -(len(man_name) + 5)]
            else:
                arm = fname[: -(len(man_name) + 1)]

            pg_file = os.path.join(preds_dir, f"pred_grids_{arm}_{man_name}.npy")
            hit_file = os.path.join(preds_dir, f"item_hit_{arm}_{man_name}.npy")

            if os.path.exists(pg_file) and os.path.exists(hit_file):
                continue

            try:
                pred_hv = np.load(fpath)
                if pred_hv.ndim != 3 or pred_hv.shape[1] != 256 or pred_hv.shape[2] != 1152:
                    continue

                pred_grids = read_grids(reader, pred_hv).astype(np.uint8)
                np.save(pg_file, pred_grids)

                if man_name.startswith("open_"):
                    outs = [open_outcome(pred_grids, pack["src_grid"][src_idxs], pack["open_valid"][src_idxs, man["open_idx"]],
                                         pack["open_count"][src_idxs, man["open_idx"]], pack["open_all"][src_idxs, man["open_idx"]])]
                    hit = (outs[0]["label"] == 0).astype(np.uint8)
                else:
                    idx, tie = nearest_candidate_grid(pred_grids, pack["tgt_grid"][src_idxs][:, :K])
                    hit = ((idx == tgt_k) & ~tie).astype(np.uint8)

                np.save(hit_file, hit)
                print(f"  [Compacted] {arm} on {man_name} -> {pg_file} ({pred_grids.nbytes / 1024:.1f} KB)")
            except Exception as e:
                print(f"  Error compacting {fpath}: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--pack", default="data/gridA_test360_httpx.npz")
    p.add_argument("--reader", default="reader.npz")
    p.add_argument("--cand-hv", default="outputs/hv/cand_test360_httpx_22.npy")
    p.add_argument("--out-dir", default="outputs/m2")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    pack_path = os.path.join(ws_root, args.pack) if not os.path.isabs(args.pack) else args.pack
    pack = SP.load_pack(pack_path)
    eval_manifests = SP.build_cg_eval_manifests(pack)

    reader_path = os.path.join(ws_root, args.reader) if not os.path.isabs(args.reader) else args.reader
    reader = load_model(reader_path)

    cand_hv_path = os.path.join(ws_root, args.cand_hv) if not os.path.isabs(args.cand_hv) else args.cand_hv
    preds_dir = os.path.join(ws_root, args.out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    summary_path = os.path.join(ws_root, args.out_dir, "nondg_baseline_manifest_summary.json")
    master_summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                master_summary = json.load(f)
        except Exception:
            master_summary = {}

    # 1. Evaluate source_copy
    sc_res = eval_source_copy(pack, eval_manifests, preds_dir)
    master_summary["source_copy"] = sc_res

    # 2. Evaluate reader_ceiling if cand_hv exists
    if os.path.exists(cand_hv_path):
        rc_res = eval_reader_ceiling(pack, eval_manifests, cand_hv_path, reader, preds_dir)
        if rc_res:
            master_summary["reader_ceiling"] = rc_res

    # 3. Compact all existing pred dumps
    compact_existing_preds(pack, eval_manifests, reader, preds_dir)

    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nSaved updated manifest summary with controls to {summary_path}")


if __name__ == "__main__":
    main()
