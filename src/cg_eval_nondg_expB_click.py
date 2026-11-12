#!/usr/bin/env python3
"""cg_eval_nondg_expB_click.py -- Non-DG Baseline Evaluation on Secondary Held-Out Repository Click (N=360).

Evaluates Experiment B arms on data/nodeB_test360_click.json.gz across:
1. seen (data/nodeB_test360_click.json.gz, instr_seen)
2. unseen (data/nodeB_test360_click.json.gz, instr_unseen)
3. instr_shuf (shuffled/deranged instruction control)

Parameter-free controls:
- floor_source_copy
- floor_training_mean
- ceiling_oracle_target

Trained arms:
- nondg_enc_lora_s42
- nondg_enc_lora_s1337
- nondg_frozen_inst_s42
- nondg_frozen_inst_s1337

Outputs:
- Compact .npy prediction arrays: outputs/m3/preds/pred_vecs_{arm}_{manifest}_click.npy
- Per-item outcome vectors: outputs/m3/preds/item_hit_{arm}_{manifest}_click.npy
- Site hit vectors: outputs/m3/preds/site_hit_{arm}_{manifest}_click.npy
- Complete metrics summary: outputs/m3/nondg_expB_click_summary.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

from cg_node_gate import Emb  # noqa: E402
from cg_node_metrics import Evaluator, default_items  # noqa: E402
from src.cg_nondg_model import NonDGTransformBaseline  # noqa: E402


def aggregate_per_primary_symbol(
    pack: dict,
    items: Sequence[Tuple[int, Any]],
    per_item_rows: List[dict],
) -> Dict[str, Any]:
    """Computes macro-average metrics across unique primary symbols (242 in test360_click)."""
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


@torch.no_grad()
def evaluate_single_pass(
    model: torch.nn.Module,
    txt_model: torch.nn.Module,
    tokenizer: Any,
    ev: Evaluator,
    items: List[Tuple[int, Any]],
    instructions: List[str],
    device: torch.device,
    batch_size: int = 48,
    max_length: int = 160,
) -> np.ndarray:
    """Evaluate single-pass predictions for a given list of instructions."""
    n_items = len(items)
    preds = np.zeros((n_items, 256, 384), dtype=np.float32)

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        b_items = items[start:end]
        b_insts = instructions[start:end]

        src_list = [ev.source(i) for i, _ in b_items]
        b_src = torch.from_numpy(np.stack(src_list)).float().to(device)

        tok = tokenizer(
            b_insts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.bool().to(device)

        txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        txt_tokens = txt_out.last_hidden_state

        b_pred = model(b_src, txt_tokens, attn_mask)
        preds[start:end] = b_pred.float().cpu().numpy()

    return preds


def run_evaluation_for_arm(
    arm_name: str,
    model: Optional[torch.nn.Module],
    txt_model: Optional[torch.nn.Module],
    tokenizer: Any,
    ds: dict,
    ev: Evaluator,
    device: torch.device,
    preds_dir: str,
) -> Dict[str, Any]:
    """Runs all click manifests for a single arm or control."""
    print(f"\n========================================================")
    print(f"=== [Click Exp B] Evaluating Arm: {arm_name} ===")
    print(f"========================================================")

    results: Dict[str, Any] = {}
    manifest_names = ["seen", "unseen", "instr_shuf"]
    items = default_items(ds)

    for m_name in manifest_names:
        t0 = time.time()
        if m_name == "seen":
            instructions = [ds["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items]
        elif m_name == "unseen":
            instructions = [ds["sources"][i]["cands"][k]["instr_unseen"][0] for i, k in items]
        else:  # instr_shuf: deranged instructions
            N = len(items)
            instructions = [
                ds["sources"][(i + 1) % N]["cands"][items[(idx + 1) % N][1]]["instr_seen"][0]
                for idx, (i, k) in enumerate(items)
            ]

        # Compute predictions
        if arm_name == "floor_source_copy":
            preds = np.stack([ev.source(i) for i, _ in items]).astype(np.float32)
        elif arm_name == "floor_training_mean":
            preds = np.zeros((len(items), 256, 384), dtype=np.float32)
        elif arm_name == "ceiling_oracle_target":
            preds = np.stack([ev.target(i, k) for i, k in items]).astype(np.float32)
        else:
            assert model is not None and txt_model is not None
            preds = evaluate_single_pass(model, txt_model, tokenizer, ev, items, instructions, device=device)

        # Score predictions
        score = ev.score(preds, items)
        per_item_rows = score["per_item"]
        summary = {k: v for k, v in score.items() if k != "per_item"}

        # Macro aggregation across 242 unique primary symbols in click
        macro_agg = aggregate_per_primary_symbol(ds, items, per_item_rows)
        summary.update(macro_agg)

        # Persist compact .npy arrays into outputs/m3/preds/
        os.makedirs(preds_dir, exist_ok=True)
        pred_file = os.path.join(preds_dir, f"pred_vecs_{arm_name}_{m_name}_click.npy")
        np.save(pred_file, preds)

        item_hits = np.array([int(r["all_sites_hit"]) for r in per_item_rows], dtype=np.uint8)
        item_hit_file = os.path.join(preds_dir, f"item_hit_{arm_name}_{m_name}_click.npy")
        np.save(item_hit_file, item_hits)

        site_hits = np.array([float(r["site_hit"]) for r in per_item_rows], dtype=np.float32)
        site_hit_file = os.path.join(preds_dir, f"site_hit_{arm_name}_{m_name}_click.npy")
        np.save(site_hit_file, site_hits)

        t1 = time.time()
        results[m_name] = summary

        print(
            f"  [{arm_name} | {m_name}] site_hit: {summary['site_hit']*100:.2f}% | "
            f"all_sites_hit: {summary['all_sites_hit']*100:.2f}% | "
            f"top_is_site: {summary['top_is_site']*100:.2f}% | "
            f"per_symbol: {summary.get('site_hit_per_symbol', 0.0)*100:.2f}% ({t1-t0:.2f}s)"
        )

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--test-pack", default="data/nodeB_test360_click.json.gz")
    p.add_argument("--emb", default="emb.npz")
    p.add_argument("--emb-train", default="emb_train.npz")
    p.add_argument("--out-summary", default="outputs/m3/nondg_expB_click_summary.json")
    p.add_argument("--preds-dir", default="outputs/m3/preds")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--arm", default="all")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    device = torch.device(args.device)
    out_summary_path = os.path.join(ws_root, args.out_summary) if not os.path.isabs(args.out_summary) else args.out_summary
    preds_dir = os.path.join(ws_root, args.preds_dir) if not os.path.isabs(args.preds_dir) else args.preds_dir
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"=== Starting Non-DG Experiment B Click Evaluations on {device} ===")
    print(f"Pack: {args.test_pack} | Summary Out: {out_summary_path}")

    # Load embeddings & stats
    print("Loading embeddings...")
    emb = Emb(os.path.join(ws_root, args.emb) if not os.path.isabs(args.emb) else args.emb)
    emb_train = Emb(os.path.join(ws_root, args.emb_train) if not os.path.isabs(args.emb_train) else args.emb_train)
    stats_vec = emb_train.vec

    # Load test dataset & evaluator
    print("Loading click test dataset...")
    pack_path = os.path.join(ws_root, args.test_pack) if not os.path.isabs(args.test_pack) else args.test_pack
    with gzip.open(pack_path, "rt") as f:
        ds = json.load(f)
    ev = Evaluator(ds, emb, stats_vec=stats_vec)

    # Tokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    control_arms = ["floor_source_copy", "floor_training_mean", "ceiling_oracle_target"]
    trained_arms = [
        "nondg_enc_lora_s42",
        "nondg_enc_lora_s1337",
        "nondg_frozen_inst_s42",
        "nondg_frozen_inst_s1337",
    ]

    if args.arm == "all":
        arms_to_run = control_arms + trained_arms
    elif args.arm in control_arms or args.arm in trained_arms:
        arms_to_run = [args.arm]
    else:
        arms_to_run = [args.arm]

    all_summaries: Dict[str, Any] = {}
    if os.path.exists(out_summary_path):
        try:
            with open(out_summary_path, "r") as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = {}

    for arm in arms_to_run:
        if arm in control_arms:
            arm_results = run_evaluation_for_arm(
                arm_name=arm,
                model=None,
                txt_model=None,
                tokenizer=tok,
                ds=ds,
                ev=ev,
                device=device,
                preds_dir=preds_dir,
            )
            all_summaries[arm] = arm_results
        else:
            candidates = [
                os.path.join(ws_root, f"outputs/m3/checkpoints/{arm}.pt"),
                os.path.join(ws_root, f"outputs/m2/baselines/nondg_expB/{arm}.pt"),
            ]
            if arm == "nondg_enc_lora_s42":
                candidates.insert(0, os.path.join(ws_root, "outputs/m2/baselines/nondg_expB/nondg_expB_s42.pt"))

            found_ckpt = None
            for c in candidates:
                if os.path.exists(c):
                    found_ckpt = c
                    break
            if found_ckpt is None:
                raise FileNotFoundError(f"Checkpoint for arm '{arm}' not found in candidates: {candidates}")

            print(f"Loading checkpoint for {arm} from {found_ckpt}...")
            model = NonDGTransformBaseline(
                hv_dim=384, txt_dim=384, d_model=768, n_heads=12, d_ff=3072, n_layers=4, max_txt_len=64
            ).to(device)
            txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
            txt_model.eval()

            loaded = torch.load(found_ckpt, map_location=device)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                model.load_state_dict(loaded["model_state_dict"])
                if "txt_model_state_dict" in loaded:
                    txt_model.load_state_dict(loaded["txt_model_state_dict"])
            else:
                model.load_state_dict(loaded)
            model.eval()

            arm_results = run_evaluation_for_arm(
                arm_name=arm,
                model=model,
                txt_model=txt_model,
                tokenizer=tok,
                ds=ds,
                ev=ev,
                device=device,
                preds_dir=preds_dir,
            )
            all_summaries[arm] = arm_results

        with open(out_summary_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Updated summary saved to {out_summary_path}")

    print(f"\nAll evaluations on click complete! Final summary written to {out_summary_path}")


if __name__ == "__main__":
    main()
