#!/usr/bin/env python3
"""cg_eval_nondg_expB.py -- Multi-Manifest Evaluation Runner for Non-DG Baseline on Experiment B.

Evaluates Experiment B arms across all 9 manifests:
1. seen (data/nodeB_test360_httpx.json.gz, instr_seen)
2. unseen (data/nodeB_test360_httpx.json.gz, instr_unseen)
3. instr_shuf (shuffled/deranged instruction control)
4. unseenargs (data/nodeB_test360_httpx_unseenargs.json.gz)
5. renameclass (data/nodeB_test180_httpx_renameclass.json.gz)
6. multi_k2_1pass (data/nodeB_multi_test360_httpx_k2.json.gz, 1-pass joint instruction)
7. multi_k2_refeed (data/nodeB_multi_test360_httpx_k2.json.gz, re-feed site instructions)
8. multi_k4_1pass (data/nodeB_multi_test360_httpx_k4.json.gz, 1-pass joint instruction)
9. multi_k4_refeed (data/nodeB_multi_test360_httpx_k4.json.gz, re-feed site instructions)

Parameter-free controls:
- floor_source_copy
- floor_training_mean
- ceiling_oracle_target

Outputs:
- Compact .npy prediction arrays: outputs/m3/preds/pred_vecs_{arm}_{manifest}.npy
- Per-item outcome vectors: outputs/m3/preds/item_hit_{arm}_{manifest}.npy
- Site hit vectors: outputs/m3/preds/site_hit_{arm}_{manifest}.npy
- Complete metrics summary: outputs/m3/nondg_expB_manifest_summary.json
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

# Workspace root
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


@torch.no_grad()
def evaluate_refeed_chain(
    model: torch.nn.Module,
    txt_model: torch.nn.Module,
    tokenizer: Any,
    ds: dict,
    ev: Evaluator,
    items: List[Tuple[int, Any]],
    k_steps: int,
    device: torch.device,
    batch_size: int = 48,
    max_length: int = 160,
) -> np.ndarray:
    """Evaluate multi-symbol manifest via re-feed chain across clean site instructions."""
    n_items = len(items)
    preds = np.zeros((n_items, 256, 384), dtype=np.float32)

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        b_items = items[start:end]

        # Initial source vectors: (B, 256, 384)
        cur_vecs = torch.from_numpy(np.stack([ev.source(i) for i, _ in b_items])).float().to(device)

        # Step through each site in the chain
        for step in range(k_steps):
            step_insts = [ds["sources"][i]["sites"][step]["instr_seen"][0] for i, _ in b_items]
            tok = tokenizer(
                step_insts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = tok.input_ids.to(device)
            attn_mask = tok.attention_mask.bool().to(device)

            txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
            txt_tokens = txt_out.last_hidden_state

            # Re-feed output of previous step
            cur_vecs = model(cur_vecs, txt_tokens, attn_mask)

        preds[start:end] = cur_vecs.float().cpu().numpy()

    return preds


def run_evaluation_for_arm(
    arm_name: str,
    model: Optional[torch.nn.Module],
    txt_model: Optional[torch.nn.Module],
    tokenizer: Any,
    manifest_packs: Dict[str, dict],
    evaluators: Dict[str, Evaluator],
    device: torch.device,
    preds_dir: str,
) -> Dict[str, Any]:
    """Runs all 9 manifests for a single arm or control."""
    print(f"\n========================================================")
    print(f"=== Evaluating Arm: {arm_name} ===")
    print(f"========================================================")

    results: Dict[str, Any] = {}

    manifest_names = [
        "seen",
        "unseen",
        "instr_shuf",
        "unseenargs",
        "renameclass",
        "multi_k2_1pass",
        "multi_k2_refeed",
        "multi_k4_1pass",
        "multi_k4_refeed",
    ]

    for m_name in manifest_names:
        t0 = time.time()
        if m_name in ("seen", "unseen", "instr_shuf"):
            ds = manifest_packs["test360"]
            ev = evaluators["test360"]
            items = default_items(ds)
            if m_name == "seen":
                instructions = [ds["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items]
            elif m_name == "unseen":
                instructions = [ds["sources"][i]["cands"][k]["instr_unseen"][0] for i, k in items]
            else:  # instr_shuf: deranged instructions
                N = len(items)
                instructions = [ds["sources"][(i + 1) % N]["cands"][items[(idx + 1) % N][1]]["instr_seen"][0] for idx, (i, k) in enumerate(items)]
        elif m_name == "unseenargs":
            ds = manifest_packs["unseenargs"]
            ev = evaluators["unseenargs"]
            items = default_items(ds)
            instructions = [ds["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items]
        elif m_name == "renameclass":
            ds = manifest_packs["renameclass"]
            ev = evaluators["renameclass"]
            items = default_items(ds)
            instructions = [ds["sources"][i]["cands"][k]["instr_seen"][0] for i, k in items]
        elif m_name == "multi_k2_1pass":
            ds = manifest_packs["k2"]
            ev = evaluators["k2"]
            items = default_items(ds)
            instructions = [ds["sources"][i]["instr_seen"][0] for i, _ in items]
        elif m_name == "multi_k2_refeed":
            ds = manifest_packs["k2"]
            ev = evaluators["k2"]
            items = default_items(ds)
        elif m_name == "multi_k4_1pass":
            ds = manifest_packs["k4"]
            ev = evaluators["k4"]
            items = default_items(ds)
            instructions = [ds["sources"][i]["instr_seen"][0] for i, _ in items]
        elif m_name == "multi_k4_refeed":
            ds = manifest_packs["k4"]
            ev = evaluators["k4"]
            items = default_items(ds)

        # Compute predictions
        if arm_name == "floor_source_copy":
            preds = np.stack([ev.source(i) for i, _ in items]).astype(np.float32)
        elif arm_name == "floor_training_mean":
            preds = np.zeros((len(items), 256, 384), dtype=np.float32)
        elif arm_name == "ceiling_oracle_target":
            preds = np.stack([ev.target(i, k) for i, k in items]).astype(np.float32)
        else:
            assert model is not None and txt_model is not None
            if m_name == "multi_k2_refeed":
                preds = evaluate_refeed_chain(model, txt_model, tokenizer, ds, ev, items, k_steps=2, device=device)
            elif m_name == "multi_k4_refeed":
                preds = evaluate_refeed_chain(model, txt_model, tokenizer, ds, ev, items, k_steps=4, device=device)
            else:
                preds = evaluate_single_pass(model, txt_model, tokenizer, ev, items, instructions, device=device)

        # Score predictions
        score = ev.score(preds, items)
        per_item_rows = score["per_item"]
        summary = {k: v for k, v in score.items() if k != "per_item"}

        # Macro aggregation across 156 primary symbols
        macro_agg = aggregate_per_primary_symbol(ds, items, per_item_rows)
        summary.update(macro_agg)

        # Persist compact .npy arrays into outputs/m3/preds/
        os.makedirs(preds_dir, exist_ok=True)
        pred_file = os.path.join(preds_dir, f"pred_vecs_{arm_name}_{m_name}.npy")
        np.save(pred_file, preds)

        item_hits = np.array([int(r["all_sites_hit"]) for r in per_item_rows], dtype=np.uint8)
        item_hit_file = os.path.join(preds_dir, f"item_hit_{arm_name}_{m_name}.npy")
        np.save(item_hit_file, item_hits)

        site_hits = np.array([float(r["site_hit"]) for r in per_item_rows], dtype=np.float32)
        site_hit_file = os.path.join(preds_dir, f"site_hit_{arm_name}_{m_name}.npy")
        np.save(site_hit_file, site_hits)

        t1 = time.time()
        results[m_name] = summary

        print(f"  [{arm_name} | {m_name}] site_hit: {summary['site_hit']*100:.2f}% | all_sites_hit: {summary['all_sites_hit']*100:.2f}% | top_is_site: {summary['top_is_site']*100:.2f}% | per_symbol: {summary.get('site_hit_per_symbol', 0.0)*100:.2f}% ({t1-t0:.2f}s)")

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--arm", default="all", help="Arm name (nondg_enc_lora_s42, nondg_enc_lora_s1337, nondg_frozen_inst_s42, nondg_frozen_inst_s1337, floor_source_copy, floor_training_mean, ceiling_oracle_target, or 'all')")
    p.add_argument("--ckpt", default=None, help="Specific checkpoint path to evaluate")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", default="outputs/m3")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    device = torch.device(args.device)
    out_dir = os.path.join(ws_root, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    preds_dir = os.path.join(out_dir, "preds")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    print(f"=== Starting Non-DG Experiment B Evaluations on {device} ===")
    print(f"Arm: {args.arm} | Workspace: {ws_root} | Out: {out_dir}")

    # 1. Load embeddings & standardization statistics
    print("Loading embeddings...")
    emb = Emb(os.path.join(ws_root, "emb.npz"))
    emb_train = Emb(os.path.join(ws_root, "emb_train.npz"))
    stats_vec = emb_train.vec

    # 2. Pre-load all test datasets & evaluators
    print("Loading datasets...")
    manifest_paths = {
        "test360": "data/nodeB_test360_httpx.json.gz",
        "unseenargs": "data/nodeB_test360_httpx_unseenargs.json.gz",
        "renameclass": "data/nodeB_test180_httpx_renameclass.json.gz",
        "k2": "data/nodeB_multi_test360_httpx_k2.json.gz",
        "k4": "data/nodeB_multi_test360_httpx_k4.json.gz",
    }
    manifest_packs = {}
    evaluators = {}
    for k, rel_path in manifest_paths.items():
        with gzip.open(os.path.join(ws_root, rel_path), "rt") as f:
            pack = json.load(f)
        manifest_packs[k] = pack
        evaluators[k] = Evaluator(pack, emb, stats_vec=stats_vec)

    # 3. Load text tokenizer
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Read existing summary if available
    summary_file = os.path.join(out_dir, "nondg_expB_manifest_summary.json")
    all_summaries: Dict[str, Any] = {}
    if os.path.exists(summary_file):
        try:
            with open(summary_file, "r") as f:
                all_summaries = json.load(f)
        except Exception:
            all_summaries = {}

    # Define arms to run
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

    for arm in arms_to_run:
        if arm in control_arms:
            arm_results = run_evaluation_for_arm(
                arm_name=arm,
                model=None,
                txt_model=None,
                tokenizer=tok,
                manifest_packs=manifest_packs,
                evaluators=evaluators,
                device=device,
                preds_dir=preds_dir,
            )
            all_summaries[arm] = arm_results
        else:
            # Locate checkpoint
            ckpt_path = args.ckpt
            if ckpt_path is None or not os.path.exists(ckpt_path):
                # Search candidates
                candidates = [
                    os.path.join(ws_root, f"outputs/m3/checkpoints/{arm}.pt"),
                    os.path.join(ws_root, f"outputs/m2/baselines/nondg_expB/{arm}.pt"),
                    os.path.join(ws_root, f"outputs/m3/checkpoints/bare_{arm}.pt"),
                ]
                if arm == "nondg_enc_lora_s42":
                    candidates.insert(0, os.path.join(ws_root, "outputs/m2/baselines/nondg_expB/nondg_expB_s42.pt"))
                    candidates.insert(1, os.path.join(ws_root, "outputs/m2/baselines/nondg_expB/best_nondg_expB_s42.pt"))

                found_ckpt = None
                for c in candidates:
                    if os.path.exists(c):
                        found_ckpt = c
                        break
                if found_ckpt is None:
                    print(f"Warning: Checkpoint for arm '{arm}' not found in candidates: {candidates}. Skipping.")
                    continue
                ckpt_path = found_ckpt

            print(f"Loading checkpoint for {arm} from {ckpt_path}...")
            model = NonDGTransformBaseline(
                hv_dim=384, txt_dim=384, d_model=768, n_heads=12, d_ff=3072, n_layers=4, max_txt_len=64
            ).to(device)
            txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
            txt_model.eval()

            loaded = torch.load(ckpt_path, map_location=device)
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
                manifest_packs=manifest_packs,
                evaluators=evaluators,
                device=device,
                preds_dir=preds_dir,
            )
            all_summaries[arm] = arm_results

        # Write progressive updates to summary file
        with open(summary_file, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Updated summary saved to {summary_file}")

    print(f"\nAll evaluations complete! Summary persisted to {summary_file}")


if __name__ == "__main__":
    main()
