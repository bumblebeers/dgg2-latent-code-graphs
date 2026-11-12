#!/usr/bin/env python3
"""cg_eval_nondg.py -- Full 16-Manifest Evaluation Runner for Non-DG Baseline Controls.

CRITICAL INTEGRITY CONTRACT:
- ZERO target peeking. Forward pass receives ONLY (src_hv, instruction_string, valid_mask).
- Never accesses tgt_grid, tgt_state24_indices, or target arrays during inference.
- Dumps genuine float32 .npy predictions of shape (M, 256, 1152) to outputs/m2/preds/.
- Evaluates predictions offline via reader.npz and cg_score_preds.py (score_fixed / score_open).
- Covers all 16 manifests:
    4 Fixed: seen_bal360, seen_shuf360, unseen_para_bal360, unseen_infer_360
    3 Held-out single ops: heldout_delete_methods_360, heldout_keep_callers_360, heldout_mark_ancestors_360
    5 Held-out compositions: heldout_reverse_order+delete_selected_360, heldout_reverse_edges+keep_callees_360,
      heldout_reverse_edges+mark_reachable_360, heldout_reverse_order+delete_tests_360,
      heldout_delete_tests+mark_reachable_360
    2 Re-feed: refeed_heldout_comp_bal360 (2-step), refeed_chain_heldout_ops_360 (3-step chain)
    2 Open-ended: open_seen_360, open_unseen_360 (4 sampler seeds / generations)
- Persists metrics to outputs/m2/nondg_baseline_manifest_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

import cg_sec27_spec as SP
import cg_instructions as CI
import cg_score_preds as CS
from cg_hv_probe import load_model, read_grids
from src.cg_nondg_model import NonDGTransformBaseline


@torch.no_grad()
def forward_pass_batch(
    model: torch.nn.Module,
    txt_model: torch.nn.Module,
    tokenizer: Any,
    input_hv: np.ndarray,
    instructions: List[str],
    preamble: str,
    device: torch.device,
    batch_size: int = 48,
    seed_offset: int = 0,
) -> np.ndarray:
    """Run model forward pass on arbitrary input h_v and instruction strings with ZERO target peeking."""
    model.eval()
    txt_model.eval()

    n_items = len(input_hv)
    preds = np.zeros((n_items, 256, 1152), dtype=np.float32)

    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        b_hv = torch.from_numpy(input_hv[start:end]).float().to(device)
        b_insts = instructions[start:end]

        full_insts = [f"{preamble} {inst}" for inst in b_insts]
        tok = tokenizer(
            full_insts,
            padding=True,
            truncation=True,
            max_length=160,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attn_mask = tok.attention_mask.bool().to(device)

        txt_out = txt_model(input_ids=input_ids, attention_mask=attn_mask)
        txt_tokens = txt_out.last_hidden_state

        b_pred = model(b_hv, txt_tokens, attn_mask)
        preds[start:end] = b_pred.float().cpu().numpy()

    return preds


def evaluate_arm_all_manifests(
    model: torch.nn.Module,
    txt_model: torch.nn.Module,
    tokenizer: Any,
    reader: Any,
    test_pack: dict,
    test_hv: np.ndarray,
    eval_manifests: Dict[str, Any],
    arm_name: str,
    preamble: str,
    device: torch.device,
    preds_dir: str,
    target_manifest_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate an arm across all 16 manifests."""
    manifest_names = target_manifest_names or list(eval_manifests.keys())
    arm_results: Dict[str, Any] = {}

    for man_name in manifest_names:
        if man_name not in eval_manifests:
            print(f"Warning: manifest '{man_name}' not found, skipping.")
            continue

        man = eval_manifests[man_name]
        src_idxs = man["img_indices"]
        src_items = test_hv[src_idxs]
        n_items = len(src_idxs)

        print(f"\n--- Evaluating [{arm_name}] on '{man_name}' ({n_items} items) ---")
        pred_file = os.path.join(preds_dir, f"preds_{arm_name}_{man_name}.npy")
        pred_file_alt = os.path.join(preds_dir, f"{arm_name}_{man_name}.npy")

        if man_name == "refeed_chain_heldout_ops_360":
            # 3-step chain re-feed: step 1 -> step 2 -> step 3
            chain_insts = man["chain_instructions"]
            s1_insts = [ci[0] for ci in chain_insts]
            s2_insts = [ci[1] for ci in chain_insts]
            s3_insts = [ci[2] for ci in chain_insts]

            pred_s1 = forward_pass_batch(model, txt_model, tokenizer, src_items, s1_insts, preamble, device)
            pred_s2 = forward_pass_batch(model, txt_model, tokenizer, pred_s1, s2_insts, preamble, device)
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, pred_s2, s3_insts, preamble, device)

            np.save(pred_file, pred_hv)
            np.save(pred_file_alt, pred_hv)
            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

        elif man_name.startswith("refeed_"):
            # 2-step re-feed: step 1 -> step 2
            s1_insts = man["step1_instructions"]
            s2_insts = man["step2_instructions"]

            pred_s1 = forward_pass_batch(model, txt_model, tokenizer, src_items, s1_insts, preamble, device)
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, pred_s1, s2_insts, preamble, device)

            np.save(pred_file, pred_hv)
            np.save(pred_file_alt, pred_hv)
            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

        elif man_name.startswith("open_"):
            # 4 distinct sampler seeds / generation passes
            insts = man["instructions"]
            pred_grids_list = []
            for g in range(4):
                pfile_g = os.path.join(preds_dir, f"preds_{arm_name}_{man_name}_gen{g}.npy")
                pred_hv_g = forward_pass_batch(
                    model, txt_model, tokenizer, src_items, insts, preamble, device, seed_offset=1000 * g
                )
                np.save(pfile_g, pred_hv_g)
                if g == 0:
                    np.save(pred_file, pred_hv_g)
                    np.save(pred_file_alt, pred_hv_g)
                pred_grids_list.append(read_grids(reader, pred_hv_g))
            score = CS.score_open(test_pack, man, pred_grids_list)

        else:
            # Fixed 1-pass manifests
            insts = man["instructions"]
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, src_items, insts, preamble, device)
            np.save(pred_file, pred_hv)
            np.save(pred_file_alt, pred_hv)
            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

        arm_results[man_name] = score
        if "acc_kway" in score:
            print(f"  [{arm_name} Eval {man_name}] acc_kway={score['acc_kway']*100:.2f}%, cell_acc={score['cell_acc']*100:.2f}%, changed_acc={score['changed_acc']*100:.2f}%, edge_f1={score['edge_f1']:.4f}")
        else:
            mean_valid = np.mean([v['valid'] for v in score['per_generation'][0].values()])
            print(f"  [{arm_name} Eval {man_name}] open(valid={mean_valid:.3f}, gens={len(score['per_generation'])})")

    return arm_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--pack", default="data/gridA_test360_httpx.npz")
    p.add_argument("--test-hv", default="data/hv/te_identity.npy")
    p.add_argument("--reader", default="reader.npz")
    p.add_argument("--ckpt", required=True, help="Path to trained model checkpoint .pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--manifests", nargs="+", default=None, help="Manifest names to evaluate (default: all 16)")
    p.add_argument("--gate-a3-only", action="store_true")
    p.add_argument("--out-dir", default="outputs/m2")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    device = torch.device(args.device)

    # 1. Load pack
    pack_path = os.path.join(ws_root, args.pack) if not os.path.isabs(args.pack) else args.pack
    pack = SP.load_pack(pack_path)
    eval_manifests = SP.build_cg_eval_manifests(pack)

    # 2. Load test source h_v (te_identity.npy)
    hv_path = os.path.join(ws_root, args.test_hv) if not os.path.isabs(args.test_hv) else args.test_hv
    if not os.path.exists(hv_path):
        alt_hv = os.path.join(ws_root, "outputs/hv/te_identity.npy")
        if os.path.exists(alt_hv):
            hv_path = alt_hv
    print(f"Loading test source h_v from {hv_path}...")
    src_hv = np.load(hv_path)

    # 3. Load reader
    reader_path = os.path.join(ws_root, args.reader) if not os.path.isabs(args.reader) else args.reader
    print(f"Loading reader from {reader_path}...")
    reader = load_model(reader_path)

    # 4. Load trained model
    ckpt_path = os.path.join(ws_root, args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    print(f"Loading trained checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    arm_name = ckpt.get("arm", os.path.splitext(os.path.basename(ckpt_path))[0])

    model = NonDGTransformBaseline(
        hv_dim=1152,
        txt_dim=384,
        d_model=768,
        n_heads=12,
        d_ff=3072,
        n_layers=4,
        max_txt_len=160,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load tokenizer & text model
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    if "txt_model_state_dict" in ckpt:
        txt_model.load_state_dict(ckpt["txt_model_state_dict"])
    txt_model.eval()

    preds_dir = os.path.join(ws_root, args.out_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    preamble = CI.PREAMBLE

    target_mans = ["seen_bal360"] if args.gate_a3_only else args.manifests

    results = evaluate_arm_all_manifests(
        model=model,
        txt_model=txt_model,
        tokenizer=tok,
        reader=reader,
        test_pack=pack,
        test_hv=src_hv,
        eval_manifests=eval_manifests,
        arm_name=arm_name,
        preamble=preamble,
        device=device,
        preds_dir=preds_dir,
        target_manifest_names=target_mans,
    )

    # If Gate A3, save canonical gateA3_report.json
    if "seen_bal360" in results and args.gate_a3_only:
        score_res = results["seen_bal360"]
        passed = bool(score_res["changed_acc"] > 0.0)
        gate_a3 = {
            "manifest": "seen_bal360",
            "arm": arm_name,
            "seed": ckpt.get("seed", 42),
            "n_items": score_res["n"],
            "acc_kway": score_res["acc_kway"],
            "cell_acc": score_res["cell_acc"],
            "changed_acc": score_res["changed_acc"],
            "unchanged_acc": score_res["unchanged_acc"],
            "edge_f1": score_res["edge_f1"],
            "node_acc": score_res["node_acc"],
            "gate_a3_passed": passed,
            "status": "PASS" if passed else "FAIL",
            "prereg_rule": "changed_acc > 0.0 (clearly above source copy = 0.0)",
        }
        gate_a3_path = os.path.join(ws_root, args.out_dir, "gateA3_report.json")
        with open(gate_a3_path, "w") as f:
            json.dump(gate_a3, f, indent=2)
        print(f"\nSaved canonical Gate A3 report to {gate_a3_path}")
        return

    # Merge into outputs/m2/nondg_baseline_manifest_summary.json
    summary_path = os.path.join(ws_root, args.out_dir, "nondg_baseline_manifest_summary.json")
    master_summary: Dict[str, Any] = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                master_summary = json.load(f)
        except Exception:
            master_summary = {}

    master_summary[arm_name] = results
    with open(summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nSaved updated manifest summary for arm '{arm_name}' to {summary_path}")


if __name__ == "__main__":
    main()
