#!/usr/bin/env python3
"""cg_eval_nondg_expA_click.py -- Non-DG Baseline Evaluation on Secondary Held-Out Repository Click (N=360) for Experiment A.

Evaluates all 16 manifests on data/gridA_test360_click.npz:
- 4 Fixed: seen_bal360, seen_shuf360, unseen_para_bal360, unseen_infer_360
- 3 Held-out single ops: heldout_delete_methods_360, heldout_keep_callers_360, heldout_mark_ancestors_360
- 5 Held-out compositions: heldout_reverse_order+delete_selected_360, heldout_reverse_edges+keep_callees_360,
  heldout_reverse_edges+mark_reachable_360, heldout_reverse_order+delete_tests_360,
  heldout_delete_tests+mark_reachable_360
- 2 Re-feed: refeed_heldout_comp_bal360 (2-step), refeed_chain_heldout_ops_360 (3-step chain)
- 2 Open-ended: open_seen_360, open_unseen_360 (4 sampler seeds / generations)

Arms:
- nondg_enc_lora_s42
- nondg_enc_lora_s1337
- nondg_frozen_inst_s42
- nondg_frozen_inst_s1337
- source_copy (parameter-free baseline)

Outputs:
- Compact .npy grid predictions: outputs/m2/preds/pred_grids_{arm}_{manifest}_click.npy
- Per-item outcome vectors: outputs/m2/preds/item_hit_{arm}_{manifest}_click.npy
- Metrics summary: outputs/m2/nondg_click_summary.json
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "cg"))
sys.path.insert(0, REPO_ROOT)

import cg_sec27_spec as SP
import cg_instructions as CI
import cg_score_preds as CS
from cg_hv_probe import load_model, read_grids
from cg_metrics import nearest_candidate_grid, open_outcome
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
    """Run model forward pass on input h_v and instruction strings with ZERO target peeking."""
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


def eval_source_copy(pack: dict, manifests: Dict[str, Any], preds_dir: str) -> Dict[str, Any]:
    """Evaluate source_copy baseline across all 16 manifests on click."""
    results: Dict[str, Any] = {}
    print("\n=== Evaluating source_copy Baseline on Click ===")

    for man_name, man in manifests.items():
        src_idxs = man["img_indices"]
        src_grids = pack["src_grid"][src_idxs].astype(np.uint8)

        pg_file = os.path.join(preds_dir, f"pred_grids_source_copy_{man_name}_click.npy")
        np.save(pg_file, src_grids)

        if man_name.startswith("open_"):
            score = CS.score_open(pack, man, [src_grids] * 4)
            outs = [
                open_outcome(
                    g,
                    pack["src_grid"][src_idxs],
                    pack["open_valid"][src_idxs, man["open_idx"]],
                    pack["open_count"][src_idxs, man["open_idx"]],
                    pack["open_all"][src_idxs, man["open_idx"]],
                )
                for g in [src_grids] * 4
            ]
            hit = (outs[0]["label"] == 0).astype(np.uint8)
            mean_valid = np.mean([v["valid"] for v in score["per_generation"][0].values()])
            print(f"  [source_copy Eval {man_name}] open(valid={mean_valid:.3f})")
        else:
            score = CS.score_fixed(pack, man, src_grids)
            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(src_grids, pack["tgt_grid"][src_idxs][:, :K])
            k = np.array(man["tgt_state24_indices"])
            hit = ((idx == k) & ~tie).astype(np.uint8)
            print(
                f"  [source_copy Eval {man_name}] acc_kway={score['acc_kway']*100:.2f}%, changed_acc={score['changed_acc']*100:.2f}%"
            )

        hit_file = os.path.join(preds_dir, f"item_hit_source_copy_{man_name}_click.npy")
        np.save(hit_file, hit)
        results[man_name] = score

    return results


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
) -> Dict[str, Any]:
    """Evaluate an arm across all 16 manifests on click."""
    arm_results: Dict[str, Any] = {}

    for man_name, man in eval_manifests.items():
        src_idxs = man["img_indices"]
        src_items = test_hv[src_idxs]
        n_items = len(src_idxs)

        print(f"\n--- Evaluating [{arm_name}] on '{man_name}' ({n_items} items) ---")
        grid_file = os.path.join(preds_dir, f"pred_grids_{arm_name}_{man_name}_click.npy")
        hit_file = os.path.join(preds_dir, f"item_hit_{arm_name}_{man_name}_click.npy")

        if man_name == "refeed_chain_heldout_ops_360":
            chain_insts = man["chain_instructions"]
            s1_insts = [ci[0] for ci in chain_insts]
            s2_insts = [ci[1] for ci in chain_insts]
            s3_insts = [ci[2] for ci in chain_insts]

            pred_s1 = forward_pass_batch(model, txt_model, tokenizer, src_items, s1_insts, preamble, device)
            pred_s2 = forward_pass_batch(model, txt_model, tokenizer, pred_s1, s2_insts, preamble, device)
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, pred_s2, s3_insts, preamble, device)

            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(pred_grids, test_pack["tgt_grid"][src_idxs][:, :K])
            k = np.array(man["tgt_state24_indices"])
            hit = ((idx == k) & ~tie).astype(np.uint8)

            np.save(grid_file, pred_grids)
            np.save(hit_file, hit)

        elif man_name.startswith("refeed_"):
            s1_insts = man["step1_instructions"]
            s2_insts = man["step2_instructions"]

            pred_s1 = forward_pass_batch(model, txt_model, tokenizer, src_items, s1_insts, preamble, device)
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, pred_s1, s2_insts, preamble, device)

            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(pred_grids, test_pack["tgt_grid"][src_idxs][:, :K])
            k = np.array(man["tgt_state24_indices"])
            hit = ((idx == k) & ~tie).astype(np.uint8)

            np.save(grid_file, pred_grids)
            np.save(hit_file, hit)

        elif man_name.startswith("open_"):
            insts = man["instructions"]
            pred_grids_list = []
            for g in range(4):
                pred_hv_g = forward_pass_batch(
                    model, txt_model, tokenizer, src_items, insts, preamble, device, seed_offset=1000 * g
                )
                pred_grids_list.append(read_grids(reader, pred_hv_g))

            score = CS.score_open(test_pack, man, pred_grids_list)
            outs = [
                open_outcome(
                    g,
                    test_pack["src_grid"][src_idxs],
                    test_pack["open_valid"][src_idxs, man["open_idx"]],
                    test_pack["open_count"][src_idxs, man["open_idx"]],
                    test_pack["open_all"][src_idxs, man["open_idx"]],
                )
                for g in pred_grids_list
            ]
            hit = (outs[0]["label"] == 0).astype(np.uint8)

            np.save(grid_file, pred_grids_list[0])
            np.save(hit_file, hit)

        else:
            insts = man["instructions"]
            pred_hv = forward_pass_batch(model, txt_model, tokenizer, src_items, insts, preamble, device)
            pred_grids = read_grids(reader, pred_hv)
            score = CS.score_fixed(test_pack, man, pred_grids)

            K = len(SP.CG_CANDIDATE_STATES)
            idx, tie = nearest_candidate_grid(pred_grids, test_pack["tgt_grid"][src_idxs][:, :K])
            k = np.array(man["tgt_state24_indices"])
            hit = ((idx == k) & ~tie).astype(np.uint8)

            np.save(grid_file, pred_grids)
            np.save(hit_file, hit)

        arm_results[man_name] = score
        if "acc_kway" in score:
            print(
                f"  [{arm_name} Eval {man_name}] acc_kway={score['acc_kway']*100:.2f}%, "
                f"cell_acc={score['cell_acc']*100:.2f}%, changed_acc={score['changed_acc']*100:.2f}%, "
                f"edge_f1={score['edge_f1']:.4f}"
            )
        else:
            mean_valid = np.mean([v["valid"] for v in score["per_generation"][0].values()])
            print(f"  [{arm_name} Eval {man_name}] open(valid={mean_valid:.3f}, gens={len(score['per_generation'])})")

    return arm_results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace-root", default=REPO_ROOT)
    p.add_argument("--pack", default="data/gridA_test360_click.npz")
    p.add_argument("--test-hv", default="data/hv/te_click_identity.npy")
    p.add_argument("--reader", default="reader.npz")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-summary", default="outputs/m2/nondg_click_summary.json")
    p.add_argument("--preds-dir", default="outputs/m2/preds")
    args = p.parse_args()

    ws_root = os.path.abspath(args.workspace_root)
    device = torch.device(args.device)
    out_summary_path = os.path.join(ws_root, args.out_summary) if not os.path.isabs(args.out_summary) else args.out_summary
    preds_dir = os.path.join(ws_root, args.preds_dir) if not os.path.isabs(args.preds_dir) else args.preds_dir
    os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
    os.makedirs(preds_dir, exist_ok=True)

    # 1. Load pack
    pack_path = os.path.join(ws_root, args.pack) if not os.path.isabs(args.pack) else args.pack
    pack = SP.load_pack(pack_path)
    eval_manifests = SP.build_cg_eval_manifests(pack)

    # 2. Load test source h_v
    hv_path = os.path.join(ws_root, args.test_hv) if not os.path.isabs(args.test_hv) else args.test_hv
    if not os.path.exists(hv_path):
        alt_hv = os.path.join(ws_root, "outputs/hv/te_click_identity.npy")
        if os.path.exists(alt_hv):
            hv_path = alt_hv
    print(f"Loading test source h_v from {hv_path}...")
    src_hv = np.load(hv_path)

    # 3. Load reader
    reader_path = os.path.join(ws_root, args.reader) if not os.path.isabs(args.reader) else args.reader
    print(f"Loading reader from {reader_path}...")
    reader = load_model(reader_path)

    # 4. Load tokenizer & text model
    tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    txt_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    txt_model.eval()

    preamble = CI.PREAMBLE

    master_summary: Dict[str, Any] = {}
    if os.path.exists(out_summary_path):
        try:
            with open(out_summary_path, "r") as f:
                master_summary = json.load(f)
        except Exception:
            master_summary = {}

    # Evaluate source_copy
    print("\n--- Evaluating parameter-free control: source_copy ---")
    master_summary["source_copy"] = eval_source_copy(pack, eval_manifests, preds_dir)
    with open(out_summary_path, "w") as f:
        json.dump(master_summary, f, indent=2)

    trained_arms = [
        ("nondg_enc_lora_s42", "outputs/m2/checkpoints/nondg_enc_lora_s42.pt"),
        ("nondg_enc_lora_s1337", "outputs/m2/checkpoints/nondg_enc_lora_s1337.pt"),
        ("nondg_frozen_inst_s42", "outputs/m2/checkpoints/nondg_frozen_inst_s42.pt"),
        ("nondg_frozen_inst_s1337", "outputs/m2/checkpoints/nondg_frozen_inst_s1337.pt"),
    ]

    for arm_name, ckpt_rel in trained_arms:
        ckpt_path = os.path.join(ws_root, ckpt_rel)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        print(f"\n========================================================")
        print(f"=== [Click Exp A] Evaluating Arm: {arm_name} ===")
        print(f"========================================================")
        print(f"Loading trained checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)

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

        if "txt_model_state_dict" in ckpt:
            txt_model.load_state_dict(ckpt["txt_model_state_dict"])
        else:
            # Re-init fresh weights for frozen instruction variant
            fresh_txt = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
            txt_model.load_state_dict(fresh_txt.state_dict())
        txt_model.eval()

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
        )

        master_summary[arm_name] = results
        with open(out_summary_path, "w") as f:
            json.dump(master_summary, f, indent=2)
        print(f"Updated summary for {arm_name} saved to {out_summary_path}")

    print(f"\nAll Experiment A evaluations on click complete! Final summary saved to {out_summary_path}")


if __name__ == "__main__":
    main()
