#!/usr/bin/env python3
"""cg_score_preds.py -- score dumped predictions for Experiment A offline (no DG, no GPU).

The runner only has to save, for each evaluation manifest, the predicted h_v in manifest order as a
float array of shape (M, 256, 1152) (.npy). Everything graph-specific is computed here:

    python3 cg/cg_score_preds.py --pack data/gridA_test360_httpx.npz --reader reader.npz \\
        --manifest unseen_para_bal360 --pred preds_unseen.npy --out unseen.json

  --reader     cell reader written by `cg_hv_probe.py --save-model` (fitted on TRUE h_v only)
  --manifest   a key of cg_sec27_spec.build_cg_eval_manifests(pack)
  --pred       one file; for open-ended manifests pass several (one per generation seed) to get diversity
  --grids      alternative to --pred/--reader: already-read palette grids (M, n, n)

Fixed-target manifests report: K-way nearest-candidate accuracy on reader grids (ties count as wrong),
exact grid match, changed/unchanged-cell accuracy, edge F1, the per-operation breakdown with the exact
predictions made, the two-step attribution (first step only / second step only / source copy), accuracy
by how many cells the edit changes, and for marking operations accuracy by reachability depth.
Open-ended manifests report valid / none / all / ambiguous rates and, with several --pred files, diversity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg_sec27_spec as SP  # noqa: E402
from cg_hv_probe import fit_ridge, load_model, read_grids  # noqa: E402
from cg_metrics import (accuracy_by_bucket, grid_metrics, nearest_candidate_grid, open_diversity, open_outcome,  # noqa: E402
                        open_summary, summarize)


def score_fixed(pack, man, pred_grid: np.ndarray) -> dict:
    src = man["img_indices"]
    k = man["tgt_state24_indices"]
    K = len(SP.CG_CANDIDATE_STATES)
    true = pack["tgt_grid"][src, k]
    m = grid_metrics(pred_grid, true, pack["src_grid"][src])
    idx, tie = nearest_candidate_grid(pred_grid, pack["tgt_grid"][src][:, :K])
    rows = [{"src": int(s), "target": o, "target_idx": int(t)} for s, o, t in zip(src, man["op_names"], k)]
    # two-step operations are named "a+b" in the candidate pool, so summarize() attributes them directly
    summ = summarize(rows, idx, SP.CG_CANDIDATE_STATES, tie)
    hit = (idx == k) & ~tie
    out = {"n": len(src), "acc_kway": float(hit.mean()), "chance": 1.0 / K,
           **{key: float(np.nanmean(m[key])) for key in ("exact", "cell_acc", "changed_acc", "unchanged_acc", "edge_f1", "node_acc")},
           "per_target": summ["per_target"], "two_step": summ["two_step"],
           "acc_by_changed_cells": accuracy_by_bucket(m["n_changed"], hit, [0, 1, 8, 32, 96, 1e9])}
    ops = np.array(man["op_names"])
    mark = np.isin(ops, ["mark_reachable", "mark_ancestors"])
    if mark.any():
        depth = pack["reach_depth"][src]     # depth of the forward reach; reported for both marking operations
        out["marking_acc_by_reach_depth"] = {"depth_1": {"n": int((mark & (depth < 2)).sum()),
                                                        "acc": float(hit[mark & (depth < 2)].mean()) if (mark & (depth < 2)).any() else None},
                                             "depth_2_plus": {"n": int((mark & (depth >= 2)).sum()),
                                                              "acc": float(hit[mark & (depth >= 2)].mean()) if (mark & (depth >= 2)).any() else None}}
    return out


def score_open(pack, man, pred_grids: list) -> dict:
    src, k = man["img_indices"], man["open_idx"]
    S, V, C, A = pack["src_grid"][src], pack["open_valid"][src, k], pack["open_count"][src, k], pack["open_all"][src, k]
    rows = [{"target": o} for o in man["op_names"]]
    outs = [open_outcome(g, S, V, C, A) for g in pred_grids]
    res = {"n": len(src), "generations": len(outs), "per_generation": [open_summary(rows, o) for o in outs]}
    if len(outs) > 1:
        res["diversity"] = open_diversity(np.stack([o["choice"] for o in outs], 1), C)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack"); ap.add_argument("--manifest"); ap.add_argument("--reader")
    ap.add_argument("--pred", nargs="+"); ap.add_argument("--grids", nargs="+"); ap.add_argument("--out")
    ap.add_argument("--list", action="store_true", help="list manifest names and sizes for --pack")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    pack = SP.load_pack(a.pack)
    mans = SP.build_cg_eval_manifests(pack)
    if a.list:
        for k, v in mans.items():
            print(f"{k:34s} {len(v['img_indices'])}")
        return
    man = mans[a.manifest]
    if a.grids:
        grids = [np.load(p) for p in a.grids]
    else:
        reader = load_model(a.reader)
        grids = [read_grids(reader, np.load(p)) for p in a.pred]
    assert all(len(g) == len(man["img_indices"]) for g in grids), "prediction count does not match the manifest"
    res = score_open(pack, man, grids) if a.manifest.startswith("open_") else score_fixed(pack, man, grids[0])
    res["manifest"], res["pack"] = a.manifest, os.path.basename(a.pack)
    txt = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


def self_test():
    """End-to-end on synthetic 'h_v': token vector = one-hot of the cell's palette class."""
    data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    pack = SP.load_pack(os.path.join(data, "gridA_test360_httpx.npz"))
    def fake_hv(grids):
        return np.eye(8, dtype=np.float32)[grids.reshape(len(grids), 256)]
    fit_grids = pack["tgt_grid"][:60, [0, 5, 6]].reshape(-1, 16, 16)       # source, mark_reachable, delete_tests:
    reader = fit_ridge(fake_hv(fit_grids).reshape(-1, 8).astype(np.float64),  # every palette class must be seen
                       fit_grids.reshape(-1, 1), 8, lam=1e-6)
    mans = SP.build_cg_eval_manifests(pack)
    for name, man in mans.items():
        if name.startswith("open_") or "chain" in name:
            continue
        true = pack["tgt_grid"][man["img_indices"], man["tgt_state24_indices"]]
        r = score_fixed(pack, man, read_grids(reader, fake_hv(true)))
        want = 0.0 if name == "seen_shuf360" else 1.0
        if name == "seen_shuf360":      # predictions follow the shuffled instruction: every item must be wrong
            wrong = [SP.STATE_TO_IDX[SP.CG_OPERATIONS[o]["target_state"]] for o in man["shuffled_from_ops"]]
            r = score_fixed(pack, man, read_grids(reader, fake_hv(pack["tgt_grid"][man["img_indices"], wrong])))
        assert r["acc_kway"] == want, (name, r["acc_kway"])
        copy = score_fixed(pack, man, pack["src_grid"][man["img_indices"]])
        assert copy["changed_acc"] == 0.0 or name.startswith("seen") or name.startswith("unseen")
    man = mans["refeed_heldout_comp_bal360"]                       # a model that only does the second step
    s2 = score_fixed(pack, man, pack["tgt_grid"][man["img_indices"], man["step2_state24_indices"]])
    assert s2["acc_kway"] == 0.0 and s2["two_step"]["second_only"] == 360
    man = mans["open_unseen_360"]
    src, k = man["img_indices"], man["open_idx"]
    rng = np.random.default_rng(0)
    gens = [pack["open_valid"][src, k, (rng.random(360) * pack["open_count"][src, k]).astype(int)] for _ in range(4)]
    r = score_open(pack, man, gens)
    assert all(v["valid"] == 1.0 for g in r["per_generation"] for v in g.values()) and r["diversity"]["mean_distinct_answers"] > 1.5
    r0 = score_open(pack, man, [pack["src_grid"][src]])
    assert all(v["none"] == 1.0 for v in r0["per_generation"][0].values())
    print("self-test ok: oracle 100%, shuffled 0%, second-step-only attributed, open-ended commit/copy separated")


if __name__ == "__main__":
    main()
