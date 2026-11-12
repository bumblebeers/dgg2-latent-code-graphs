import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cg"))
import numpy as np
from cg_metrics import *
from cg_ops import ALL_TARGETS, render, grid_from_rgb16
from cg_grid_dataset import build_eval_manifests, build_train_schedule, per_string_manifest, text_question_answers

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
z = np.load(os.path.join(D, "gridA_test360_httpx.npz"))

def test_oracle_and_copy():
    man = build_eval_manifests(360)
    for name, rows in man.items():
        ti = np.array([r["target_idx"] for r in rows]); src = np.array([r["src"] for r in rows])
        true = z["tgt_grid"][src, ti]
        m = grid_metrics(true, true, z["src_grid"][src])
        assert m["exact"].mean() == 1 and m["edge_f1"].mean() == 1
        idx, tie = nearest_candidate_grid(true, z["tgt_grid"][src])
        s = summarize(rows, idx, ALL_TARGETS, tie)
        assert s["acc"] == 1.0 and abs(s["chance"] - 1 / 22) < 1e-9, (name, s["acc"])
        # source-copy baseline: right only on identity items, 0 on changed cells
        mc = grid_metrics(z["src_grid"][src], true, z["src_grid"][src])
        idc, _ = nearest_candidate_grid(z["src_grid"][src], z["tgt_grid"][src])
        sc = summarize(rows, idc, ALL_TARGETS)
        nonid = ti != 0
        assert np.nanmax(mc["changed_acc"][nonid]) == 0 and (idc == 0).all()
        assert abs(sc["acc"] - (ti == 0).mean()) < 1e-9
        if name.startswith("comp"):
            assert sc["two_step"]["source_copy"] == len(rows)

def test_balance_and_strings():
    man = build_eval_manifests(360)
    assert all(len(v) == 360 for v in man.values())
    from collections import Counter
    assert set(Counter(r["target"] for r in man["seen_bal"]).values()) == {40}
    assert set(Counter(r["target"] for r in man["heldout_op"]).values()) == {120}
    seen = {r["instruction"] for r in man["seen_bal"]} | {r["instruction"] for r in man["comp_seen"]}
    for k in ("unseen_bal", "heldout_op", "comp_unseen", "comp_heldout"):
        assert not ({r["instruction"] for r in man[k]} & seen)
    sched = build_train_schedule(1608)
    assert len(sched) == 3 and all(len(e) == 1608 * 6 for e in sched) and sum(map(len, sched)) // 8 == 3618
    tr_strings = {r["instruction"] for e in sched for r in e}
    assert tr_strings <= seen | {r["instruction"] for r in per_string_manifest(1, "seen")}
    for e in sched:
        for a, b in zip(e[0::2], e[1::2]):
            assert a["src"] == b["src"] and a["target"] != b["target"]
    trained = {r["target"] for e in sched for r in e}
    assert "keep_callers" not in trained and "mark_ancestors" not in trained and "delete_methods" not in trained and len(trained) == 14
    assert not any(t in trained for t in ALL_TARGETS[17:])

def test_rgb16_path_and_hv():
    src = np.arange(40); ti = (np.arange(40) % 21) + 1
    true = z["tgt_grid"][src, ti]
    rgb16 = np.stack([render(g).reshape(16, 48, 16, 48, 3).mean((1, 3)) for g in true])       # (M,16,16,3)
    noisy = rgb16 + np.random.default_rng(0).normal(0, 10, rgb16.shape)
    pred = grid_from_rgb16(np.moveaxis(noisy, -1, 1))                                            # (M,3,16,16)
    assert grid_metrics(pred, true, z["src_grid"][src])["cell_acc"].mean() > 0.995
    rng = np.random.default_rng(1); proto = rng.normal(size=(8, 32))
    cand_hv = proto[z["tgt_grid"][src].reshape(40, 22, 256)] + 5.0                               # big shared offset
    pred_hv = cand_hv[np.arange(40), ti] + rng.normal(scale=0.3, size=(40, 256, 32))
    idx, cos = nearest_candidate_hv(pred_hv, cand_hv, hv_mean=cand_hv.mean((0, 1, 2)))
    assert (idx == ti).all()
    assert candidate_separation(cand_hv, cand_hv.mean((0, 1, 2))).max() < 0.999
    ans = text_question_answers(z)
    assert (ans["selected_position"] == z["selected"] + 1).all() and (ans["n_callees"] >= 1).all()

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
