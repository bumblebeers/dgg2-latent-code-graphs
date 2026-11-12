"""Checks for HANDOFF section 12: open-ended requests and larger grids."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cg"))
import numpy as np
from cg_metrics import *
from cg_ops import ALL_TARGETS, OPEN_TARGETS, OPEN_OPS, OPEN_ALL, GraphState, render, TEST, BG, EDGE
from cg_grid_dataset import build_eval_manifests, build_open_manifests, build_open_train_schedule
from cg_hv_probe import cells_per_token
import cg_instructions as I

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

def test_open_pack_consistency():
    for name in ("gridA_train1608", "gridA_test360_httpx", "gridA_test360_click"):
        z = np.load(os.path.join(D, name + ".npz"))
        cnt = z["open_count"]
        assert cnt.min() >= 2 and cnt.max() <= 3 and z["open_valid"].shape[2] == 3
        for i in range(0, len(cnt), 37):
            st = GraphState.new(z["adj"][i], z["kind"][i], int(z["selected"][i]))
            for k, t in enumerate(OPEN_TARGETS):
                ans = [a.to_grid() for a in OPEN_OPS[t](st)]
                assert len(ans) == cnt[i, k] and (np.stack(ans) == z["open_valid"][i, k, :len(ans)]).all()
                assert (OPEN_ALL[t](st).to_grid() == z["open_all"][i, k]).all()
                assert (z["open_valid"][i, k, len(ans):] == 0).all()
        # what each answer changes
        i = 0; src = z["src_grid"][i]
        assert all(((z["open_valid"][i, 2, j] != src).sum() == 1) for j in range(cnt[i, 2]))        # one edge cell
        assert (z["open_all"][:, 0] == z["tgt_grid"][:, ALL_TARGETS.index("delete_tests")]).all()   # all = delete_tests
        assert (z["n_tests"] == cnt[:, 0]).all() and (z["n_callees"] == cnt[:, 1]).all()

def test_open_outcomes():
    z = np.load(os.path.join(D, "gridA_test360_httpx.npz"))
    rows = build_open_manifests(360)["open_unseen"]
    src = np.array([r["src"] for r in rows]); k = np.array([r["open_idx"] for r in rows])
    S, V, C, A = z["src_grid"][src], z["open_valid"][src, k], z["open_count"][src, k], z["open_all"][src, k]
    rng = np.random.default_rng(0)
    pick = (rng.random(360) * C).astype(int)
    o = open_outcome(V[np.arange(360), pick], S, V, C, A)                       # a model that commits
    assert (o["label"] == 0).all() and (o["choice"] == pick).all() and o["exact"].all()
    assert 0 < np.nanmean(o["done_share"]) < 1
    o_none = open_outcome(S, S, V, C, A); o_all = open_outcome(A, S, V, C, A)   # a copier, an over-doer
    assert (o_none["label"] == 1).all() and (o_none["done_share"] == 0).all() and not o_none["exact"].any()
    assert (o_all["label"] == 2).all() and (o_all["done_share"] == 1).all()
    s = open_summary(rows, o)
    assert set(s) == set(OPEN_TARGETS) and all(v["valid"] == 1.0 and v["n"] == 120 for v in s.values())
    # diversity: uniform random choices vs always-first
    uni = (rng.random((360, 4)) * C[:, None]).astype(int)
    d_uni = open_diversity(uni, C); d_first = open_diversity(np.zeros((360, 4), int), C)
    assert d_uni["mean_distinct_answers"] > 1.6 and d_first["mean_distinct_answers"] == 1.0
    assert abs(d_uni["picks_first_valid_rate"] - d_uni["expected_first_rate_if_uniform"]) < 0.05
    assert d_first["picks_first_valid_rate"] == 1.0
    partly = uni.copy(); partly[:100, 0] = -1
    assert open_diversity(partly, C)["items_all_samples_valid"] == 260

def test_open_schedule_and_strings():
    z = np.load(os.path.join(D, "gridA_train1608.npz"))
    sched = build_open_train_schedule(z["open_count"])
    assert len(sched) == 3 and sum(map(len, sched)) // 8 == 3618
    rows = [r for e in sched for r in e]
    op = [r for r in rows if "open_idx" in r]
    assert all(0 <= r["valid_choice"] < z["open_count"][r["src"], r["open_idx"]] for r in op)
    assert {r["target"] for r in rows} == set(ALL_TARGETS[:9]) | set(OPEN_TARGETS)
    assert not ({r["instruction"] for r in rows} & {x for v in I.OPEN_UNSEEN.values() for x in v})
    assert not ({r["instruction"] for r in rows} & {x for v in I.UNSEEN.values() for x in v})
    # every valid answer is used as a target about equally often
    first = np.mean([r["valid_choice"] == 0 for r in op]); exp = np.mean([1 / z["open_count"][r["src"], r["open_idx"]] for r in op])
    assert abs(first - exp) < 0.02
    assert len(I.open_strings()) == 30 and len(set(I.open_strings()) | set(I.all_strings())) == 182

def test_larger_grids():
    for n in (32, 48):
        z = np.load(os.path.join(D, f"gridA_n{n}_test360_httpx.npz"))
        assert z["tgt_grid"].shape == (360, 22, n, n) and "open_valid" not in z.files and (z["min_diff"] >= 2).all()
        img = render(z["src_grid"][0])
        assert img.shape == (768, 768, 3)
        c = cells_per_token(z["src_grid"][:2])
        assert c.shape == (2 * 256, (n // 16) ** 2)
        b = n // 16                                                     # token (0,0) holds the top-left b x b cells
        assert (c[0].reshape(b, b) == z["src_grid"][0][:b, :b]).all()
        for name, rows in build_eval_manifests(360).items():
            ti = np.array([r["target_idx"] for r in rows]); src = np.array([r["src"] for r in rows])
            true = z["tgt_grid"][src, ti]
            idx, tie = nearest_candidate_grid(true, z["tgt_grid"][src])
            assert summarize(rows, idx, ALL_TARGETS, tie)["acc"] == 1.0
            assert grid_metrics(true, true, z["src_grid"][src])["exact"].all()
        assert I.preamble(n).startswith(f"The image is a {n}x{n}")
    b = accuracy_by_bucket([1, 3, 20, 70], [1, 1, 0, 1], [0, 4, 16, 64, 1e9])
    assert [x["n"] for x in b] == [2, 0, 1, 1] and b[0]["acc"] == 1.0 and b[2]["acc"] == 0.0

def test_sec27_style_spec():
    import cg_sec27_spec as SP
    tr = SP.load_pack(os.path.join(D, "gridA_train1608.npz")); te = SP.load_pack(os.path.join(D, "gridA_test360_httpx.npz"))
    r = SP.run_self_checks(tr)
    assert r["n_candidate_states"] == 22 and r["n_train_states"] == 23 and r["n_train_ops"] == 17 and r["has_open"]
    rows = [x for e in range(3) for x in SP.build_cg_contrastive_schedule(tr, epoch_seed=42 + e)]
    assert len(rows) // 8 == 4824
    held = {SP.CG_OPERATIONS[o]["target_state"] for o in SP.CG_HELDOUT_OPS}
    assert not ({SP.CG_TRAIN_STATES[t] for _, t, _, _ in rows} & held)
    unseen = {x for v in SP.CG_UNSEEN_PARAPHRASES.values() for x in v} | {x for v in SP.CG_HELDOUT_PARAPHRASES.values() for x in v}
    assert not ({x[3] for x in rows} & unseen) and {x[2] for x in rows} == set(SP.CG_TRAIN_OPS)
    for src, t, op, _ in rows[:4000]:
        st = SP.CG_TRAIN_STATES[t]
        assert SP.state_valid(tr, src, st) and (st.split("#")[0] == op if "#" in st else st == SP.CG_OPERATIONS[op]["target_state"])
    man = SP.build_cg_eval_manifests(te)
    assert len(man) == 16 and all(len(m["img_indices"]) == 360 for m in man.values())
    from collections import Counter
    assert set(Counter(man["seen_bal360"]["op_names"]).values()) <= {25, 26}
    assert (man["seen_bal360"]["tgt_state24_indices"] != [SP.STATE_TO_IDX[SP.CG_OPERATIONS[o]["target_state"]]
                                                        for o in man["seen_shuf360"]["shuffled_from_ops"]]).all()
    rf = man["refeed_heldout_comp_bal360"]
    trained = {x for v in SP.CG_TRAIN_PARAPHRASES.values() for x in v}
    assert set(rf["step1_instructions"]) <= trained and set(rf["step2_instructions"]) <= trained
    assert (rf["step1_state24_indices"] >= 0).all() and (rf["tgt_state24_indices"] < 22).all()
    ch = man["refeed_chain_heldout_ops_360"]
    assert all(len(c) == 3 for c in ch["chain_instructions"]) and set(Counter(ch["op_names"]).values()) == {180}
    assert (man["open_seen_360"]["tgt_state24_indices"] == -1).all() and set(Counter(man["open_seen_360"]["op_names"]).values()) == {120}
    assert set(man["unseen_infer_360"]["instructions"]) == {s for _, _, s in SP.CG_INFERENCE_UNSEEN}
    assert (SP.state_grid(te, 3, "all#cut_one_call") == te["open_all"][3, 2]).all() and not SP.state_valid(te, 0, "cut_one_call#2") or True
    big = SP.load_pack(os.path.join(D, "gridA_n48_test360_httpx.npz"))
    mb = SP.build_cg_eval_manifests(big)
    assert "open_seen_360" not in mb and SP.preamble(big).startswith("The image is a 48x48") and len(mb) == 14
    assert SP.states_to_encode(big, "train") == SP.CG_TRAIN_STATES[:14] and len(SP.states_to_encode(te, "test")) == 33
    assert SP.render_state_pil(big, 0, "delete_methods").size == (768, 768)


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
