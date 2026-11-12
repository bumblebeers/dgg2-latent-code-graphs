import ast, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cg"))
import numpy as np
from cg_grid_dataset import load_graph
from cg_node_dataset import Repo, build, parse_def, symbol_text, edit_add_param, edit_rename_def, edit_rename_in_caller
import cg_node_gate as G

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")

def test_edit_primitives():
    t, _ = parse_def("@dec\ndef f(a, b=1):\n    # c\n    return f(a - 1) + g(b)\n")
    assert symbol_text(t) == "@dec\ndef f(a, b=1):\n    return f(a - 1) + g(b)"
    assert "def h(a, b=1)" in symbol_text(edit_rename_def(t, "f", "h")) and "h(a - 1)" in symbol_text(edit_rename_def(t, "f", "h"))
    assert "def f(a, b=1, timeout=None)" in symbol_text(edit_add_param(t, "timeout", "None"))
    assert edit_add_param(t, "b", "2") is None
    t2, _ = parse_def("def k(*args):\n    return self.f(1) or f(2)\n")
    assert "def k(*args, strict=True)" in symbol_text(edit_add_param(t2, "strict", "True"))
    assert symbol_text(edit_rename_in_caller(t2, "f", "h")).count("h(") == 2
    c, _ = parse_def('class A(B):\n    """doc"""\n    x: int = 1\n    def m(self, q):\n        return q * 2\n')
    assert symbol_text(c) == 'class A(B):\n    """doc"""\n    x: int = 1\n\n    def m(self, q):\n        ...'

def test_build_small():
    repo = Repo(load_graph(os.path.join(D, "graphs", "requests.json.gz")))
    ds = build([repo], 8, seed=1, bank="seen", primary_kind="function")
    assert ds["meta"]["n"] == 8
    for s in ds["sources"]:
        assert len(s["slot_text"]) == 256 and len(set(s["slot_node"])) == 256
        p = s["primary_slot"]
        assert s["cands"][0]["edit"] == "identity" and s["cands"][0]["changes"] == []
        prim = []
        for c in s["cands"][1:]:
            ch = dict(map(tuple, c["changes"]))
            assert p in ch
            prim.append(ch[p])
            for slot, t in ch.items():
                if t >= 0:
                    ast.parse(ds["texts"][t])                      # every edited text is valid Python
                    assert ds["texts"][t] != ds["texts"][s["slot_text"][slot]]
            if c["edit"] == "delete":
                assert ch[p] == -1
            if c["edit"] == "rename":
                assert c["args"]["new"] in ds["texts"][ch[p]] and c["args"]["new"] in c["instr_seen"][0]
        assert len(set(prim)) == len(prim)                          # candidates are distinguishable as text
    # deterministic
    assert build([repo], 8, seed=1, bank="seen", primary_kind="function")["sources"][3]["primary_id"] == ds["sources"][3]["primary_id"]

def test_gate_helpers(tmp="/tmp"):
    repo = Repo(load_graph(os.path.join(D, "graphs", "requests.json.gz")))
    ds = build([repo], 4, seed=2, bank="seen", primary_kind="function")
    enc = G.make_encoder("hash")
    keys = sorted({G.text_key(t): t for t in ds["texts"]})
    lut = {G.text_key(t): t for t in ds["texts"]}
    path = os.path.join(os.environ.get("TMPDIR", tmp), "cg_test_emb.npz")
    np.savez(path, keys=np.array(keys), vec=enc([lut[k] for k in keys]), encoder=np.array("hash"))
    emb = G.Emb(path)
    src = G.source_vectors(ds, emb, 0)
    null = emb.vec.mean(0)
    tgt = G.target_vectors(ds, emb, 0, 3, src, null)                # candidate 3 = delete
    p = ds["sources"][0]["primary_slot"]
    changed = {s for s, _ in ds["sources"][0]["cands"][3]["changes"]}
    assert np.allclose(tgt[p], null) and all(np.allclose(tgt[i], src[i]) for i in range(256) if i not in changed)
    assert (G.target_vectors(ds, emb, 0, 0, src, null) == src).all()
    os.remove(path)

if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)


def _emb_for(ds, tmp="/tmp"):
    enc = G.make_encoder("hash")
    lut = {G.text_key(t): t for t in ds["texts"]}
    keys = sorted(lut)
    path = os.path.join(os.environ.get("TMPDIR", tmp), "cg_test_emb2.npz")
    np.savez(path, keys=np.array(keys), vec=enc([lut[k] for k in keys]), encoder=np.array("hash"))
    emb = G.Emb(path); os.remove(path)
    return emb

def test_node_metrics_single_and_multi():
    from cg_node_dataset import build_multi
    from cg_node_metrics import Evaluator, default_items, EDITED, DELETED
    repo = Repo(load_graph(os.path.join(D, "graphs", "requests.json.gz")))
    for ds in (build([repo], 8, seed=3, bank="seen", primary_kind="function"),
               build_multi([repo], 8, seed=4, bank="seen", site_counts=[2, 4])):
        ev = Evaluator(ds, _emb_for(ds))
        items = default_items(ds)
        oracle = np.stack([ev.target(i, k) for i, k in items])
        r = ev.score(oracle, items)
        assert r["site_hit"] == 1.0 and r["all_sites_hit"] == 1.0 and r["top_is_site"] >= 0.0
        assert r["changed_recall"] == 1.0 and r["drift"] == 0.0 and abs(r["chance_site_hit"] - 1 / 8) < 1e-9
        copy = np.stack([ev.source(i) for i, _ in items])
        rc = ev.score(copy, items)
        assert rc["site_hit"] == 0.0 and rc["site_outcomes"]["identity"] == 1.0
        for i, k in items:
            w, lab = ev.loss_weights(i, k), ev.canvas_labels(i, k)
            ch = dict(ev.changes(i, k))
            assert abs(w.sum() - 1) < 1e-9 and abs(w[list(ch)].sum() - 0.5) < 1e-9
            assert set(np.flatnonzero(lab)) == set(ch) and all(lab[s] == (DELETED if t < 0 else EDITED) for s, t in ch.items())
            assert (np.abs(ev.target(i, k) - ev.source(i)).sum(1) > 0).sum() <= len(ch)
    # multi: sites do not overlap, instruction has one sentence per site, wrong-argument errors are detected
    ds = build_multi([repo], 6, seed=5, bank="seen", site_counts=[2])
    ev = Evaluator(ds, _emb_for(ds)); items = default_items(ds)
    for s in ds["sources"]:
        assert s["k"] == 2 == len(s["sites"]) and len({st["primary_slot"] for st in s["sites"]}) == 2
        union = [c[0] for st in s["sites"] for c in st["cands"][st["true"]]["changes"]]
        assert len(union) == len(set(union)) == len(s["changes"])
        assert all(st["primary_id"].split(".")[-1] in s["instr_seen"][0] for st in s["sites"])
    wrong = []
    for i, _ in items:
        t = ev.source(i).copy()
        for st in ev.sites(i):
            alt = {1: 5, 2: 6, 4: 7}.get(st["true"])            # same edit type, other argument
            if alt is not None:
                t[st["slot"]] = ev.site_candidates(i, st, ev.source(i))[alt]
        wrong.append(t)
    rw = ev.score(np.stack(wrong), items)
    assert rw["site_outcomes"]["correct"] == 0.0 and rw["site_outcomes"]["wrong_arg"] > 0.5


if __name__ == "__main__":
    test_node_metrics_single_and_multi(); print("ok test_node_metrics_single_and_multi")
