#!/usr/bin/env python3
"""cg_grid_dataset.py -- build the Experiment A packs: 16-symbol call-graph grids + all 22 exact targets.

Input : graph JSON files written by cg_extract.py (plain or .gz)
Output: <out>/gridA_<name>.npz      arrays, see `PACK_FIELDS`
        <out>/gridA_<name>.nodes.json   symbol ids behind every source (provenance)

Sampling of one source:
  1. pick a non-test "hub" symbol with >=2 callers and >=2 callees, add 2-5 of each, then fill up to
     16 symbols by random frontier expansion over call edges (so the set is mostly connected and the
     hub has something upstream and downstream; plain random growth rarely yields such a symbol
     because real call graphs are shallow),
  2. put the symbols in a random order (row/column index carries no meaning),
  3. choose the selected symbol: the hub first, else any non-test symbol with a caller and a callee,
  4. accept only if all 22 candidate target grids differ pairwise in >= MIN_DIFF cells
     (so every operation is identifiable from its target: nearest-candidate chance is exactly 1/22).

The same module builds evaluation manifests and the training schedule (pure functions of indices
and a seed; no randomness at run time).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cg_instructions as I  # noqa: E402
from cg_ops import (ALL_TARGETS, CLASS, FUNC, HELDOUT_COMPS, HELDOUT_OPS, METHOD, N_NODES, OPEN_ALL,  # noqa: E402
                    OPEN_MAX_VALID, OPEN_OPS, OPEN_TARGETS, TEST, TRAIN_COMPS, TRAIN_OPS, GraphState, _reach,
                    all_target_grids, comp_name)

MIN_DIFF = 2          # minimum number of differing cells between any two candidate targets of one source
MIN_EDGES, MAX_EDGES = 14, 80   # for 16 symbols; in general (n - 2, 5 * n), see edge_bounds()


def edge_bounds(n: int):
    return n - 2, 5 * n
PACK_FIELDS = {
    "src_grid": "(N,n,n) uint8 palette grid of the source (== tgt_grid[:,0]); n = 16, 32 or 48 symbols",
    "tgt_grid": "(N,22,n,n) uint8 palette grids, candidate order = cg_ops.ALL_TARGETS",
    "adj": "(N,16,16) bool, adj[n,i,j] = symbol i calls symbol j",
    "kind": "(N,16) int8 palette index of each symbol's kind",
    "selected": "(N,) int8 index of the selected symbol",
    "repo_idx": "(N,) int8 index into meta['repos']",
    "min_diff": "(N,) int16 smallest pairwise cell difference among the 22 candidates",
    "open_valid/open_count/open_all": "only in --open packs: (N,3,3,n,n) valid answers zero-padded, (N,3) how many, "
                                      "(N,3,n,n) the did-it-to-all result; axis 1 follows cg_ops.OPEN_TARGETS",
    "n_edges/n_tests/n_callees/n_callers/n_reachable/n_ancestors/reach_depth": "(N,) int16 per-source statistics",
}


def load_graph(path: str) -> dict:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f)


class CallGraph:
    def __init__(self, g: dict, include_loose: bool = True):
        self.repo = g["graph"].get("repo", "")
        self.commit = g["graph"].get("commit", "")
        self.ids = [n["id"] for n in g["nodes"]]
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.kind = np.array([TEST if n["is_test"] else {"class": CLASS, "method": METHOD}.get(n["kind"], FUNC)
                              for n in g["nodes"]], dtype=np.int8)
        self.out: Dict[int, set] = defaultdict(set)
        self.und: Dict[int, set] = defaultdict(set)
        for e in g["edges"]:
            if e["type"] != "calls" or e["source"] == e["target"]:
                continue
            if not include_loose and e.get("how") == "unique_name":
                continue
            a, b = self.index[e["source"]], self.index[e["target"]]
            self.out[a].add(b)
            self.und[a].add(b)
            self.und[b].add(a)
        self.inn: Dict[int, set] = defaultdict(set)
        for a, bs in self.out.items():
            for b in bs:
                self.inn[b].add(a)
        self.hubs = np.array(sorted(n for n in self.und if self.kind[n] != TEST
                                    and len(self.out[n]) >= 2 and len(self.inn[n]) >= 2))
        self.seeds = np.array(sorted(n for n, nb in self.und.items() if len(nb) >= 2))

    def grow(self, rng: np.random.Generator, n: int):
        """Returns (nodes, hub). The hub and a few of its callers/callees come first, then random frontier."""
        hub = int(rng.choice(self.hubs))
        chosen: List[int] = [hub]
        for pool in (sorted(self.out[hub]), sorted(self.inn[hub])):
            k = min(len(pool), int(rng.integers(2, 6)))
            chosen.extend(int(x) for x in rng.choice(pool, size=k, replace=False) if int(x) not in chosen)
        chosen = chosen[:n]
        inset = set(chosen)
        frontier: List[int] = [x for c in chosen for x in sorted(self.und[c]) if x not in inset]
        while len(chosen) < n:
            if not frontier:
                nxt = int(rng.choice(self.seeds))
            else:
                nxt = frontier.pop(int(rng.integers(len(frontier))))
            if nxt in inset:
                continue
            chosen.append(nxt)
            inset.add(nxt)
            frontier.extend(x for x in sorted(self.und[nxt]) if x not in inset)
        return chosen, hub

    def adjacency(self, nodes: List[int]) -> np.ndarray:
        pos = {x: i for i, x in enumerate(nodes)}
        adj = np.zeros((len(nodes), len(nodes)), bool)
        for x in nodes:
            for y in self.out[x]:
                if y in pos:
                    adj[pos[x], pos[y]] = True
        return adj


def min_pairwise_diff(grids: np.ndarray) -> int:
    flat = grids.reshape(len(grids), -1)
    d = (flat[:, None, :] != flat[None, :, :]).sum(-1)
    d[np.arange(len(d)), np.arange(len(d))] = 10 ** 6
    return int(d.min())


def _depth(adj: np.ndarray, start: int) -> int:
    dist = {start: 0}
    q = [start]
    while q:
        x = q.pop(0)
        for y in np.flatnonzero(adj[x]):
            if int(y) not in dist:
                dist[int(y)] = dist[x] + 1
                q.append(int(y))
    return max(dist.values())


def open_grids(st: GraphState):
    """Valid answers of the open-ended operations for one source.
    Returns (valid (3, OPEN_MAX_VALID, n, n) uint8 zero-padded, count (3,), all (3, n, n)) or None if a
    count is outside 2..OPEN_MAX_VALID or an answer collides with the source / the 'all' result."""
    n = st.adj.shape[0]
    valid = np.zeros((len(OPEN_TARGETS), OPEN_MAX_VALID, n, n), np.uint8)
    count = np.zeros(len(OPEN_TARGETS), np.int16)
    allg = np.zeros((len(OPEN_TARGETS), n, n), np.uint8)
    src = st.to_grid()
    for k, name in enumerate(OPEN_TARGETS):
        answers = [a.to_grid() for a in OPEN_OPS[name](st)]
        if not (2 <= len(answers) <= OPEN_MAX_VALID):
            return None
        allg[k] = OPEN_ALL[name](st).to_grid()
        keys = {a.tobytes() for a in answers}
        if len(keys) != len(answers) or src.tobytes() in keys or allg[k].tobytes() in keys:
            return None
        valid[k, :len(answers)] = np.stack(answers)
        count[k] = len(answers)
    return valid, count, allg


def sample_source(cg: CallGraph, rng: np.random.Generator, n: int = N_NODES, want_open: bool = False):
    lo, hi = edge_bounds(n)
    for _ in range(200):
        nodes, hub = cg.grow(rng, n)
        nodes = [nodes[i] for i in rng.permutation(n)]
        adj = cg.adjacency(nodes)
        if not (lo <= adj.sum() <= hi):
            continue
        kind = cg.kind[nodes]
        cands = [i for i in rng.permutation(n) if kind[i] != TEST and adj[i].any() and adj[:, i].any()]
        cands = [nodes.index(hub)] + [i for i in cands if nodes[i] != hub]
        for sel in cands:
            st = GraphState.new(adj, kind, int(sel))
            grids = all_target_grids(st)
            md = min_pairwise_diff(grids)
            if md >= MIN_DIFF:
                if want_open:
                    og = open_grids(st)
                    if og is None:
                        continue
                    return nodes, st, grids, md, og
                return nodes, st, grids, md
    return None


def build_pack(graphs: List[CallGraph], n_total: int, seed: int, n_nodes: int = N_NODES,
               want_open: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    per = [n_total // len(graphs) + (i < n_total % len(graphs)) for i in range(len(graphs))]
    rows, seen = [], set()
    for ri, (cg, quota) in enumerate(zip(graphs, per)):
        got = fails = 0
        while got < quota:
            r = sample_source(cg, rng, n_nodes, want_open)
            if r is None:
                fails += 1
                if fails > 50:
                    raise RuntimeError(f"cannot sample enough sources from {cg.repo}")
                continue
            nodes, st, grids, md = r[:4]
            key = grids[0].tobytes()
            if key in seen:
                continue
            seen.add(key)
            rows.append((ri, nodes, st, grids, md) + tuple(r[4:]))
            got += 1
    order = rng.permutation(len(rows))          # interleave repositories
    rows = [rows[i] for i in order]
    N = len(rows)
    out = {
        "src_grid": np.stack([r[3][0] for r in rows]), "tgt_grid": np.stack([r[3] for r in rows]),
        "adj": np.stack([r[2].adj for r in rows]), "kind": np.stack([r[2].kind for r in rows]),
        "selected": np.array([r[2].selected for r in rows], np.int8),
        "repo_idx": np.array([r[0] for r in rows], np.int8), "min_diff": np.array([r[4] for r in rows], np.int16),
    }
    if want_open:
        out["open_valid"] = np.stack([r[5][0] for r in rows])     # (N,3,OPEN_MAX_VALID,n,n)
        out["open_count"] = np.stack([r[5][1] for r in rows])     # (N,3)
        out["open_all"] = np.stack([r[5][2] for r in rows])       # (N,3,n,n)
    stats = defaultdict(list)
    for _, _, st, *_ in rows:
        a, s = st.adj, st.selected
        stats["n_edges"].append(a.sum())
        stats["n_tests"].append((st.kind == TEST).sum())
        stats["n_callees"].append(a[s].sum())
        stats["n_callers"].append(a[:, s].sum())
        stats["n_reachable"].append(_reach(a, s).sum())
        stats["n_ancestors"].append(_reach(a.T, s).sum())
        stats["reach_depth"].append(_depth(a, s))
    for k, v in stats.items():
        out[k] = np.array(v, np.int16)
    node_ids = [[graphs[r[0]].ids[x] for x in r[1]] for r in rows]
    assert N == n_total and (out["src_grid"] == out["tgt_grid"][:, 0]).all()
    return {"arrays": out, "node_ids": node_ids}


# ----------------------------------------------------------------------------- manifests (index-only)
def text_question_answers(pack: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {"n_callees": pack["n_callees"], "n_callers": pack["n_callers"], "n_reachable": pack["n_reachable"],
            "n_tests": pack["n_tests"], "selected_position": pack["selected"].astype(np.int16) + 1}


def _manifest(n: int, targets: List[str], split: str) -> List[dict]:
    rows = []
    for i in range(n):
        t = targets[i % len(targets)]
        bank = I.bank_for(t, split)
        rows.append({"src": i, "target": t, "target_idx": ALL_TARGETS.index(t),
                     "instruction": bank[(i // len(targets)) % len(bank)]})
    return rows


def build_eval_manifests(n_test: int = 360) -> Dict[str, List[dict]]:
    """Balanced evaluation manifests over a test pack of `n_test` sources (360 = 9*40 = 3*120 = 5*72)."""
    comps_tr = [comp_name(p) for p in TRAIN_COMPS]
    comps_ho = [comp_name(p) for p in HELDOUT_COMPS]
    return {
        "seen_bal": _manifest(n_test, TRAIN_OPS, "seen"),                 # trained ops, trained strings
        "unseen_bal": _manifest(n_test, TRAIN_OPS, "unseen"),             # trained ops, never-seen strings
        "heldout_op": _manifest(n_test, HELDOUT_OPS, "heldout"),          # never-trained operations
        "comp_seen": _manifest(n_test, comps_tr, "seen"),                 # trained two-step pairs, trained strings
        "comp_unseen": _manifest(n_test, comps_tr, "unseen"),             # trained pairs, never-seen strings
        "comp_heldout": _manifest(n_test, comps_ho, "heldout"),           # never-trained pairs
    }


def build_open_manifests(n_test: int = 360) -> Dict[str, List[dict]]:
    """Evaluation manifests for the open-ended operations (360 = 3 * 120). `open_idx` indexes
    cg_ops.OPEN_TARGETS and the first axis of the pack fields open_valid / open_count / open_all."""
    out = {}
    for split, bank in (("open_seen", I.OPEN_SEEN), ("open_unseen", I.OPEN_UNSEEN)):
        rows = []
        for i in range(n_test):
            k = i % len(OPEN_TARGETS)
            t = OPEN_TARGETS[k]
            rows.append({"src": i, "target": t, "open_idx": k,
                         "instruction": bank[t][(i // len(OPEN_TARGETS)) % len(bank[t])]})
        out[split] = rows
    return out


def build_open_train_schedule(open_count: np.ndarray, epochs: int = 3, per_source: int = 6, seed: int = 42):
    """Training order for the OPEN-ENDED run: the 9 trained single operations plus the 3 open-ended ones
    (no compositions). Same shape as build_train_schedule. For an open-ended row, `valid_choice` is the
    index of the ONE valid answer used as the target this time; it is drawn uniformly at random on every
    visit, so over training each valid answer is the target equally often.
    open_count: the pack field (N,3)."""
    assert per_source % 2 == 0
    targets = [("fixed", t) for t in TRAIN_OPS] + [("open", t) for t in OPEN_TARGETS]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(epochs):
        pairs = []
        for i in range(len(open_count)):
            pick = rng.choice(len(targets), size=per_source, replace=False)
            for a, b in zip(pick[0::2], pick[1::2]):
                pair = []
                for t_i in (a, b):
                    fam, t = targets[int(t_i)]
                    if fam == "fixed":
                        bank = I.bank_for(t, "seen")
                        pair.append({"src": i, "target": t, "target_idx": ALL_TARGETS.index(t),
                                     "instruction": bank[int(rng.integers(len(bank)))]})
                    else:
                        k = OPEN_TARGETS.index(t)
                        bank = I.OPEN_SEEN[t]
                        pair.append({"src": i, "target": t, "open_idx": k,
                                     "valid_choice": int(rng.integers(int(open_count[i, k]))),
                                     "instruction": bank[int(rng.integers(len(bank)))]})
                pairs.append(pair)
        order = rng.permutation(len(pairs))
        out.append([row for j in order for row in pairs[j]])
    return out


def per_string_manifest(n_test: int, split: str = "unseen") -> List[dict]:
    """Every string of a split on every test source (Section 27 Part 0.3 style audit). Large: use sparingly."""
    rows = []
    bank = {"unseen": {**I.UNSEEN, **I.COMP_UNSEEN}, "heldout": {**I.HELDOUT_OP_STRINGS, **I.HELDOUT_COMP_STRINGS},
            "seen": {**I.SEEN, **I.COMP_SEEN}}[split]
    for t, strings in bank.items():
        for s in strings:
            rows.extend({"src": i, "target": t, "target_idx": ALL_TARGETS.index(t), "instruction": s}
                        for i in range(n_test))
    return rows


def build_train_schedule(n_src: int, epochs: int = 3, per_source: int = 6, seed: int = 42) -> List[List[dict]]:
    """Training order. Each epoch visits every source `per_source` times with distinct trained targets.
    Consecutive rows (2k, 2k+1) share the SAME source with two DIFFERENT instructions, as in Section 26,
    so a batch of 8 = 4 sources x 2 instructions. With n_src=1608, per_source=6, epochs=3 this gives
    28,944 samples = 3,618 optimiser steps at batch 8 -- the same budget as Section 26 Phase 2."""
    assert per_source % 2 == 0
    targets = TRAIN_OPS + [comp_name(p) for p in TRAIN_COMPS]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(epochs):
        pairs = []
        for i in range(n_src):
            pick = rng.choice(len(targets), size=per_source, replace=False)
            for a, b in zip(pick[0::2], pick[1::2]):
                pair = []
                for t_i in (a, b):
                    t = targets[int(t_i)]
                    bank = I.bank_for(t, "seen")
                    pair.append({"src": i, "target": t, "target_idx": ALL_TARGETS.index(t),
                                 "instruction": bank[int(rng.integers(len(bank)))]})
                pairs.append(pair)
        order = rng.permutation(len(pairs))
        out.append([row for j in order for row in pairs[j]])
    return out


# ----------------------------------------------------------------------------- CLI
def _sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="+", required=True, help="graph JSON(.gz) files from cg_extract.py")
    ap.add_argument("--name", required=True, help="pack name, e.g. train1608 or test360_httpx")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strict-edges", action="store_true", help="drop edges resolved by the unique-name fallback")
    ap.add_argument("--n-nodes", type=int, default=N_NODES, choices=[16, 32, 48], help="symbols per grid")
    ap.add_argument("--open", action="store_true", help="also require and store the open-ended answers")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    graphs = [CallGraph(load_graph(p), include_loose=not a.strict_edges) for p in a.graphs]
    pack = build_pack(graphs, a.n, a.seed, a.n_nodes, a.open)
    npz = os.path.join(a.out, f"gridA_{a.name}.npz")
    np.savez_compressed(npz, **pack["arrays"])
    meta = {"name": a.name, "n": a.n, "seed": a.seed, "repos": [g.repo for g in graphs],
            "commits": [g.commit for g in graphs], "targets": ALL_TARGETS, "min_diff_required": MIN_DIFF,
            "strict_edges": a.strict_edges, "n_nodes": a.n_nodes, "open": a.open, "sha256": _sha(npz),
            "node_ids": pack["node_ids"]}
    with open(os.path.join(a.out, f"gridA_{a.name}.nodes.json"), "w") as f:
        json.dump(meta, f)
    arr = pack["arrays"]
    print(json.dumps({k: (meta[k] if k in meta else None) for k in ("name", "n", "seed", "repos", "sha256")}))
    for k in ("n_edges", "n_tests", "n_callees", "n_callers", "n_reachable", "n_ancestors", "reach_depth", "min_diff"):
        v = arr[k]
        print(f"  {k:12s} min {v.min():3d}  median {int(np.median(v)):3d}  mean {v.mean():6.2f}  max {v.max():3d}")
    print("  per-repo", np.bincount(arr["repo_idx"]).tolist())
    if a.open:
        print("  open valid answers per source (mean):", dict(zip(OPEN_TARGETS, np.round(arr["open_count"].mean(0), 2).tolist())))


if __name__ == "__main__":
    main()
