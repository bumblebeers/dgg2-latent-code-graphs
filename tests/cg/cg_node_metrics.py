#!/usr/bin/env python3
"""cg_node_metrics.py -- scoring for Experiment B (symbol vectors), single-symbol and multi-symbol edits.

All vectors are STANDARDISED (per dimension, training statistics) and all distances are Euclidean.

    ev = Evaluator(dataset, emb, stats_vec)          # dataset: nodeB_*.json.gz (single or multi format)
    src  = ev.source(i)                               # (256, d)  model input for source i
    tgt  = ev.target(i, k)                            # (256, d)  training target (k = candidate index;
                                                      #           ignored for multi format)
    w    = ev.loss_weights(i, k)                      # (256,)    changed slots share half the weight
    plan = ev.canvas_labels(i, k)                     # (256,)    0 unchanged, 1 edited, 2 deleted
    res  = ev.score(pred, items)                      # pred (M,256,d); items = [(i, k), ...]

`score` reports, per item and aggregated:
  site_hit            for every edited symbol: nearest of its 8 candidates is the instructed one
  all_sites_hit       every site of the item is right (the whole multi-symbol edit)
  error kind          identity (nothing done) / wrong_arg (right edit type, wrong argument) / other_edit
  top_is_site         the slot that moved most is an edited symbol (localisation)
  changed_recall      share of the truly changed slots among the n slots that moved most (n = how many changed)
  secondary_hit       changed slots that are not a site (callers, class stubs): nearer the edited vector than the source
  drift               mean movement of unchanged slots
"""
from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_node_gate import Emb, standardiser  # noqa: E402

UNCHANGED, EDITED, DELETED = 0, 1, 2


class Evaluator:
    def __init__(self, ds: dict, emb: Emb, stats_vec: np.ndarray = None):
        self.ds, self.emb = ds, emb
        stats = emb.vec if stats_vec is None else stats_vec
        self.z = standardiser(stats)
        self.null_raw = stats.mean(0)
        self.multi = ds["meta"].get("format") == "multi"

    # ---- vectors -----------------------------------------------------------
    def _vec(self, t: int) -> np.ndarray:
        raw = self.null_raw if t < 0 else self.emb.of([self.ds["texts"][t]])[0]
        return self.z(raw[None])[0]

    def source(self, i: int) -> np.ndarray:
        s = self.ds["sources"][i]
        return self.z(self.emb.of([self.ds["texts"][t] for t in s["slot_text"]]))

    def changes(self, i: int, k: int = None) -> List[Tuple[int, int]]:
        s = self.ds["sources"][i]
        return [tuple(c) for c in (s["changes"] if self.multi else s["cands"][k]["changes"])]

    def sites(self, i: int, k: int = None) -> List[dict]:
        """[{"slot", "true", "cands"}]: single format has one site whose true candidate is k."""
        s = self.ds["sources"][i]
        if self.multi:
            return [{"slot": st["primary_slot"], "true": st["true"], "cands": st["cands"]} for st in s["sites"]]
        return [{"slot": s["primary_slot"], "true": k, "cands": s["cands"]}]

    def target(self, i: int, k: int = None) -> np.ndarray:
        out = self.source(i).copy()
        for slot, t in self.changes(i, k):
            out[slot] = self._vec(t)
        return out

    def loss_weights(self, i: int, k: int = None, n_slots: int = 256) -> np.ndarray:
        ch = [slot for slot, _ in self.changes(i, k)]
        w = np.full(n_slots, 1.0 / n_slots)
        if ch:
            w[:] = 0.5 / (n_slots - len(ch))
            w[ch] = 0.5 / len(ch)
        return w

    def canvas_labels(self, i: int, k: int = None, n_slots: int = 256) -> np.ndarray:
        lab = np.zeros(n_slots, np.int8)
        for slot, t in self.changes(i, k):
            lab[slot] = DELETED if t < 0 else EDITED
        return lab

    def site_candidates(self, i: int, site: dict, src: np.ndarray) -> np.ndarray:
        """(K, d) vectors of the site's slot under each of its candidates."""
        out = []
        for c in site["cands"]:
            ch = {slot: t for slot, t in c["changes"]}
            out.append(src[site["slot"]] if site["slot"] not in ch else self._vec(ch[site["slot"]]))
        return np.stack(out)

    # ---- scoring -----------------------------------------------------------
    def score(self, pred: np.ndarray, items: Sequence[Tuple[int, int]]) -> dict:
        rows = []
        kinds = Counter()
        per_edit = defaultdict(lambda: [0, 0])
        for m, (i, k) in enumerate(items):
            src = self.source(i)
            p = np.asarray(pred[m], np.float64)
            sites = self.sites(i, k)
            hits = []
            for st in sites:
                cand = self.site_candidates(i, st, src)
                j = int(np.linalg.norm(cand - p[st["slot"]], axis=1).argmin())
                true_edit = st["cands"][st["true"]]["edit"]
                hit = j == st["true"]
                hits.append(hit)
                per_edit[true_edit][0] += hit
                per_edit[true_edit][1] += 1
                if hit:
                    kinds["correct"] += 1
                elif j == 0:
                    kinds["identity"] += 1
                elif st["cands"][j]["edit"] == true_edit:
                    kinds["wrong_arg"] += 1
                else:
                    kinds["other_edit"] += 1
            ch = dict(self.changes(i, k))
            move = np.linalg.norm(p - src, axis=1)
            row = {"site_hit": float(np.mean(hits)), "all_sites_hit": float(all(hits)), "n_sites": len(sites),
                   "n_changed": len(ch), "top_is_site": np.nan, "changed_recall": np.nan, "secondary_hit": np.nan,
                   "drift": float(np.delete(move, list(ch)).mean()) if len(ch) < len(move) else np.nan}
            if ch:
                order = np.argsort(-move)
                site_slots = {st["slot"] for st in sites}
                row["top_is_site"] = float(int(order[0]) in site_slots)
                row["changed_recall"] = len(set(order[:len(ch)].tolist()) & set(ch)) / len(ch)
                sec = [s for s in ch if s not in site_slots]
                if sec:
                    row["secondary_hit"] = float(np.mean([
                        np.linalg.norm(p[s] - self._vec(ch[s])) < np.linalg.norm(p[s] - src[s]) for s in sec]))
            rows.append(row)
        agg = {k: float(np.nanmean([r[k] for r in rows])) for k in
               ("site_hit", "all_sites_hit", "top_is_site", "changed_recall", "secondary_hit", "drift")}
        total = sum(kinds.values())
        return {"n_items": len(rows), **agg,
                "site_outcomes": {k: kinds[k] / total for k in ("correct", "identity", "wrong_arg", "other_edit")},
                "site_hit_by_edit": {e: v[0] / v[1] for e, v in per_edit.items()},
                "chance_site_hit": 1.0 / len(self.sites(*items[0])[0]["cands"]), "per_item": rows}


def default_items(ds: dict) -> List[Tuple[int, int]]:
    """One instructed edit per source. Single format: cycles through the four real edits (candidates 1..4);
    multi format: the instruction is fixed per source."""
    if ds["meta"].get("format") == "multi":
        return [(i, None) for i in range(len(ds["sources"]))]
    n_real = min(4, len(ds["sources"][0]["cands"]) - 1)
    if ds["sources"][0]["cands"][2]["edit"] == "delete":      # class packs: identity, rename, delete, rename
        return [(i, 1 + (i % 2)) for i in range(len(ds["sources"]))]
    return [(i, 1 + (i % n_real)) for i in range(len(ds["sources"]))]
