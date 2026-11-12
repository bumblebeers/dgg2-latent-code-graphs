#!/usr/bin/env python3
"""cg_metrics.py -- graph-native metrics for Experiment A (numpy only).

The existing pipeline already produces, for every prediction, a coarse 16x16 RGB image ("rgb16").
Because one grid cell == one coarse pixel, `cg_ops.grid_from_rgb16` turns that into a predicted
palette grid, and everything here is computed on grids:

  cell_acc            fraction of the 256 cells correct
  exact               all 256 cells correct (the whole edited graph is right)
  edge_f1/prec/rec    call edges only (off-diagonal cells)
  node_acc            the 16 diagonal cells (symbol present? kind? selected? marked?)
  changed_acc         accuracy on cells where target != source   (copying the source scores 0 here)
  unchanged_acc       accuracy on cells where target == source   (copying the source scores 1 here)
  nearest candidate   which of the K candidate targets of the same source the prediction is closest to
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np

BG, EDGE = 0, 1


def grid_metrics(pred: np.ndarray, true: np.ndarray, src: np.ndarray) -> Dict[str, np.ndarray]:
    """pred/true/src: (M,n,n) palette grids, n = 16, 32 or 48. Returns per-item arrays (NaN where undefined)."""
    pred, true, src = (np.asarray(x) for x in (pred, true, src))
    m, n, _ = true.shape
    eye = np.eye(n, dtype=bool)[None]
    ok = pred == true
    off = ~eye
    pe, te = (pred == EDGE) & off, (true == EDGE) & off
    tp = (pe & te).sum((1, 2)).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(pe.sum((1, 2)) > 0, tp / pe.sum((1, 2)), np.where(te.sum((1, 2)) == 0, 1.0, 0.0))
        rec = np.where(te.sum((1, 2)) > 0, tp / te.sum((1, 2)), 1.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
        ch = true != src
        changed = np.where(ch.sum((1, 2)) > 0, (ok & ch).sum((1, 2)) / ch.sum((1, 2)), np.nan)
        unchanged = np.where((~ch).sum((1, 2)) > 0, (ok & ~ch).sum((1, 2)) / (~ch).sum((1, 2)), np.nan)
    return {"cell_acc": ok.mean((1, 2)), "exact": ok.all((1, 2)).astype(float), "edge_prec": prec, "edge_rec": rec,
            "edge_f1": f1, "node_acc": (ok & eye).sum((1, 2)) / n, "changed_acc": changed, "unchanged_acc": unchanged,
            "n_changed": ch.sum((1, 2))}


def nearest_candidate_grid(pred: np.ndarray, cands: np.ndarray):
    """pred (M,16,16), cands (M,K,16,16). Returns (idx, tie): nearest candidate by cell disagreement;
    tie[i] is True when two candidates are equally near (count those as wrong)."""
    d = (pred[:, None] != cands).reshape(len(pred), cands.shape[1], -1).sum(-1)
    idx = d.argmin(1)
    tie = (d == d.min(1, keepdims=True)).sum(1) > 1
    return idx, tie


def _centered_flat(x: np.ndarray, mean: Optional[np.ndarray]) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if mean is not None:
        x = x - np.asarray(mean, dtype=np.float32)
    x = x.reshape(*x.shape[:-2], -1)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def nearest_candidate_hv(pred_hv: np.ndarray, cand_hv: np.ndarray, hv_mean: Optional[np.ndarray] = None):
    """pred_hv (M,256,D), cand_hv (M,K,256,D), hv_mean (D,) or (256,D). Mean-centred cosine, as in Section 26.
    Returns (idx (M,), cos (M,K))."""
    p, c = _centered_flat(pred_hv, hv_mean), _centered_flat(cand_hv, hv_mean)
    cos = np.einsum("md,mkd->mk", p, c)
    return cos.argmax(1), cos


def candidate_separation(cand_hv: np.ndarray, hv_mean: Optional[np.ndarray] = None) -> np.ndarray:
    """Pre-flight near-tie check. cand_hv (M,K,256,D) -> (M,) the largest centred cosine between two
    DIFFERENT candidates of the same source. Section 26 required none above 0.85 (max was 0.559)."""
    c = _centered_flat(cand_hv, hv_mean)
    cos = np.einsum("mkd,mjd->mkj", c, c)
    k = cos.shape[1]
    cos[:, np.arange(k), np.arange(k)] = -1.0
    return cos.max((1, 2))


def summarize(manifest: Sequence[dict], pred_idx: np.ndarray, targets: List[str],
              tie: Optional[np.ndarray] = None) -> dict:
    """Nearest-candidate accuracy overall and per target, the exact prediction breakdown for errors, and,
    for two-step items, how often the output matches only the first or only the second step."""
    k = len(targets)
    per = defaultdict(lambda: {"n": 0, "correct": 0, "pred": Counter()})
    step = {"n": 0, "correct": 0, "first_only": 0, "second_only": 0, "source_copy": 0, "other": 0}
    correct = 0
    for i, row in enumerate(manifest):
        t, p = row["target"], int(pred_idx[i])
        hit = (p == row["target_idx"]) and not (tie is not None and tie[i])
        correct += hit
        per[t]["n"] += 1
        per[t]["correct"] += hit
        per[t]["pred"][targets[p]] += 1
        if "+" in t:
            a, b = t.split("+")
            step["n"] += 1
            if hit:
                step["correct"] += 1
            elif targets[p] == a:
                step["first_only"] += 1
            elif targets[p] == b:
                step["second_only"] += 1
            elif targets[p] == "identity":
                step["source_copy"] += 1
            else:
                step["other"] += 1
    return {"n": len(manifest), "acc": correct / max(1, len(manifest)), "chance": 1.0 / k,
            "per_target": {t: {"n": v["n"], "acc": v["correct"] / v["n"], "pred": dict(v["pred"].most_common())}
                           for t, v in per.items()},
            "two_step": step if step["n"] else None}


# ----------------------------------------------------------------------------- open-ended requests
OPEN_LABELS = ["valid", "none", "all", "ambiguous"]


def open_outcome(pred: np.ndarray, src: np.ndarray, valid: np.ndarray, count: np.ndarray,
                 allg: np.ndarray) -> Dict[str, np.ndarray]:
    """Grade outputs for a request with several valid answers (HANDOFF section 12.1).

    pred, src, allg: (M,n,n); valid: (M,V,n,n) zero-padded; count: (M,) number of real valid answers.
    Each output is assigned to the nearest (by differing cells) of: every valid answer, the unchanged
    source ("none"), and the did-it-to-all result ("all"). A tie between different categories is
    "ambiguous". Returns per item:
      label        index into OPEN_LABELS
      choice       which valid answer (0..count-1, in position order) when label is "valid", else -1
      exact        the output equals a valid answer cell for cell
      done_share   of the cells that differ between source and "all", the share the output changed:
                   0 = nothing done, 1 = done to every eligible element; one valid answer sits in between
    """
    m = len(pred)
    label = np.zeros(m, np.int8); choice = np.full(m, -1, np.int16)
    exact = np.zeros(m, bool); done = np.full(m, np.nan)
    for i in range(m):
        c = int(count[i])
        dv = (pred[i][None] != valid[i, :c]).reshape(c, -1).sum(1)
        dn, da = int((pred[i] != src[i]).sum()), int((pred[i] != allg[i]).sum())
        best = min(int(dv.min()), dn, da)
        cats = [int(dv.min()) == best, dn == best, da == best]
        if sum(cats) > 1 or (cats[0] and (dv == best).sum() > 1):
            label[i] = 3
        else:
            label[i] = cats.index(True)
            if cats[0]:
                choice[i] = int(dv.argmin())
        exact[i] = bool((dv == 0).any())
        elig = src[i] != allg[i]
        if elig.any():
            done[i] = float((pred[i] == allg[i])[elig].mean())
    return {"label": label, "choice": choice, "exact": exact, "done_share": done}


def open_summary(manifest: Sequence[dict], outcome: Dict[str, np.ndarray]) -> dict:
    per = defaultdict(lambda: {"n": 0, **{k: 0 for k in OPEN_LABELS}, "exact": 0, "done_share": []})
    for i, row in enumerate(manifest):
        d = per[row["target"]]
        d["n"] += 1
        d[OPEN_LABELS[int(outcome["label"][i])]] += 1
        d["exact"] += int(outcome["exact"][i])
        d["done_share"].append(float(outcome["done_share"][i]))
    out = {}
    for t, d in per.items():
        out[t] = {"n": d["n"], **{k: d[k] / d["n"] for k in OPEN_LABELS}, "exact_valid": d["exact"] / d["n"],
                  "done_share_mean": float(np.nanmean(d["done_share"]))}
    return out


def open_diversity(choices: np.ndarray, count: np.ndarray) -> dict:
    """choices: (M,S) valid-answer index chosen in each of S independent generations of the same item
    (-1 where the output was not valid); count: (M,). Tells apart "commits to one answer at random"
    from "always picks the same one"."""
    choices, count = np.asarray(choices), np.asarray(count)
    m, s = choices.shape
    allv = (choices >= 0).all(1)
    distinct = np.array([len(set(r.tolist())) for r in choices[allv]]) if allv.any() else np.array([])
    cap = np.minimum(s, count[allv]) if allv.any() else np.array([])
    c, n = choices[choices >= 0], np.repeat(count, s).reshape(m, s)[choices >= 0]
    pos = c / np.maximum(n - 1, 1)
    return {"items_all_samples_valid": int(allv.sum()),
            "mean_distinct_answers": float(distinct.mean()) if len(distinct) else float("nan"),
            "mean_distinct_over_possible": float((distinct / cap).mean()) if len(distinct) else float("nan"),
            "picks_first_valid_rate": float((c == 0).mean()) if len(c) else float("nan"),
            "expected_first_rate_if_uniform": float((1.0 / n).mean()) if len(c) else float("nan"),
            "mean_relative_position": float(pos.mean()) if len(c) else float("nan")}


def accuracy_by_bucket(x: np.ndarray, hit: np.ndarray, edges: Sequence[float]) -> List[dict]:
    """Free scale analysis: accuracy as a function of a per-item size measure, e.g. x = n_changed cells
    (from grid_metrics) or the number of changed slots; edges = bucket boundaries, e.g. [0, 4, 16, 64, 1e9]."""
    x, hit = np.asarray(x, float), np.asarray(hit, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        out.append({"from": lo, "to": hi, "n": int(m.sum()), "acc": float(hit[m].mean()) if m.any() else float("nan")})
    return out
