#!/usr/bin/env python3
"""cg_node_gate.py -- embed the Experiment B texts and run GATE B0 (is this encoder a usable target space?).

Two sub-commands:

  embed  --dataset nodeB_*.json.gz [...] --encoder <spec> --out emb.npz
         Embeds every unique text once. Encoder specs:
           st:<model>   any sentence-transformers model you can run locally (GPU if available)
           hash         a bag-of-character-n-grams MOCK, for pipeline tests only -- never report it
         To use another encoder (e.g. EmbeddingGemma 2 through transformers), add a branch in
         `make_encoder`; it only has to map List[str] -> float32 (n, d).

  gate   --dataset nodeB_test*.json.gz --emb emb.npz [--stats-from emb_train.npz]
         Prints the gate table and exits 0 on PASS, 1 on FAIL.

What the gate measures (all on vectors standardised per dimension with TRAIN statistics):
  spread       effective dimensionality and random-pair cosine. The shipped Kaggle vectors score
               2.4 effective dimensions and 0.76 raw cosine; a usable space needs far more room.
  duplicates   how many of the 256 slots of a source share a vector with another slot.
  visibility   for the edited (primary) slot: distance from source to each edit, and the smallest
               distance between two candidates, both relative to the distance from that symbol to
               its nearest DIFFERENT symbol in the same source ("nn_gap"). A ratio of 1 means an
               edit moves a symbol as far as the gap to its closest neighbour; near 0 means the
               encoder barely notices the edit.
  oracle       nearest-candidate accuracy when the "prediction" is the true target plus Gaussian
               noise of size sigma * nn_gap: how much prediction error the candidate set tolerates.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from typing import Callable, Dict, List

import numpy as np

# ---- PRE-REGISTERED thresholds for GATE B0 ----
MIN_PARTICIPATION = 20.0      # effective dimensions (shipped Kaggle vectors: 2.4)
MAX_CENTERED_COS = 0.20       # mean centred cosine of random pairs
MAX_DUP_RATE = 0.05           # slots sharing a vector with another slot of the same source
MIN_VISIBILITY = 0.10         # median over items of (smallest candidate-candidate distance / nn_gap)
MIN_ORACLE_ACC = 0.90         # nearest-candidate accuracy at noise sigma = 0.25 * nn_gap


def load_dataset(path: str) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def text_key(t: str) -> str:
    return hashlib.sha1(t.encode()).hexdigest()


def make_encoder(spec: str) -> Callable[[List[str]], np.ndarray]:
    if spec == "hash":
        dim = 256

        def enc(texts):
            out = np.zeros((len(texts), dim), np.float32)
            for i, t in enumerate(texts):
                toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\S", t)
                for tok in toks:
                    for g in {tok} | {tok[j:j + 3] for j in range(max(1, len(tok) - 2))}:
                        h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16)
                        out[i, h % dim] += 1.0 if (h >> 9) & 1 else -1.0
            return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-8)
        return enc
    if spec.startswith("st:"):
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(spec[3:], trust_remote_code=False)

        def enc(texts):
            return np.asarray(model.encode(texts, batch_size=32, show_progress_bar=True,
                                           convert_to_numpy=True, normalize_embeddings=False), np.float32)
        return enc
    raise ValueError(f"unknown encoder spec {spec!r}")


def cmd_embed(a):
    texts: Dict[str, str] = {}
    for p in a.dataset:
        for t in load_dataset(p)["texts"]:
            texts.setdefault(text_key(t), t)
    keys = sorted(texts)
    vec = make_encoder(a.encoder)([texts[k] for k in keys])
    np.savez_compressed(a.out, keys=np.array(keys), vec=vec.astype(np.float32), encoder=np.array(a.encoder))
    print(f"embedded {len(keys)} unique texts -> {vec.shape} with {a.encoder} -> {a.out}")


class Emb:
    def __init__(self, path: str):
        z = np.load(path, allow_pickle=False)
        self.encoder = str(z["encoder"])
        self.vec = z["vec"].astype(np.float32)
        self.row = {k: i for i, k in enumerate(z["keys"].tolist())}

    def of(self, texts: List[str]) -> np.ndarray:
        return self.vec[[self.row[text_key(t)] for t in texts]]


def standardiser(train_vec: np.ndarray):
    mu, sd = train_vec.mean(0), train_vec.std(0) + 1e-6
    return lambda x: (x - mu) / sd


def source_vectors(ds: dict, emb: Emb, i: int) -> np.ndarray:
    """(256, d) raw vectors of source i."""
    return emb.of([ds["texts"][t] for t in ds["sources"][i]["slot_text"]])


def target_vectors(ds: dict, emb: Emb, i: int, k: int, src: np.ndarray, null: np.ndarray) -> np.ndarray:
    """(256, d) raw vectors of candidate k of source i: the source with that candidate's changes applied.
    A deleted symbol becomes `null` (use the TRAIN mean vector, i.e. zero after standardisation)."""
    out = src.copy()
    for slot, t in ds["sources"][i]["cands"][k]["changes"]:
        out[slot] = null if t < 0 else emb.of([ds["texts"][t]])[0]
    return out


def primary_candidates(ds: dict, emb: Emb, i: int, z, null: np.ndarray) -> np.ndarray:
    """(K, d) standardised vectors of the PRIMARY slot under each candidate of source i (index 0 = unchanged)."""
    s = ds["sources"][i]
    p = s["primary_slot"]
    out = []
    for c in s["cands"]:
        ch = {slot: t for slot, t in c["changes"]}
        if p not in ch:
            out.append(emb.of([ds["texts"][s["slot_text"][p]]])[0])
        elif ch[p] < 0:
            out.append(null)
        else:
            out.append(emb.of([ds["texts"][ch[p]]])[0])
    return z(np.stack(out))


def spread(v: np.ndarray, rng) -> dict:
    i, j = rng.integers(0, len(v), 20000), rng.integers(0, len(v), 20000)
    keep = i != j
    i, j = i[keep], j[keep]

    def cos(a, b):
        return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
    c = v - v.mean(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(c.T)), 0, None)
    return {"dim": int(v.shape[1]), "raw_cos_mean": float(cos(v[i], v[j]).mean()),
            "centered_cos_mean": float(cos(c[i], c[j]).mean()),
            "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "pc1_share": float(ev.max() / ev.sum())}


def cmd_gate(a):
    ds, emb = load_dataset(a.dataset), Emb(a.emb)
    if ds["meta"].get("format") == "multi":      # multi-symbol packs: gate on the first edited symbol of each source
        for s_ in ds["sources"]:
            s_["primary_slot"], s_["cands"] = s_["sites"][0]["primary_slot"], s_["sites"][0]["cands"]
    stats = Emb(a.stats_from).vec if a.stats_from else emb.vec
    z = standardiser(stats)
    null = stats.mean(0)
    rng = np.random.default_rng(0)
    sp = spread(stats, rng)
    dup, vis, src_move, margins, gaps = [], [], {}, [], []
    oracle = {s: [] for s in (0.1, 0.25, 0.5, 1.0)}
    n_items = len(ds["sources"]) if a.limit is None else min(a.limit, len(ds["sources"]))
    for i in range(n_items):
        s = ds["sources"][i]
        src = z(source_vectors(ds, emb, i))
        p = s["primary_slot"]
        d_all = np.linalg.norm(src - src[p], axis=1)
        d_all[p] = np.inf
        same = d_all < 1e-6
        nn_gap = float(d_all[~same].min())
        gaps.append(nn_gap)
        # duplicate slots
        _, inv, cnt = np.unique(np.round(src, 4), axis=0, return_inverse=True, return_counts=True)
        dup.append(float((cnt[inv.ravel()] > 1).mean()))
        cand = primary_candidates(ds, emb, i, z, null)
        dm = np.linalg.norm(cand[:, None] - cand[None], axis=-1)
        k = len(cand)
        for kk, c in enumerate(s["cands"][1:], 1):
            src_move.setdefault(c["edit"], []).append(dm[0, kk] / nn_gap)
        dm[np.arange(k), np.arange(k)] = np.inf
        margins.append(float(dm.min()))
        vis.append(float(dm.min() / nn_gap))
        for sig in oracle:
            hit = []
            for t in range(k):
                noisy = cand[t] + rng.normal(scale=sig * nn_gap / np.sqrt(cand.shape[1]), size=cand.shape[1])
                hit.append(int(np.linalg.norm(cand - noisy, axis=1).argmin() == t))
            oracle[sig].append(np.mean(hit))
    res = {
        "encoder": emb.encoder, "dataset": os.path.basename(a.dataset), "items": n_items,
        "candidates_per_item": len(ds["sources"][0]["cands"]), "spread": sp,
        "duplicate_slot_rate": float(np.mean(dup)),
        "source_to_edit_distance_over_nn_gap_median": {e: float(np.median(v)) for e, v in src_move.items()},
        "min_candidate_distance_over_nn_gap": {"p05": float(np.percentile(vis, 5)), "median": float(np.median(vis)),
                                               "frac_zero": float(np.mean(np.array(margins) < 1e-6))},
        "oracle_nearest_candidate_acc_at_noise": {str(s): float(np.mean(v)) for s, v in oracle.items()},
    }
    checks = {
        f"participation_ratio >= {MIN_PARTICIPATION}": sp["participation_ratio"] >= MIN_PARTICIPATION,
        f"centered_cos_mean <= {MAX_CENTERED_COS}": sp["centered_cos_mean"] <= MAX_CENTERED_COS,
        f"duplicate_slot_rate <= {MAX_DUP_RATE}": res["duplicate_slot_rate"] <= MAX_DUP_RATE,
        f"median visibility >= {MIN_VISIBILITY}": res["min_candidate_distance_over_nn_gap"]["median"] >= MIN_VISIBILITY,
        f"oracle acc @0.25 >= {MIN_ORACLE_ACC}": res["oracle_nearest_candidate_acc_at_noise"]["0.25"] >= MIN_ORACLE_ACC,
        "no identical candidates": res["min_candidate_distance_over_nn_gap"]["frac_zero"] == 0.0,
    }
    res["checks"] = {k: bool(v) for k, v in checks.items()}
    res["PASS"] = bool(all(checks.values())) and emb.encoder != "hash"
    if emb.encoder == "hash":
        res["note"] = "MOCK encoder: pipeline test only, result is not evidence about any real encoder"
    print(json.dumps(res, indent=1))
    if a.json_out:
        json.dump(res, open(a.json_out, "w"), indent=1)
    return 0 if res["PASS"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed"); e.add_argument("--dataset", nargs="+", required=True)
    e.add_argument("--encoder", required=True); e.add_argument("--out", required=True)
    g = sub.add_parser("gate"); g.add_argument("--dataset", required=True); g.add_argument("--emb", required=True)
    g.add_argument("--stats-from", help="embedding file whose vectors define mean/std (use the TRAIN file)")
    g.add_argument("--limit", type=int); g.add_argument("--json-out")
    a = ap.parse_args(argv)
    sys.exit(cmd_embed(a) or 0 if a.cmd == "embed" else cmd_gate(a))


if __name__ == "__main__":
    main()
