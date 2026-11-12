#!/usr/bin/env python3
"""cg_hv_probe.py -- GATE A1: can each adjacency cell be read from its own image soft token?

Run AFTER encoding the rendered grids with the frozen DG vision tower, BEFORE any training.

Inputs (numpy):  --train-hv     one or more (N,256,D) float h_v files of rendered TRAIN grids (.npy)
                 --train-states the state each file encodes, e.g. identity mark_reachable delete_tests
                 --train-pack   gridA_train*.npz
                 --test-hv / --test-states / --test-pack   same for a test pack
The reader must be fitted on states that together show EVERY palette class. Source grids never contain
yellow (marked) cells, so a reader fitted on sources alone cannot read the marking operations.
A ridge classifier (closed form, one-vs-rest on one-hot targets) is fitted per token: h_v[token] -> the
palette class of the cell under that token. It is strictly pointwise (no neighbouring tokens), like the
readout head. For the larger grids of HANDOFF section 12.2 one token covers 2x2 (32 symbols) or 3x3
(48 symbols) cells, and the reader predicts each of those cells from the one token.

PASS (pre-registered): overall test cell accuracy >= 0.99 and every palette class >= 0.95.
If it fails: the vision tower blurs single cells; switch to cg_ops.N_NODES = 8 with 96 px cells
(2x2 tokens per cell) before training anything.

The fitted probe doubles as the CELL READER for evaluation: `--save-model reader.npz` stores it, and
`read_grids(load_model("reader.npz"), pred_hv)` turns any predicted h_v (M,256,D) into (M,16,16) palette
grids for cg_metrics. It is fitted on TRUE vision-tower h_v only, never on model outputs.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_ops import PALETTE_NAMES  # noqa: E402


GRID_TOKENS = 16   # DG image soft tokens form a 16x16 grid for a 768x768 image


def cells_per_token(grid: np.ndarray) -> np.ndarray:
    """(M,n,n) palette grids -> (M*256, b*b) classes of the b x b cells under each soft token (b = n // 16),
    tokens in row-major order, matching h_v.reshape(-1, D)."""
    g = np.asarray(grid)
    m, n, _ = g.shape
    b = n // GRID_TOKENS
    assert b * GRID_TOKENS == n, "grid size must be a multiple of 16"
    return g.reshape(m, GRID_TOKENS, b, GRID_TOKENS, b).transpose(0, 1, 3, 2, 4).reshape(m * GRID_TOKENS ** 2, b * b)


def fit_ridge(X: np.ndarray, y: np.ndarray, n_cls: int, lam: float = 1.0):
    """X (T,D); y (T,) or (T,S): S cells per token. One ridge solve with S*n_cls one-hot outputs."""
    y = y.reshape(len(y), -1)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Z1 = np.concatenate([Z, np.ones((len(Z), 1), Z.dtype)], 1)
    Y = np.eye(n_cls, dtype=np.float64)[y].reshape(len(y), -1)        # (T, S*n_cls)
    A = Z1.T @ Z1 + lam * np.eye(Z1.shape[1])
    W = np.linalg.solve(A, Z1.T @ Y)
    return mu, sd, W


def predict(model, X):
    """-> (T, S) predicted class of each cell under each token."""
    mu, sd, W = model
    n_cls = len(PALETTE_NAMES)
    Z = (X - mu) / sd
    out = np.concatenate([Z, np.ones((len(Z), 1), Z.dtype)], 1) @ W
    return out.reshape(len(X), -1, n_cls).argmax(-1)


def read_grids(model, hv: np.ndarray) -> np.ndarray:
    """Predicted or true h_v (M,256,D) -> (M,n,n) palette grids, one token at a time. The grid size n is
    fixed by the reader: 16 for a one-cell-per-token reader, 32 or 48 for readers fitted on larger grids."""
    hv = np.asarray(hv, np.float64)
    p = predict(model, hv.reshape(-1, hv.shape[-1]))                    # (M*256, b*b)
    b = int(round(p.shape[1] ** 0.5))
    g = p.reshape(len(hv), GRID_TOKENS, GRID_TOKENS, b, b).transpose(0, 1, 3, 2, 4)
    return g.reshape(len(hv), GRID_TOKENS * b, GRID_TOKENS * b).astype(np.uint8)


def save_model(model, path: str):
    np.savez(path, mu=model[0], sd=model[1], W=model[2])


def load_model(path: str):
    z = np.load(path)
    return z["mu"], z["sd"], z["W"]


def run(train_hv, train_grid, test_hv, test_grid, max_train_tokens=400_000, seed=0, return_model=False):
    n_cls = len(PALETTE_NAMES)
    Xtr = np.asarray(train_hv, np.float64).reshape(-1, train_hv.shape[-1])
    ytr = cells_per_token(train_grid)
    if len(Xtr) > max_train_tokens:
        keep = np.random.default_rng(seed).choice(len(Xtr), max_train_tokens, replace=False)
        Xtr, ytr = Xtr[keep], ytr[keep]
    model = fit_ridge(Xtr, ytr, n_cls)
    test_grid = np.asarray(test_grid)
    p = read_grids(model, test_hv)
    yte = test_grid.reshape(-1); pf = p.reshape(-1)
    per = {PALETTE_NAMES[c]: float((pf[yte == c] == c).mean()) for c in range(n_cls) if (yte == c).any()}
    exact = float((p == test_grid).all((1, 2)).mean())
    acc = float((pf == yte).mean())
    res = {"grid_size": int(test_grid.shape[1]), "cells_per_token": int((test_grid.shape[1] // GRID_TOKENS) ** 2),
           "cell_acc": acc, "per_class": per, "exact_grid": exact,
           "PASS": bool(acc >= 0.99 and min(per.values()) >= 0.95)}
    return (res, model) if return_model else res


def self_test():
    rng = np.random.default_rng(0)
    for n in (16, 32, 48):
        b = n // GRID_TOKENS
        proto = rng.normal(size=(b * b, 8, 96))                      # one prototype per (cell position, class)

        def fake(m):
            g = rng.integers(0, 2, size=(m, n, n)); i = np.arange(n); g[:, i, i] = rng.integers(2, 8, size=(m, n))
            c = cells_per_token(g)                                    # (m*256, b*b)
            x = proto[np.arange(b * b)[None], c].sum(1) + rng.normal(scale=0.3, size=(len(c), 96))
            return x.reshape(m, 256, 96).astype(np.float32), g
        xa, ga = fake(80); xb, gb = fake(20)
        r, model = run(xa, ga, xb, gb, return_model=True)
        assert r["PASS"] and r["cells_per_token"] == b * b, r
        assert read_grids(model, xb).shape == gb.shape and (read_grids(model, xb) == gb).mean() > 0.99
        print(f"self-test ok  grid {n}: {r['cells_per_token']} cells/token, cell_acc {r['cell_acc']:.4f}")


def _load_pairs(hv_paths, pack_path, states):
    """Stack several (h_v file, state) pairs: file k holds h_v of `states[k]` for every source of the pack."""
    import cg_sec27_spec as SP
    pack = SP.load_pack(pack_path)
    assert len(hv_paths) == len(states), "give one state name per h_v file"
    hv = np.concatenate([np.load(p) for p in hv_paths])
    grids = np.concatenate([np.stack([SP.state_grid(pack, i, st) for i in range(len(pack["src_grid"]))]) for st in states])
    assert len(hv) == len(grids), "h_v files must cover every source of the pack, in pack order"
    return hv, grids


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-hv", nargs="+"); ap.add_argument("--train-pack")
    ap.add_argument("--test-hv", nargs="+"); ap.add_argument("--test-pack")
    ap.add_argument("--train-states", nargs="+", default=["identity"],
                    help="state encoded in each --train-hv file. The reader must see EVERY palette class: sources "
                         "never contain yellow (marked) cells, so always include mark_reachable.")
    ap.add_argument("--test-states", nargs="+", default=["identity"])
    ap.add_argument("--save-model", help="write the fitted cell reader here (.npz)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test(); sys.exit(0)
    xtr, gtr = _load_pairs(a.train_hv, a.train_pack, a.train_states)
    xte, gte = _load_pairs(a.test_hv, a.test_pack, a.test_states)
    missing = sorted(set(range(len(PALETTE_NAMES))) - set(np.unique(gtr).tolist()))
    if missing:
        print("ERROR: the training grids never show palette classes", [PALETTE_NAMES[c] for c in missing],
              "- add states that contain them (e.g. mark_reachable) to --train-hv/--train-states"); sys.exit(2)
    r, model = run(xtr, gtr, xte, gte, return_model=True)
    if a.save_model:
        save_model(model, a.save_model)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r["PASS"] else 1)
