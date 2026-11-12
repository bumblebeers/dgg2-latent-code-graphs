#!/usr/bin/env python3
"""cg_ops.py -- graph state, the edit operations, and rendering to a 16x16 palette grid / 768x768 image.

A *graph state* is a small directed call graph of N (=16) code symbols:

    adj[i, j]   True  <=>  symbol i CALLS symbol j           (row = caller, column = callee)
    kind[i]     FUNC / METHOD / CLASS / TEST                  (shown on the diagonal)
    present[i]  False once a symbol has been deleted
    selected    index of the one highlighted symbol (the argument of "selected-node" operations)
    marked[i]   True for symbols highlighted by a marking operation

`to_grid()` turns a state into a 16x16 array of palette indices; `render()` upscales each cell to
48x48 px, giving a 768x768 image in which ONE CELL == ONE DG IMAGE SOFT TOKEN (patch 16, 3x3 pool).
Larger graphs (32 or 48 symbols) are drawn into the same 768x768 image, so one soft token then covers
2x2 or 3x3 cells (HANDOFF section 12.2).

Every operation is a pure function state -> state, so two-step compositions are plain function
composition and every target is exact.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Tuple

import numpy as np

N_NODES = 16
CELL_PX = 48
IMG_PX = N_NODES * CELL_PX  # 768

# palette index -> meaning, RGB.  Off-diagonal cells use BG/EDGE only; diagonal cells use the rest.
BG, EDGE, FUNC, METHOD, CLASS, TEST, SEL, MARK = range(8)
PALETTE_NAMES = ["background", "call_edge", "function", "method", "class", "test", "selected", "marked"]
PALETTE_RGB = np.array([
    [245, 245, 245],   # BG       no edge / no symbol
    [20, 30, 90],      # EDGE     row symbol calls column symbol
    [60, 170, 90],     # FUNC     diagonal: plain function
    [40, 170, 190],    # METHOD   diagonal: method
    [140, 80, 190],    # CLASS    diagonal: class
    [240, 150, 40],    # TEST     diagonal: test code
    [220, 40, 50],     # SEL      diagonal: the selected symbol
    [250, 225, 50],    # MARK     diagonal: symbol highlighted by a marking operation
], dtype=np.uint8)


@dataclass(frozen=True)
class GraphState:
    adj: np.ndarray       # (N, N) bool, no self loops
    kind: np.ndarray      # (N,) int8 in {FUNC, METHOD, CLASS, TEST}
    present: np.ndarray   # (N,) bool
    selected: int         # index, or -1 if none / deleted
    marked: np.ndarray    # (N,) bool

    @staticmethod
    def new(adj, kind, selected) -> "GraphState":
        adj = np.array(adj, dtype=bool)
        np.fill_diagonal(adj, False)
        n = adj.shape[0]
        return GraphState(adj, np.array(kind, dtype=np.int8), np.ones(n, bool), int(selected), np.zeros(n, bool))

    def to_grid(self) -> np.ndarray:
        n = self.adj.shape[0]
        live = self.present[:, None] & self.present[None, :]
        grid = np.where(self.adj & live, EDGE, BG).astype(np.uint8)
        diag = np.where(self.present, self.kind, BG).astype(np.uint8)
        diag = np.where(self.present & self.marked, MARK, diag)
        if 0 <= self.selected < n and self.present[self.selected]:
            diag[self.selected] = SEL
        grid[np.arange(n), np.arange(n)] = diag
        return grid


# ----------------------------------------------------------------------------- helpers
def _reach(adj: np.ndarray, start: int) -> np.ndarray:
    """Boolean mask of nodes reachable from `start` by >=1 directed step (start excluded unless on a cycle)."""
    n = adj.shape[0]
    seen = np.zeros(n, bool)
    stack = list(np.flatnonzero(adj[start]))
    while stack:
        x = stack.pop()
        if not seen[x]:
            seen[x] = True
            stack.extend(np.flatnonzero(adj[x] & ~seen))
    seen[start] = False
    return seen


def _live_adj(s: GraphState) -> np.ndarray:
    return s.adj & s.present[:, None] & s.present[None, :]


def _permute(s: GraphState, order: np.ndarray) -> GraphState:
    """New position p holds old node order[p]."""
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    sel = int(inv[s.selected]) if s.selected >= 0 else -1
    return GraphState(s.adj[np.ix_(order, order)], s.kind[order], s.present[order], sel, s.marked[order])


def _keep(s: GraphState, keep: np.ndarray) -> GraphState:
    present = s.present & keep
    return replace(s, present=present, adj=s.adj & present[:, None] & present[None, :])


def _need_sel(s: GraphState):
    if s.selected < 0 or not s.present[s.selected]:
        raise ValueError("operation needs a selected, present node")


# ----------------------------------------------------------------------------- operations
def identity(s: GraphState) -> GraphState:
    return s


def reverse_edges(s: GraphState) -> GraphState:
    """Every call i->j becomes j->i (matrix transpose). Content-independent."""
    return replace(s, adj=s.adj.T.copy())


def reverse_order(s: GraphState) -> GraphState:
    """Relabel symbols in reverse order (same graph, rows and columns both reversed). Content-independent."""
    return _permute(s, np.arange(s.adj.shape[0])[::-1].copy())


def delete_selected(s: GraphState) -> GraphState:
    """Remove the selected symbol and every edge touching it."""
    _need_sel(s)
    keep = np.ones(s.adj.shape[0], bool)
    keep[s.selected] = False
    return replace(_keep(s, keep), selected=-1)


def isolate_selected(s: GraphState) -> GraphState:
    """Keep the selected symbol but remove every edge into or out of it."""
    _need_sel(s)
    adj = s.adj.copy()
    adj[s.selected, :] = False
    adj[:, s.selected] = False
    return replace(s, adj=adj)


def keep_callees(s: GraphState) -> GraphState:
    """Keep only the selected symbol and the symbols it calls directly (induced subgraph)."""
    _need_sel(s)
    keep = _live_adj(s)[s.selected].copy()
    keep[s.selected] = True
    return _keep(s, keep)


def keep_callers(s: GraphState) -> GraphState:
    """Keep only the selected symbol and the symbols that call it directly (induced subgraph)."""
    _need_sel(s)
    keep = _live_adj(s)[:, s.selected].copy()
    keep[s.selected] = True
    return _keep(s, keep)


def mark_reachable(s: GraphState) -> GraphState:
    """Mark every symbol the selected one can reach through one or more calls (its descendants)."""
    _need_sel(s)
    return replace(s, marked=s.marked | _reach(_live_adj(s), s.selected))


def mark_ancestors(s: GraphState) -> GraphState:
    """Mark every symbol that can reach the selected one through one or more calls (its ancestors)."""
    _need_sel(s)
    return replace(s, marked=s.marked | _reach(_live_adj(s).T, s.selected))


def _delete_kind(s: GraphState, k: int) -> GraphState:
    keep = s.kind != k
    if s.selected >= 0:
        keep[s.selected] = True          # the selected symbol is drawn red, so its kind is not visible
    return _keep(s, keep)


def delete_tests(s: GraphState) -> GraphState:
    """Remove every test symbol (orange diagonal) and its edges."""
    return _delete_kind(s, TEST)


def delete_classes(s: GraphState) -> GraphState:
    """Remove every class symbol (purple diagonal) and its edges."""
    return _delete_kind(s, CLASS)


def delete_methods(s: GraphState) -> GraphState:
    """Remove every method symbol (teal diagonal) and its edges."""
    return _delete_kind(s, METHOD)


def delete_functions(s: GraphState) -> GraphState:
    """Remove every plain function symbol (green diagonal) and its edges."""
    return _delete_kind(s, FUNC)


def selected_first(s: GraphState) -> GraphState:
    """Relabel so the selected symbol comes first; all others keep their relative order."""
    _need_sel(s)
    n = s.adj.shape[0]
    order = np.array([s.selected] + [i for i in range(n) if i != s.selected])
    return _permute(s, order)


OPS: Dict[str, Callable[[GraphState], GraphState]] = {
    "identity": identity, "reverse_edges": reverse_edges, "reverse_order": reverse_order,
    "delete_selected": delete_selected, "isolate_selected": isolate_selected,
    "keep_callees": keep_callees, "keep_callers": keep_callers,
    "mark_reachable": mark_reachable, "mark_ancestors": mark_ancestors,
    "delete_tests": delete_tests, "selected_first": selected_first,
    "delete_classes": delete_classes, "delete_methods": delete_methods, "delete_functions": delete_functions,
}

# What each operation needs from the model (used for per-family reporting).
OP_FAMILY: Dict[str, str] = {
    "identity": "fixed", "reverse_edges": "fixed", "reverse_order": "fixed",       # same patch permutation for every image
    "delete_selected": "selected", "isolate_selected": "selected",                  # must find the red cell
    "selected_first": "selected",                                                   # permutation that depends on the image
    "keep_callees": "neighbourhood", "keep_callers": "neighbourhood",              # must read one row / one column
    "mark_reachable": "multi_hop", "mark_ancestors": "multi_hop",                  # must follow chains of edges
    "delete_tests": "attribute", "delete_classes": "attribute",                    # must read node kinds
    "delete_methods": "attribute", "delete_functions": "attribute",
}

# ---- PRE-REGISTERED SPLIT (do not edit after the first training launch) ---------------------
TRAIN_OPS: List[str] = ["identity", "reverse_edges", "reverse_order", "delete_selected",
                        "keep_callees", "mark_reachable", "delete_tests", "selected_first", "delete_classes"]
# `isolate_selected` is implemented but NOT used: its target differs from `delete_selected` in exactly one
# cell (the diagonal), which would make the pair a built-in near-tie for the nearest-candidate metric.
#
# Held-out operations come in the two kinds that Section 27 Part 2 separated:
#   mirror of a trained operation   keep_callers, mark_ancestors   (like `darken` after training `brighten`: 8.2%)
#   new member of a trained family  delete_methods                 (like `blue_tint` after red + green tint: 60.2%)
HELDOUT_OPS: List[str] = ["keep_callers", "mark_ancestors", "delete_methods"]
TRAIN_COMPS: List[Tuple[str, str]] = [
    ("reverse_edges", "reverse_order"),     # fixed x fixed
    ("reverse_edges", "delete_selected"),
    ("reverse_order", "keep_callees"),
    ("reverse_order", "mark_reachable"),
    ("reverse_edges", "delete_tests"),
]
HELDOUT_COMPS: List[Tuple[str, str]] = [
    ("reverse_order", "delete_selected"),
    ("reverse_edges", "keep_callees"),
    ("reverse_edges", "mark_reachable"),
    ("reverse_order", "delete_tests"),
    ("delete_tests", "mark_reachable"),     # exploratory: two content-dependent steps
]


def comp_name(pair: Tuple[str, str]) -> str:
    return f"{pair[0]}+{pair[1]}"


# Fixed candidate order used everywhere (index 0 = identity = the source itself).
ALL_TARGETS: List[str] = (TRAIN_OPS + HELDOUT_OPS + [comp_name(p) for p in TRAIN_COMPS]
                          + [comp_name(p) for p in HELDOUT_COMPS])
TARGET_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(ALL_TARGETS)}
TARGET_SPLIT: Dict[str, str] = {**{t: "train_op" for t in TRAIN_OPS}, **{t: "heldout_op" for t in HELDOUT_OPS},
                                **{comp_name(p): "train_comp" for p in TRAIN_COMPS},
                                **{comp_name(p): "heldout_comp" for p in HELDOUT_COMPS}}


def apply(s: GraphState, target: str) -> GraphState:
    """Apply a single operation ("keep_callees") or a two-step composition ("reverse_edges+keep_callees")."""
    for step in target.split("+"):
        s = OPS[step](s)
    return s


def all_target_grids(s: GraphState) -> np.ndarray:
    """(K, N, N) uint8 palette grids for every entry of ALL_TARGETS."""
    return np.stack([apply(s, t).to_grid() for t in ALL_TARGETS])


# ----------------------------------------------------------------------------- open-ended operations
# Requests with SEVERAL valid answers (HANDOFF section 12.1). Each function returns the list of every
# valid result, in ascending position order of the element that was removed. OPEN_ALL gives the
# "did it to all of them" result, which is what a blurred or over-eager model drifts toward.
def open_delete_one_test(s: GraphState) -> List[GraphState]:
    """Delete exactly one test symbol (any one)."""
    out = []
    for i in np.flatnonzero(s.present & (s.kind == TEST)):
        keep = np.ones(s.adj.shape[0], bool)
        keep[i] = False
        out.append(_keep(s, keep))
    return out


def open_delete_one_callee(s: GraphState) -> List[GraphState]:
    """Delete exactly one of the symbols the selected symbol calls directly (any one)."""
    _need_sel(s)
    out = []
    for j in np.flatnonzero(_live_adj(s)[s.selected]):
        keep = np.ones(s.adj.shape[0], bool)
        keep[j] = False
        out.append(_keep(s, keep))
    return out


def open_cut_one_call(s: GraphState) -> List[GraphState]:
    """Remove exactly one call made by the selected symbol (any one); both symbols stay."""
    _need_sel(s)
    out = []
    for j in np.flatnonzero(_live_adj(s)[s.selected]):
        adj = s.adj.copy()
        adj[s.selected, j] = False
        out.append(replace(s, adj=adj))
    return out


def _delete_all_callees(s: GraphState) -> GraphState:
    _need_sel(s)
    return _keep(s, ~_live_adj(s)[s.selected])


def _cut_all_calls(s: GraphState) -> GraphState:
    _need_sel(s)
    adj = s.adj.copy()
    adj[s.selected, :] = False
    return replace(s, adj=adj)


OPEN_OPS: Dict[str, Callable[[GraphState], List[GraphState]]] = {
    "delete_one_test": open_delete_one_test, "delete_one_callee": open_delete_one_callee,
    "cut_one_call": open_cut_one_call,
}
OPEN_ALL: Dict[str, Callable[[GraphState], GraphState]] = {
    "delete_one_test": delete_tests, "delete_one_callee": _delete_all_callees, "cut_one_call": _cut_all_calls,
}
OPEN_TARGETS: List[str] = list(OPEN_OPS)
OPEN_MAX_VALID = 3      # sources carry 2..3 valid answers per open-ended operation (keeps the state cache small)


# ----------------------------------------------------------------------------- rendering
def render(grid: np.ndarray, cell_px: int = None) -> np.ndarray:
    """(N, N) palette grid -> (768, 768, 3) uint8 RGB, flat colour per cell (NEAREST upscale).
    cell_px defaults to 768 // N: 48 px for 16 symbols, 24 px for 32, 16 px for 48."""
    grid = np.asarray(grid)
    if cell_px is None:
        assert IMG_PX % grid.shape[0] == 0, "grid size must divide 768"
        cell_px = IMG_PX // grid.shape[0]
    rgb = PALETTE_RGB[grid.astype(np.int64)]
    return np.repeat(np.repeat(rgb, cell_px, axis=0), cell_px, axis=1)


def render_pil(grid: np.ndarray, cell_px: int = None):
    from PIL import Image
    return Image.fromarray(render(grid, cell_px), mode="RGB")


def grid_from_rgb16(rgb16: np.ndarray) -> np.ndarray:
    """Inverse of `render` at coarse resolution: (..., 3, 16, 16) or (..., 16, 16, 3) RGB (0-255 or 0-1)
    -> (..., 16, 16) palette indices by nearest palette colour."""
    a = np.asarray(rgb16, dtype=np.float32)
    if a.shape[-1] != 3:
        a = np.moveaxis(a, -3, -1)
    if a.max() <= 1.5:
        a = a * 255.0
    d = ((a[..., None, :] - PALETTE_RGB.astype(np.float32)) ** 2).sum(-1)
    return d.argmin(-1).astype(np.uint8)
