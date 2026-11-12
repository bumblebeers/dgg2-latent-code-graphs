#!/usr/bin/env python3
"""cg_node_dataset.py -- build the Experiment B dataset: 256-symbol code states + exact edited states, as TEXT.

Experiment B puts one code symbol in each of the 256 slots (instead of one image patch). A frozen text
encoder that YOU can run turns every symbol's source text into one vector; the model sees the 256
source vectors plus an instruction ("Rename the function `send` to `send_impl`.") and must emit the 256
vectors of the edited codebase. This script produces everything up to (not including) the encoder:

  texts      table of unique symbol texts (source and edited), to be embedded once
  sources    per source: 256 slots -> text index, the primary (edited) slot, candidates
  candidates per source: identity + edits + wrong-argument distractors, each with the exact list of
             slots whose text changes (an edit to one symbol also changes its callers / its class)

Symbol text: functions/methods = their source, normalised by ast.unparse (comments and formatting
removed, so two texts differ only by the edit). Classes = a stub: header, docstring, class-level
assignments and method signatures (bodies replaced by `...`), so a class vector does not duplicate
its methods but does change when a method is renamed, added to, or deleted.

Edits (all exact, applied to the AST):
  rename         def name, recursive self-calls, the class stub, and every call site in callers
  add_param      new keyword parameter with a default (signature in the symbol and in its class stub)
  delete         the slot becomes empty (None); the class stub loses the method
  add_decorator  decorator added to the symbol (and to its signature in the class stub)
Held-out edit:  rename applied to a CLASS (--primary-kind class): stub, call sites, subclasses' bases.
Multi-symbol:   --sites 1 2 builds sources whose single instruction edits several symbols (build_multi).
"""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_grid_dataset import load_graph  # noqa: E402

N_SLOTS = 256
MAX_LINES = 80

# ---- PRE-REGISTERED argument banks: "seen" is used in training, "unseen" only in evaluation -----
RENAME = {"seen": ["{n}_impl", "{n}_internal", "do_{n}", "{n}_v2", "{n}_legacy", "safe_{n}", "{n}_helper", "base_{n}"],
          "unseen": ["{n}_core", "run_{n}", "{n}_ex", "old_{n}"]}
PARAMS = {"seen": [("timeout", "None"), ("verbose", "False"), ("strict", "True"), ("retries", "3"),
                   ("logger", "None"), ("dry_run", "False"), ("encoding", "'utf-8'"), ("limit", "100")],
          "unseen": [("cache", "None"), ("debug", "False"), ("max_depth", "5"), ("callback", "None")]}
DECOS = {"seen": ["deprecated", "functools.lru_cache(maxsize=None)", "trace_calls", "retry(3)", "profile", "synchronized"],
         "unseen": ["timed", "validate_args", "memoize"]}

TEMPLATES = {
    "identity": {"seen": ["Leave the codebase unchanged.", "Make no edits to any symbol.",
                          "Reproduce every symbol exactly as it is.", "Keep the code as given."],
                 "unseen": ["Return the codebase untouched.", "Apply no change at all."]},
    "rename": {"seen": ["Rename the {kind} `{sym}` to `{new}`.", "Change the name of `{sym}` to `{new}` everywhere it is used.",
                        "`{sym}` should now be called `{new}`; update its callers.", "Rename `{sym}` to `{new}` and fix every call site."],
               "unseen": ["Give `{sym}` the new name `{new}` throughout the code.", "Replace the identifier `{sym}` with `{new}`, including where it is called."]},
    "add_param": {"seen": ["Add a keyword parameter `{p}` with default `{d}` to `{sym}`.", "Give `{sym}` a new optional argument `{p}={d}`.",
                           "Extend the signature of `{sym}` with `{p}={d}`.", "`{sym}` should accept an extra parameter `{p}` defaulting to `{d}`."],
                  "unseen": ["Let callers pass `{p}` to `{sym}`; when omitted it is `{d}`.", "Append `{p}={d}` to the parameters of `{sym}`."]},
    "delete": {"seen": ["Delete the {kind} `{sym}`.", "Remove `{sym}` from the codebase.",
                        "Drop the definition of `{sym}`.", "`{sym}` is dead code; take it out."],
               "unseen": ["Get rid of `{sym}` entirely.", "Erase the definition of `{sym}`."]},
    "add_decorator": {"seen": ["Decorate `{sym}` with `@{dec}`.", "Add the decorator `@{dec}` to `{sym}`.",
                               "Apply `@{dec}` to the {kind} `{sym}`.", "Put `@{dec}` on top of `{sym}`."],
                      "unseen": ["Wrap `{sym}` using the `@{dec}` decorator.", "`{sym}` should be marked with `@{dec}`."]},
}


# ----------------------------------------------------------------------------- text + edits
def parse_def(text: str):
    tree = ast.parse(text)
    node = tree.body[0]
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        raise ValueError("not a definition")
    return tree, node


def _stub_class(cls: ast.ClassDef) -> ast.ClassDef:
    c = copy.deepcopy(cls)
    body = []
    for i, st in enumerate(c.body):
        if i == 0 and isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant) \
                and isinstance(st.value.value, str):
            body.append(st)
        elif isinstance(st, (ast.Assign, ast.AnnAssign)):
            body.append(st)
        elif isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            st.body = [ast.Expr(ast.Constant(Ellipsis))]
            body.append(st)
        elif isinstance(st, ast.ClassDef):
            st.body = [ast.Expr(ast.Constant(Ellipsis))]
            st.decorator_list = []
            body.append(st)
    c.body = body or [ast.Pass()]
    return c


def symbol_text(tree: ast.Module) -> str:
    """Canonical text that gets embedded."""
    node = tree.body[0]
    if isinstance(node, ast.ClassDef):
        node = _stub_class(node)
    return ast.unparse(ast.fix_missing_locations(ast.Module([node], [])))


class _RenameCalls(ast.NodeTransformer):
    def __init__(self, old, new, names=True, attrs=True, everywhere=False):
        self.old, self.new, self.names, self.attrs, self.everywhere = old, new, names, attrs, everywhere

    def visit_Call(self, node):
        self.generic_visit(node)
        f = node.func
        if self.names and isinstance(f, ast.Name) and f.id == self.old:
            f.id = self.new
        elif self.attrs and isinstance(f, ast.Attribute) and f.attr == self.old:
            f.attr = self.new
        return node

    def visit_Name(self, node):
        if self.everywhere and node.id == self.old:
            node.id = self.new
        return node


def edit_rename_def(tree, old, new):
    t = copy.deepcopy(tree)
    t.body[0].name = new
    return _RenameCalls(old, new).visit(t)


def edit_rename_in_caller(tree, old, new, is_class_target=False):
    t = copy.deepcopy(tree)
    return _RenameCalls(old, new, attrs=not is_class_target, everywhere=is_class_target).visit(t)


def edit_rename_member(tree, old, new):
    """Class tree: rename the method `old` (signature in the stub)."""
    t = copy.deepcopy(tree)
    for st in t.body[0].body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == old:
            st.name = new
    return t


def _add_param(fn, p, d):
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs} | {x.arg for x in (a.vararg, a.kwarg) if x}
    if p in names:
        return False
    default = ast.parse(d, mode="eval").body
    if a.vararg or a.kwonlyargs:
        a.kwonlyargs.append(ast.arg(p)); a.kw_defaults.append(default)
    else:
        a.args.append(ast.arg(p)); a.defaults.append(default)
    return True


def edit_add_param(tree, p, d, member: Optional[str] = None):
    t = copy.deepcopy(tree)
    node = t.body[0]
    targets = [node] if member is None else [s for s in node.body
                                             if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == member]
    ok = [_add_param(fn, p, d) for fn in targets]
    return t if ok and all(ok) else None


def edit_add_decorator(tree, dec, member: Optional[str] = None):
    t = copy.deepcopy(tree)
    node = t.body[0]
    targets = [node] if member is None else [s for s in node.body
                                             if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == member]
    for fn in targets:
        fn.decorator_list.insert(0, ast.parse(dec, mode="eval").body)
    return t


def edit_delete_member(tree, member):
    t = copy.deepcopy(tree)
    node = t.body[0]
    node.body = [s for s in node.body if not (isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and s.name == member)] \
        or [ast.Pass()]
    return t


def edit_rename_base(tree, old, new):
    t = copy.deepcopy(tree)
    for b in ast.walk(ast.Module(t.body[0].bases, [])):
        if isinstance(b, ast.Name) and b.id == old:
            b.id = new
    return t


# ----------------------------------------------------------------------------- repository view
class Repo:
    def __init__(self, g: dict):
        self.name = g["graph"].get("repo", "")
        self.commit = g["graph"].get("commit", "")
        self.nodes = g["nodes"]
        self.idx = {n["id"]: i for i, n in enumerate(self.nodes)}
        self.trees: Dict[int, ast.Module] = {}
        self.text: Dict[int, str] = {}
        for i, n in enumerate(self.nodes):
            try:
                tree, _ = parse_def(n["text"])
                self.trees[i] = tree
                self.text[i] = symbol_text(tree)
            except (SyntaxError, ValueError, IndexError, RecursionError):
                pass
        self.out, self.inn, self.und = defaultdict(set), defaultdict(set), defaultdict(set)
        self.subclasses = defaultdict(set)
        for e in g["edges"]:
            a, b = self.idx[e["source"]], self.idx[e["target"]]
            if a == b or a not in self.trees or b not in self.trees:
                continue
            if e["type"] == "calls":
                self.out[a].add(b); self.inn[b].add(a)
            elif e["type"] == "inherits":
                self.subclasses[b].add(a)
            self.und[a].add(b); self.und[b].add(a)
        self.parent = {i: self.idx.get(n["parent"]) for i, n in enumerate(self.nodes) if n.get("parent") in self.idx}

    def primaries(self, kind: str) -> List[int]:
        out = []
        for i, n in enumerate(self.nodes):
            if i not in self.trees or n["is_test"] or n["name"].startswith("__") or n["text"].count("\n") > MAX_LINES:
                continue
            callers = [c for c in self.inn[i] if self.nodes[c]["kind"] != "class"]
            if kind == "class" and n["kind"] == "class" and callers:
                out.append(i)
            elif kind == "function" and n["kind"] in ("function", "method") and callers:
                out.append(i)
        return out

    def state(self, primary: int, rng: np.random.Generator, n: int = N_SLOTS) -> List[int]:
        chosen = [primary]
        for pool in (sorted(self.inn[primary]), sorted(self.out[primary]), sorted(self.subclasses[primary])):
            pool = [x for x in pool if x not in chosen]
            chosen.extend(int(x) for x in rng.permutation(pool)[:12])
        if primary in self.parent and self.parent[primary] in self.trees and self.parent[primary] not in chosen:
            chosen.append(self.parent[primary])
        inset = set(chosen)
        frontier = [x for c in chosen for x in sorted(self.und[c]) if x not in inset]
        pool_all = sorted(self.trees)
        while len(chosen) < n:
            nxt = frontier.pop(int(rng.integers(len(frontier)))) if frontier else int(rng.choice(pool_all))
            if nxt in inset:
                continue
            chosen.append(nxt); inset.add(nxt)
            frontier.extend(x for x in sorted(self.und[nxt]) if x not in inset)
        return [chosen[i] for i in rng.permutation(n)]

    def short_name(self, i: int, state: List[int]) -> str:
        parts = self.nodes[i]["id"].split(".")
        others = [self.nodes[j]["id"] for j in state if j != i]
        for k in range(1, len(parts) + 1):
            suf = ".".join(parts[-k:])
            if not any(o == suf or o.endswith("." + suf) for o in others):
                return suf
        return self.nodes[i]["id"]

    # returns {node index: new text or None} for the slots that change
    def apply(self, edit: str, args: dict, primary: int, state: List[int]) -> Optional[Dict[int, Optional[str]]]:
        n = self.nodes[primary]
        inst = set(state)
        par = self.parent.get(primary)
        par = par if (par in inst and par in self.trees and self.nodes[par]["kind"] == "class") else None
        ch: Dict[int, Optional[str]] = {}
        if edit == "identity":
            return {}
        if edit == "rename":
            old, new = n["name"], args["new"]
            is_cls = n["kind"] == "class"
            ch[primary] = symbol_text(edit_rename_def(self.trees[primary], old, new))
            if par is not None:
                ch[par] = symbol_text(edit_rename_member(self.trees[par], old, new))
            for c in self.inn[primary]:
                if c in inst and c != primary and self.nodes[c]["kind"] != "class":
                    ch[c] = symbol_text(edit_rename_in_caller(self.trees[c], old, new, is_cls))
            for c in self.subclasses[primary] if is_cls else ():
                if c in inst:
                    ch[c] = symbol_text(edit_rename_base(self.trees[c], old, new))
        elif edit == "add_param":
            t = edit_add_param(self.trees[primary], args["p"], args["d"])
            if t is None:
                return None
            ch[primary] = symbol_text(t)
            if par is not None:
                tp = edit_add_param(self.trees[par], args["p"], args["d"], member=n["name"])
                if tp is not None:
                    ch[par] = symbol_text(tp)
        elif edit == "delete":
            ch[primary] = None
            if par is not None:
                ch[par] = symbol_text(edit_delete_member(self.trees[par], n["name"]))
        elif edit == "add_decorator":
            ch[primary] = symbol_text(edit_add_decorator(self.trees[primary], args["dec"]))
            if par is not None:
                ch[par] = symbol_text(edit_add_decorator(self.trees[par], args["dec"], member=n["name"]))
        else:
            raise KeyError(edit)
        ch = {k: v for k, v in ch.items() if v is None or v != self.text[k]}
        return ch if primary in ch else None


def candidates_for(repo: Repo, primary: int, bank: str, rng: np.random.Generator) -> List[Tuple[str, dict]]:
    name = repo.nodes[primary]["name"].strip("_") or "fn"
    r = [t.format(n=name) for t in rng.permutation(RENAME[bank])[:2]]
    if repo.nodes[primary]["kind"] == "class":
        r = [x.title().replace("_", "") for x in r]
        return [("identity", {}), ("rename", {"new": r[0]}), ("delete", {}), ("rename", {"new": r[1]})]
    p = [PARAMS[bank][i] for i in rng.permutation(len(PARAMS[bank]))[:2]]
    d = [DECOS[bank][i] for i in rng.permutation(len(DECOS[bank]))[:2]]
    return [("identity", {}), ("rename", {"new": r[0]}), ("add_param", {"p": p[0][0], "d": p[0][1]}), ("delete", {}),
            ("add_decorator", {"dec": d[0]}),
            # wrong-argument distractors: same edit type, different argument
            ("rename", {"new": r[1]}), ("add_param", {"p": p[1][0], "d": p[1][1]}), ("add_decorator", {"dec": d[1]})]


def build(repos: List[Repo], n_total: int, seed: int, bank: str, primary_kind: str) -> dict:
    rng = np.random.default_rng(seed)
    texts: List[str] = []
    tix: Dict[str, int] = {}

    def T(s: str) -> int:
        if s not in tix:
            tix[s] = len(texts); texts.append(s)
        return tix[s]

    per = [n_total // len(repos) + (i < n_total % len(repos)) for i in range(len(repos))]
    sources = []
    for ri, (repo, quota) in enumerate(zip(repos, per)):
        prim = repo.primaries(primary_kind)
        if not prim:
            raise RuntimeError(f"no usable primary symbols in {repo.name}")
        order = [prim[i] for i in rng.permutation(len(prim))]
        got = j = 0
        while got < quota:
            primary = order[j % len(order)]; j += 1
            if j > 20 * quota + 20 * len(order):
                raise RuntimeError(f"cannot build enough sources from {repo.name}")
            state = repo.state(primary, rng)
            cands = []
            for edit, args in candidates_for(repo, primary, bank, rng):
                ch = repo.apply(edit, args, primary, state) if edit != "identity" else {}
                if ch is None:
                    cands = None; break
                kind = repo.nodes[primary]["kind"]
                fmt = dict(sym=repo.short_name(primary, state), kind=kind, **args)
                cands.append({"edit": edit, "args": args,
                              "changes": sorted([state.index(k), (-1 if v is None else T(v))] for k, v in ch.items()),
                              "instr_seen": [s.format(**fmt) for s in TEMPLATES[edit]["seen"]],
                              "instr_unseen": [s.format(**fmt) for s in TEMPLATES[edit]["unseen"]]})
            if cands is None:
                continue
            prim_texts = [next((c for s, c in cd["changes"] if s == state.index(primary)), None) for cd in cands[1:]]
            if len(set(prim_texts)) != len(prim_texts):
                continue
            sources.append({"repo": ri, "primary_slot": state.index(primary), "primary_id": repo.nodes[primary]["id"],
                            "primary_kind": repo.nodes[primary]["kind"], "slot_node": state,
                            "slot_text": [T(repo.text[x]) for x in state], "cands": cands})
            got += 1
    order = rng.permutation(len(sources))
    sources = [sources[i] for i in order]
    return {"meta": {"n": len(sources), "seed": seed, "arg_bank": bank, "primary_kind": primary_kind, "n_slots": N_SLOTS,
                     "repos": [r.name for r in repos], "commits": [r.commit for r in repos],
                     "cand_edits": [c["edit"] for c in sources[0]["cands"]], "n_texts": len(texts)},
            "texts": texts, "sources": sources}


def build_multi(repos: List[Repo], n_total: int, seed: int, bank: str, site_counts: List[int]) -> dict:
    """Multi-symbol edits (HANDOFF section 12.3): one instruction edits k different symbols of the same
    source, k cycling through `site_counts`. Each edited symbol is a *site* with its own 8 candidates and
    the index `true` of the edit that the instruction asks for. Sites are chosen so that the slots changed
    by their true edits do not overlap, so the target is simply the union of the per-site changes.

    Source record: {"repo", "k", "n_files", "slot_node", "slot_text", "sites": [{"primary_slot", "primary_id",
    "primary_kind", "true", "instr_seen", "instr_unseen", "cands": [{"edit", "args", "changes"}]}],
    "changes": [[slot, text_idx]],
    "instr_seen": [4 strings], "instr_unseen": [2 strings]}; the instruction is one sentence per site."""
    rng = np.random.default_rng(seed)
    texts: List[str] = []
    tix: Dict[str, int] = {}

    def T(s: str) -> int:
        if s not in tix:
            tix[s] = len(texts); texts.append(s)
        return tix[s]

    per = [n_total // len(repos) + (i < n_total % len(repos)) for i in range(len(repos))]
    sources = []
    for ri, (repo, quota) in enumerate(zip(repos, per)):
        prim = repo.primaries("function")
        primset = set(prim)
        order = [prim[i] for i in rng.permutation(len(prim))]
        got = j = 0
        while got < quota:
            k = site_counts[got % len(site_counts)]
            first = order[j % len(order)]; j += 1
            if j > 40 * quota + 40 * len(order):
                raise RuntimeError(f"cannot build enough multi-site sources from {repo.name}")
            state = repo.state(first, rng)
            pool = [first] + [int(x) for x in rng.permutation([x for x in state if x in primset and x != first])]
            sites, used = [], set()
            for p in pool:
                if len(sites) == k:
                    break
                cands, bad = [], False
                for edit, args in candidates_for(repo, p, bank, rng):
                    ch = repo.apply(edit, args, p, state) if edit != "identity" else {}
                    if ch is None:
                        bad = True; break
                    cands.append((edit, args, ch))
                if bad:
                    continue
                prim_texts = [c[2][p] for c in cands[1:]]
                if len(set(prim_texts)) != len(prim_texts):
                    continue
                true = int(rng.integers(1, 5))
                touched = set(cands[true][2])
                if touched & used or len(touched) > 24:      # no overlap with earlier sites; skip giant renames
                    continue
                used |= touched
                fmt = dict(sym=repo.short_name(p, state), kind=repo.nodes[p]["kind"], **cands[true][1])
                sites.append({"node": p, "true": true, "cands": cands,
                              "seen": [x.format(**fmt) for x in TEMPLATES[cands[true][0]]["seen"]],
                              "unseen": [x.format(**fmt) for x in TEMPLATES[cands[true][0]]["unseen"]]})
            if len(sites) < k:
                continue
            union = {}
            for st in sites:
                union.update(st["cands"][st["true"]][2])
            sources.append({
                "repo": ri, "k": k, "n_files": len({repo.nodes[st["node"]]["file"] for st in sites}),
                "slot_node": state, "slot_text": [T(repo.text[x]) for x in state],
                "sites": [{"primary_slot": state.index(st["node"]), "primary_id": repo.nodes[st["node"]]["id"],
                           "primary_kind": repo.nodes[st["node"]]["kind"], "true": st["true"],
                           "instr_seen": st["seen"], "instr_unseen": st["unseen"],
                           "cands": [{"edit": e, "args": a,
                                      "changes": sorted([state.index(x), (-1 if v is None else T(v))] for x, v in ch.items())}
                                     for e, a, ch in st["cands"]]} for st in sites],
                "changes": sorted([state.index(x), (-1 if v is None else T(v))] for x, v in union.items()),
                "instr_seen": [" ".join(st["seen"][v] for st in sites) for v in range(4)],
                "instr_unseen": [" ".join(st["unseen"][v] for st in sites) for v in range(2)],
            })
            got += 1
    order = rng.permutation(len(sources))
    sources = [sources[i] for i in order]
    return {"meta": {"format": "multi", "n": len(sources), "seed": seed, "arg_bank": bank, "site_counts": site_counts,
                     "n_slots": N_SLOTS, "repos": [r.name for r in repos], "commits": [r.commit for r in repos],
                     "n_texts": len(texts)},
            "texts": texts, "sources": sources}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arg-bank", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--primary-kind", choices=["function", "class"], default="function")
    ap.add_argument("--sites", type=int, nargs="+",
                    help="multi-symbol edits: symbols edited per instruction, cycled (e.g. --sites 1 2)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    repos = [Repo(load_graph(p)) for p in a.graphs]
    os.makedirs(a.out, exist_ok=True)
    if a.sites:
        ds = build_multi(repos, a.n, a.seed, a.arg_bank, a.sites)
        path = os.path.join(a.out, f"nodeB_multi_{a.name}.json.gz")
        with gzip.GzipFile(path, "wb", mtime=0) as f:
            f.write(json.dumps(ds).encode())
        nch = np.array([len(x["changes"]) for x in ds["sources"]])
        print(json.dumps({**ds["meta"], "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}))
        print("  changed slots per source: mean %.2f max %d | sites per source:" % (nch.mean(), nch.max()),
              {int(a_): int(b_) for a_, b_ in zip(*np.unique([x["k"] for x in ds["sources"]], return_counts=True))},
              "| share whose sites span >1 file: %.2f" % np.mean([x["n_files"] > 1 for x in ds["sources"] if x["k"] > 1]))
        return
    ds = build(repos, a.n, a.seed, a.arg_bank, a.primary_kind)
    path = os.path.join(a.out, f"nodeB_{a.name}.json.gz")
    raw = json.dumps(ds).encode()
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(raw)
    nch = np.array([[len(c["changes"]) for c in s["cands"][1:]] for s in ds["sources"]])
    print(json.dumps({**ds["meta"], "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}))
    print("  distinct primaries", len({s["primary_id"] for s in ds["sources"]}),
          "| changed slots per edit: mean", np.round(nch.mean(0), 2).tolist(), "max", nch.max(0).tolist())


if __name__ == "__main__":
    main()
