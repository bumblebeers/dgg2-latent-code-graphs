#!/usr/bin/env python3
"""cg_extract.py -- directed call-graph extractor for a Python source tree.

Standard library only. The repository is PARSED, never imported or executed.

Output: one JSON file in node-link form (same top-level keys as the Kaggle
`graphs/*.json` files, so existing loaders work):

    {"directed": true, "multigraph": false, "graph": {...metadata...},
     "nodes": [{"id", "name", "kind", "module", "file", "lineno", "is_test",
                "is_async", "parent", "text"}, ...],
     "edges": [{"source", "target", "type", "how", "count"}, ...]}

Differences from the shipped Kaggle graphs (see HANDOFF.md section 3):
  * edges are genuinely DIRECTED (caller -> callee); the shipped ones are symmetric
  * `async def` functions are included
  * ids follow real Python scoping (module.Class.method, module.outer.inner)
  * class nodes keep their real source in `text`

Edge types:  "calls" (function/method/class body calls something),
             "inherits" (class -> base class).
`how` records how the callee was resolved, so strict/loose subsets can be taken:
  local | module | import | self | super | dotted | class_attr | unique_name
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import textwrap
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

EXTRACTOR_VERSION = "cg_extract/1.0"
SKIP_DIRS = {".git", ".hg", ".tox", ".nox", ".venv", "venv", "env", "node_modules", "build", "dist",
             "__pycache__", ".eggs", "site-packages", ".mypy_cache", ".pytest_cache"}
TEST_DIR_NAMES = {"tests", "test", "testing"}
STRIP_PREFIXES = ("src", "lib")


def module_name_for(rel_path: str) -> str:
    parts = rel_path[:-3].replace("\\", "/").split("/")
    if len(parts) > 1 and parts[0] in STRIP_PREFIXES:
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else "__root__"


def is_test_path(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    base = parts[-1]
    return (any(p in TEST_DIR_NAMES for p in parts[:-1]) or base.startswith("test_")
            or base.endswith("_test.py") or base == "conftest.py")


class ModuleInfo:
    def __init__(self, name: str, rel: str, is_pkg: bool):
        self.name, self.rel, self.is_pkg = name, rel, is_pkg
        self.imports: Dict[str, str] = {}      # local alias -> absolute dotted target
        self.defs: Dict[str, str] = {}         # module-level name -> node id
        self.star: List[str] = []              # modules star-imported


def _node_text(lines: List[str], node: ast.AST) -> str:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return textwrap.dedent("".join(lines[start - 1: node.end_lineno])).rstrip() + "\n"


class Collector(ast.NodeVisitor):
    """Pass 1: definitions and imports for one module."""

    def __init__(self, mod: ModuleInfo, lines: List[str], is_test: bool, nodes: Dict[str, dict],
                 children: Dict[str, Dict[str, str]], body_of: Dict[str, ast.AST]):
        self.mod, self.lines, self.is_test = mod, lines, is_test
        self.nodes, self.children, self.body_of = nodes, children, body_of
        self.stack: List[Tuple[str, str]] = []  # (kind, node id)

    # ---- imports -------------------------------------------------------
    def _pkg(self) -> str:
        return self.mod.name if self.mod.is_pkg else self.mod.name.rpartition(".")[0]

    def visit_Import(self, node: ast.Import):
        if self.stack and self.stack[-1][0] != "module":
            pass  # function-level imports are still recorded at module level (good enough)
        for a in node.names:
            if a.asname:
                self.mod.imports[a.asname] = a.name
            else:
                top = a.name.split(".")[0]
                self.mod.imports.setdefault(top, top)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        base = node.module or ""
        if node.level:
            pkg = self._pkg().split(".") if self._pkg() else []
            up = node.level - 1
            pkg = pkg[: len(pkg) - up] if up else pkg
            base = ".".join([p for p in pkg if p] + ([base] if base else []))
        for a in node.names:
            if a.name == "*":
                self.mod.star.append(base)
            else:
                self.mod.imports[a.asname or a.name] = f"{base}.{a.name}" if base else a.name

    # ---- definitions ---------------------------------------------------
    def _scope_id(self) -> str:
        return self.stack[-1][1] if self.stack else self.mod.name

    def _add(self, node, kind: str):
        parent = self._scope_id()
        nid = f"{parent}.{node.name}"
        if nid in self.nodes:  # redefinition (overloads, if/else defs): keep both, suffix the later one
            nid = f"{nid}@L{node.lineno}"
        in_class = bool(self.stack) and self.stack[-1][0] == "class"
        if kind == "function" and in_class:
            kind = "method"
        self.nodes[nid] = {
            "id": nid, "name": node.name, "kind": kind, "module": self.mod.name, "file": self.mod.rel,
            "lineno": node.lineno, "end_lineno": node.end_lineno, "is_test": self.is_test,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "parent": parent if self.stack else None, "text": _node_text(self.lines, node),
        }
        if isinstance(node, ast.ClassDef):
            self.nodes[nid]["bases_raw"] = [ast.unparse(b) for b in node.bases]
        self.children[parent].setdefault(node.name, nid)
        if not self.stack:
            self.mod.defs.setdefault(node.name, nid)
        self.body_of[nid] = node
        self.stack.append(("class" if isinstance(node, ast.ClassDef) else "func", nid))
        for child in node.body:
            self.visit(child)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._add(node, "function")

    def visit_AsyncFunctionDef(self, node):
        self._add(node, "function")

    def visit_ClassDef(self, node):
        self._add(node, "class")


def _own_calls(defnode: ast.AST):
    """Yield (ast.Call) found in a definition's own body, not inside nested defs/classes."""
    todo = list(ast.iter_child_nodes(defnode))
    while todo:
        n = todo.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            todo.extend(n.decorator_list)   # decorators run in the enclosing scope
            continue
        if isinstance(n, ast.Call):
            yield n
        todo.extend(ast.iter_child_nodes(n))


def _dotted(expr: ast.AST) -> Optional[List[str]]:
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return parts[::-1]
    return None


class Resolver:
    def __init__(self, modules: Dict[str, ModuleInfo], nodes: Dict[str, dict], children, loose: bool):
        self.modules, self.nodes, self.children, self.loose = modules, nodes, children, loose
        self.by_name: Dict[str, List[str]] = defaultdict(list)
        for nid, n in nodes.items():
            if n["kind"] in ("function", "method"):
                self.by_name[n["name"]].append(nid)
        self.bases: Dict[str, List[str]] = {}

    # resolve an absolute dotted path (e.g. "pkg.mod.Class.meth") to a node id
    def dotted(self, path: str, depth: int = 0) -> Optional[str]:
        if path in self.nodes:
            return path
        if depth > 6:
            return None
        parts = path.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            head, rest = ".".join(parts[:cut]), parts[cut:]
            if head in self.modules:
                m = self.modules[head]
                first = rest[0]
                target = m.defs.get(first)
                if target is None and first in m.imports:
                    target = self.dotted(m.imports[first], depth + 1) or None
                    if target is None:  # may be a module alias
                        return self.dotted(".".join([m.imports[first]] + rest[1:]), depth + 1)
                if target is None:
                    for sm in m.star:
                        target = self.dotted(f"{sm}.{first}", depth + 1)
                        if target:
                            break
                if target is None:
                    return None
                return self._descend(target, rest[1:])
            if head in self.nodes:
                return self._descend(head, rest)
        return None

    def _descend(self, nid: str, rest: List[str]) -> Optional[str]:
        for name in rest:
            nxt = self.member(nid, name)
            if nxt is None:
                return None
            nid = nxt
        return nid

    def class_bases(self, cid: str) -> List[str]:
        if cid not in self.bases:
            self.bases[cid] = []  # guard against cycles
            out = []
            for raw in self.nodes[cid].get("bases_raw", []):
                try:
                    expr = ast.parse(raw, mode="eval").body
                except SyntaxError:
                    continue
                if isinstance(expr, ast.Subscript):   # Generic[T] style
                    expr = expr.value
                d = _dotted(expr)
                if d:
                    r = self.name_in_scope(self.nodes[cid]["module"], self.nodes[cid]["parent"], d)
                    if r and self.nodes[r]["kind"] == "class" and r != cid:
                        out.append(r)
            self.bases[cid] = out
        return self.bases[cid]

    def member(self, nid: str, name: str, seen=None) -> Optional[str]:
        got = self.children.get(nid, {}).get(name)
        if got:
            return got
        if self.nodes[nid]["kind"] == "class":
            seen = seen or set()
            for b in self.class_bases(nid):
                if b not in seen:
                    seen.add(b)
                    got = self.member(b, name, seen)
                    if got:
                        return got
        return None

    def name_in_scope(self, module: str, scope: Optional[str], d: List[str]) -> Optional[str]:
        """Resolve dotted name parts `d` as seen from `scope` (a node id or None) in `module`."""
        first, rest = d[0], d[1:]
        s = scope
        while s and s in self.nodes:            # enclosing function scopes (classes are skipped by Python)
            if self.nodes[s]["kind"] != "class" and first in self.children.get(s, {}):
                return self._descend(self.children[s][first], rest)
            s = self.nodes[s]["parent"]
        m = self.modules[module]
        if first in m.defs:
            return self._descend(m.defs[first], rest)
        if first in m.imports:
            return self.dotted(".".join([m.imports[first]] + rest))
        for sm in m.star:
            r = self.dotted(".".join([sm, first] + rest))
            if r:
                return r
        return None

    def call(self, src: str, call: ast.Call) -> Optional[Tuple[str, str]]:
        n = self.nodes[src]
        f = call.func
        encl_class = None
        s = src
        while s and s in self.nodes:
            if self.nodes[s]["kind"] == "class":
                encl_class = s
                break
            s = self.nodes[s]["parent"]
        if isinstance(f, ast.Name):
            scope = src if n["kind"] != "class" else n["parent"]
            r = self.name_in_scope(n["module"], scope, [f.id])
            if r:
                how = "local" if self.nodes[r]["parent"] and self.nodes[r]["module"] == n["module"] and \
                    self.nodes[r]["parent"] in self.nodes and self.nodes[self.nodes[r]["parent"]]["kind"] != "class" \
                    else ("module" if self.nodes[r]["module"] == n["module"] else "import")
                return r, how
            return None
        if isinstance(f, ast.Attribute):
            v = f.value
            if isinstance(v, ast.Name) and v.id in ("self", "cls") and encl_class:
                r = self.member(encl_class, f.attr)
                if r:
                    return r, "self"
            elif isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "super" and encl_class:
                for b in self.class_bases(encl_class):
                    r = self.member(b, f.attr)
                    if r:
                        return r, "super"
            else:
                d = _dotted(f)
                if d:
                    scope = src if n["kind"] != "class" else n["parent"]
                    r = self.name_in_scope(n["module"], scope, d)
                    if r:
                        return r, ("class_attr" if self.nodes[r]["kind"] == "method" else "dotted")
            if self.loose:
                c = self.by_name.get(f.attr, [])
                if len(c) == 1 and not f.attr.startswith("__"):
                    return c[0], "unique_name"
        return None


def extract(root: str, loose: bool = True, repo: str = "", commit: str = "") -> dict:
    root = os.path.abspath(root)
    modules: Dict[str, ModuleInfo] = {}
    nodes: Dict[str, dict] = {}
    children: Dict[str, Dict[str, str]] = defaultdict(dict)
    body_of: Dict[str, ast.AST] = {}
    n_files = n_bad = 0
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS and not d.startswith("."))
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            full = os.path.join(dp, fn)
            rel = os.path.relpath(full, root)
            try:
                src = open(full, encoding="utf-8", errors="replace").read()
                tree = ast.parse(src)
            except (SyntaxError, ValueError, RecursionError):
                n_bad += 1
                continue
            n_files += 1
            name = module_name_for(rel)
            if name in modules:           # two files mapping to one module name: keep the first
                continue
            mod = ModuleInfo(name, rel.replace(os.sep, "/"), fn == "__init__.py")
            modules[name] = mod
            Collector(mod, src.splitlines(keepends=True), is_test_path(rel), nodes, children, body_of).visit(tree)

    res = Resolver(modules, nodes, children, loose)
    edges: Dict[Tuple[str, str, str], dict] = {}
    for nid, defnode in body_of.items():
        if nodes[nid]["kind"] == "class":
            for b in res.class_bases(nid):
                edges[(nid, b, "inherits")] = {"source": nid, "target": b, "type": "inherits", "how": "base", "count": 1}
        for call in _own_calls(defnode):
            r = res.call(nid, call)
            if not r or r[0] == nid and False:
                continue
            tgt, how = r
            key = (nid, tgt, "calls")
            if key in edges:
                edges[key]["count"] += 1
            else:
                edges[key] = {"source": nid, "target": tgt, "type": "calls", "how": how, "count": 1}
    for n in nodes.values():
        n.pop("end_lineno", None)
    kinds = defaultdict(int)
    for n in nodes.values():
        kinds[n["kind"]] += 1
    calls = [e for e in edges.values() if e["type"] == "calls" and e["source"] != e["target"]]
    pair = {(e["source"], e["target"]) for e in calls}
    meta = {
        "extractor": EXTRACTOR_VERSION, "repo": repo, "commit": commit, "files_parsed": n_files, "files_failed": n_bad,
        "node_kinds": dict(kinds), "n_async": sum(n["is_async"] for n in nodes.values()),
        "n_test_nodes": sum(n["is_test"] for n in nodes.values()),
        "n_call_edges": len(calls), "n_inherit_edges": sum(e["type"] == "inherits" for e in edges.values()),
        "n_asymmetric_call_edges": sum((b, a) not in pair for a, b in pair), "loose_resolution": loose,
    }
    return {"directed": True, "multigraph": False, "graph": meta,
            "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
            "edges": sorted(edges.values(), key=lambda e: (e["source"], e["target"], e["type"]))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="repository root to parse")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--repo", default="", help="label stored in metadata, e.g. encode/httpx")
    ap.add_argument("--commit", default="", help="commit / tree hash stored in metadata")
    ap.add_argument("--strict", action="store_true", help="disable the unique-method-name fallback")
    a = ap.parse_args(argv)
    g = extract(a.root, loose=not a.strict, repo=a.repo, commit=a.commit)
    with open(a.out, "w") as f:
        json.dump(g, f)
    print(json.dumps(g["graph"], indent=1))


if __name__ == "__main__":
    main()
