#!/usr/bin/env python3
"""Build a self-contained (<500 MB) external reviewer archive for the Kaggle Gemma 4 submission."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tarfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
PARENT_ROOT = Path.home() / "workspaces" / "arc-agi-3"
ARCHIVE_PREFIX = "dgg2_kaggle_gemma4_external_review"
OUT_DIR = REPO_ROOT / "outputs" / "external_review"
MAX_BYTES = 500 * 1024 * 1024  # 500 MB hard ceiling

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".swp"}
ALLOWED_EMAIL = "21022766+bumblebeers@users.noreply.github.com"
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def collect_tree(src_dir: Path, dst_prefix: str, *, suffixes: Tuple[str, ...] | None = None) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if not src_dir.exists():
        return items
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(files):
            if any(fn.endswith(s) for s in SKIP_SUFFIXES):
                continue
            if suffixes is not None and not fn.endswith(suffixes):
                continue
            full = Path(root) / fn
            if not full.is_file():
                continue
            rel = full.relative_to(src_dir).as_posix()
            arc_rel = f"{dst_prefix}/{rel}".strip("/")
            items.append((arc_rel, full))
    return items


def gather_entries() -> List[Tuple[str, Path]]:
    entries: Dict[str, Path] = {}

    def add(arc_rel: str, src: Path) -> None:
        if src.is_file():
            entries[arc_rel] = src

    # 1. Top-level reviewer entrypoints
    add("README_EXTERNAL_REVIEW.md", REPO_ROOT / "docs" / "EXTERNAL_REVIEW_GUIDE.md")
    add("KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md", REPO_ROOT / "docs" / "KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md")

    # 2. Root project governance, pre-registration, and config files
    for root_file in [
        "AGENTS.md",
        "PREREG.json",
        "PROJECT.md",
        "GOALS.md",
        "DEAD_ENDS.md",
        "TEST_INFRA.md",
        "TEST_READY.md",
        "pyproject.toml",
        ".gitignore",
        "reader.npz",
        "reader_n32.npz",
        "reader_n48.npz",
    ]:
        add(root_file, REPO_ROOT / root_file)

    # 3. Full tracked source, docs, notebooks, and tests
    for folder in ["docs", "cg", "src", "tests", "notebooks"]:
        for arc_rel, src in collect_tree(REPO_ROOT / folder, folder):
            add(arc_rel, src)

    # 4. Compact AST graphs, 16x16 / 32x32 / 48x48 grids, and 256-symbol packs (excluding multi-GB data/hv)
    for arc_rel, src in collect_tree(REPO_ROOT / "data" / "graphs", "data/graphs"):
        add(arc_rel, src)
    data_dir = REPO_ROOT / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.iterdir()):
            if p.is_file() and (p.name.startswith("gridA_") or p.name.startswith("nodeB_")):
                add(f"data/{p.name}", p)

    # 5. Milestone 1-4 JSON evaluation reports, probes, decoded visual, and built Python wheels
    for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / "m1", "outputs/m1"):
        add(arc_rel, src)
    for m_dir in ["m2", "m3"]:
        for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / m_dir, f"outputs/{m_dir}", suffixes=(".json", ".md")):
            add(arc_rel, src)
    add("outputs/m2/decoded_grids_all_arms.npz", REPO_ROOT / "outputs" / "m2" / "decoded_grids_all_arms.npz")
    add("outputs/m4/staging_manifest.json", REPO_ROOT / "outputs" / "m4" / "staging_manifest.json")
    for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / "m4" / "wheels", "outputs/m4/wheels", suffixes=(".whl",)):
        add(arc_rel, src)
    add("outputs/qual_dev_000031_rot90_decoded.png", REPO_ROOT / "outputs" / "qual_dev_000031_rot90_decoded.png")

    # 6. Read-only parent natural-image (§26-§27) reference & stock contract
    if PARENT_ROOT.is_dir():
        add("parent_sec26_27_reference/DG-STOCK-REFERENCE.md", PARENT_ROOT / "DG-STOCK-REFERENCE.md")
        pkg = PARENT_ROOT / "rendered" / "dgg2" / "paper_data_package"
        for fn in [
            "claims_ledger.json",
            "claims_ledger.md",
            "not_run_and_failures.md",
            "paper_data_master.json",
            "per_item_and_figures_manifest.json",
        ]:
            add(f"parent_sec26_27_reference/paper_data_package/{fn}", pkg / fn)
        for sub in ["tables", "figures"]:
            for arc_rel, src in collect_tree(pkg / sub, f"parent_sec26_27_reference/paper_data_package/{sub}"):
                add(arc_rel, src)

    return sorted(entries.items(), key=lambda kv: kv[0])


def audit_privacy(entries: List[Tuple[str, Path]]) -> None:
    binary_exts = (".png", ".webp", ".npz", ".gz", ".whl", ".pt")
    bad: Dict[str, List[str]] = {}
    for arc_rel, src in entries:
        if arc_rel.endswith(binary_exts):
            continue
        txt = src.read_text(encoding="utf-8", errors="ignore")
        found = sorted(
            {
                m
                for m in EMAIL_RE.findall(txt)
                if m != ALLOWED_EMAIL and not re.search(r"@L\d+\.", m)
            }
        )
        if found:
            bad[arc_rel] = found
    if bad:
        raise RuntimeError(f"Privacy check failed — unexpected email strings in archive candidates: {bad}")


def build_archives() -> Tuple[Path, Path, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = gather_entries()
    audit_privacy(entries)

    file_records = []
    sha_lines = []
    total_raw_bytes = 0

    for arc_rel, src in entries:
        size = src.stat().st_size
        digest = sha256_file(src)
        total_raw_bytes += size
        file_records.append({"path": arc_rel, "size_bytes": size, "sha256": digest})
        sha_lines.append(f"{digest}  {arc_rel}")

    manifest_obj = {
        "archive_name": ARCHIVE_PREFIX,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": "bumblebeers",
        "competition": "Kaggle Gemma 4 Developer Agent Paper Track",
        "competition_url": "https://www.kaggle.com/competitions/gemma-4-developer-agent-paper",
        "file_count": len(file_records) + 2,  # + MANIFEST.json & SHA256SUMS
        "uncompressed_payload_bytes": total_raw_bytes,
        "max_archive_budget_bytes": MAX_BYTES,
        "files": file_records,
    }
    manifest_bytes = (json.dumps(manifest_obj, indent=2) + "\n").encode("utf-8")
    manifest_sha = sha256_bytes(manifest_bytes)
    sha_lines.append(f"{manifest_sha}  MANIFEST.json")
    sha_bytes = ("\n".join(sha_lines) + "\n").encode("utf-8")

    now_ts = int(time.time())
    tar_path = OUT_DIR / f"{ARCHIVE_PREFIX}_pack.tar.gz"
    zip_path = OUT_DIR / f"{ARCHIVE_PREFIX}_pack.zip"

    # Write sanitized .tar.gz
    with tarfile.open(tar_path, "w:gz", compresslevel=6) as tf:
        for arc_rel, src in entries:
            ti = tf.gettarinfo(str(src), arcname=f"{ARCHIVE_PREFIX}/{arc_rel}")
            ti.uid = 0
            ti.gid = 0
            ti.uname = "bumblebeers"
            ti.gname = "bumblebeers"
            with open(src, "rb") as f:
                tf.addfile(ti, f)
        for meta_name, payload in [("MANIFEST.json", manifest_bytes), ("SHA256SUMS", sha_bytes)]:
            ti = tarfile.TarInfo(name=f"{ARCHIVE_PREFIX}/{meta_name}")
            ti.size = len(payload)
            ti.mtime = now_ts
            ti.mode = 0o644
            ti.uid = 0
            ti.gid = 0
            ti.uname = "bumblebeers"
            ti.gname = "bumblebeers"
            tf.addfile(ti, io.BytesIO(payload))

    # Write companion .zip
    dt_tuple = time.gmtime(now_ts)[:6]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for arc_rel, src in entries:
            zi = zipfile.ZipInfo(filename=f"{ARCHIVE_PREFIX}/{arc_rel}", date_time=dt_tuple)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if os.access(src, os.X_OK) else 0o644) << 16
            zf.writestr(zi, src.read_bytes())
        for meta_name, payload in [("MANIFEST.json", manifest_bytes), ("SHA256SUMS", sha_bytes)]:
            zi = zipfile.ZipInfo(filename=f"{ARCHIVE_PREFIX}/{meta_name}", date_time=dt_tuple)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, payload)

    for p in (tar_path, zip_path):
        sz = p.stat().st_size
        if sz > MAX_BYTES:
            raise RuntimeError(f"Archive {p} ({sz} bytes) exceeds 500 MB limit!")

    summary = {
        "tar_gz_path": str(tar_path),
        "tar_gz_mb": round(tar_path.stat().st_size / (1024 * 1024), 2),
        "tar_gz_sha256": sha256_file(tar_path),
        "zip_path": str(zip_path),
        "zip_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "zip_sha256": sha256_file(zip_path),
        "file_count": manifest_obj["file_count"],
        "uncompressed_mb": round(total_raw_bytes / (1024 * 1024), 2),
    }
    (OUT_DIR / "external_review_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return tar_path, zip_path, summary


if __name__ == "__main__":
    _, _, info = build_archives()
    print(json.dumps(info, indent=2))
