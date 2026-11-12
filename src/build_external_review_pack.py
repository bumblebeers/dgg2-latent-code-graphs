#!/usr/bin/env python3
"""Build a self-contained (<= 100 MB) project-only external review archive."""

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
ARCHIVE_PREFIX = "dgg2_latent_code_graphs_review"
OUT_DIR = REPO_ROOT / "outputs" / "external_review"
MAX_BYTES = 100 * 1024 * 1024  # Strict 100 MB hard ceiling

SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", ".mypy_cache", ".ruff_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".swp"}

# Exclude competition meta-instructions and internal orchestration/planning docs
SKIP_DOC_FILES = {
    "KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md",
    "PAPER_BLUEPRINT.md",
    "EXTERNAL_REVIEW_GUIDE.md",  # placed cleanly at archive root as README_REVIEW.md
}
SKIP_SRC_FILES = {
    "publish_github_repos.py",
}

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


def sanitize_json_bytes(src: Path, arc_rel: str = "") -> bytes:
    """Return bytes with any local host path strings stripped from JSON or read-only parent markdown."""
    local_user = Path.home().name
    home_str = str(Path.home())
    raw = src.read_bytes()
    if arc_rel.startswith("parent_sec26_27_reference/") and src.name.endswith((".md", ".json")):
        txt = raw.decode("utf-8", errors="ignore")
        if local_user and local_user in txt:
            txt = txt.replace(home_str, "~").replace(local_user, "user")
            return txt.encode("utf-8")
    if not src.name.endswith(".json"):
        return raw
    txt = raw.decode("utf-8", errors="ignore")
    if local_user and local_user in txt:
        data = json.loads(txt)
        if isinstance(data, dict) and "assets" in data and isinstance(data["assets"], list):
            for item in data["assets"]:
                if isinstance(item, dict):
                    item.pop("absolute_path", None)
                    for k, v in list(item.items()):
                        if isinstance(v, str) and local_user in v:
                            item[k] = Path(v).name
        return (json.dumps(data, indent=2) + "\n").encode("utf-8")
    return raw


def collect_tree(
    src_dir: Path,
    dst_prefix: str,
    *,
    suffixes: Tuple[str, ...] | None = None,
    skip_names: set[str] | None = None,
) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    if not src_dir.exists():
        return items
    skip = skip_names or set()
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for fn in sorted(files):
            if fn in skip or any(fn.endswith(s) for s in SKIP_SUFFIXES):
                continue
            if suffixes is not None and not fn.endswith(suffixes):
                continue
            full = Path(root) / fn
            if not full.is_file() or full.is_symlink():
                continue
            rel = full.relative_to(src_dir).as_posix()
            arc_rel = f"{dst_prefix}/{rel}".strip("/")
            items.append((arc_rel, full))
    return items


def gather_entries() -> List[Tuple[str, Path]]:
    entries: Dict[str, Path] = {}

    def add(arc_rel: str, src: Path) -> None:
        if src.is_file() and not src.is_symlink():
            entries[arc_rel] = src

    # 1. Top-level project review guide (project-specific info only)
    add("README_REVIEW.md", REPO_ROOT / "docs" / "EXTERNAL_REVIEW_GUIDE.md")

    # 2. Root project files, pre-registration lock, and clean-h_v Ridge probes
    for root_file in [
        "README.md",
        "LICENSE",
        "PREREG.json",
        "DEAD_ENDS.md",
        "pyproject.toml",
        ".gitignore",
        "reader.npz",
        "reader_n32.npz",
        "reader_n48.npz",
    ]:
        add(root_file, REPO_ROOT / root_file)

    # 3. Tracked project docs (excluding competition meta files), source, notebooks, and tests
    for arc_rel, src in collect_tree(REPO_ROOT / "docs", "docs", skip_names=SKIP_DOC_FILES):
        add(arc_rel, src)
    for arc_rel, src in collect_tree(REPO_ROOT / "src", "src", skip_names=SKIP_SRC_FILES):
        add(arc_rel, src)
    for folder in ["cg", "tests", "notebooks"]:
        for arc_rel, src in collect_tree(REPO_ROOT / folder, folder, skip_names={"kernel-metadata.json"}):
            add(arc_rel, src)

    # 4. Compact AST graphs, 16x16 / 32x32 / 48x48 grids, and 256-symbol packs (excluding multi-GB data/hv)
    for arc_rel, src in collect_tree(REPO_ROOT / "data" / "graphs", "data/graphs"):
        add(arc_rel, src)
    data_dir = REPO_ROOT / "data"
    if data_dir.is_dir():
        for p in sorted(data_dir.iterdir()):
            if p.is_file() and (p.name.startswith("gridA_") or p.name.startswith("nodeB_")):
                add(f"data/{p.name}", p)

    # 5. JSON evaluation reports, probes, decoded grids, and built Python wheels
    for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / "m1", "outputs/m1"):
        add(arc_rel, src)
    for m_dir in ["m2", "m3"]:
        for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / m_dir, f"outputs/{m_dir}", suffixes=(".json", ".md")):
            add(arc_rel, src)
    add("outputs/m2/decoded_grids_all_arms.npz", REPO_ROOT / "outputs" / "m2" / "decoded_grids_all_arms.npz")
    add("outputs/m2/refeed_trace_item0.npz", REPO_ROOT / "outputs" / "m2" / "refeed_trace_item0.npz")
    add("outputs/m4/staging_manifest.json", REPO_ROOT / "outputs" / "m4" / "staging_manifest.json")
    for arc_rel, src in collect_tree(REPO_ROOT / "outputs" / "m4" / "wheels", "outputs/m4/wheels", suffixes=(".whl",)):
        add(arc_rel, src)

    # 6. Read-only parent natural-image (§26-§27) reference & stock contract (tables & JSON/MD claims; omit multi-MB parent debug PNGs to stay well under 100 MB)
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
        for arc_rel, src in collect_tree(pkg / "tables", "parent_sec26_27_reference/paper_data_package/tables"):
            add(arc_rel, src)

    return sorted(entries.items(), key=lambda kv: kv[0])


def audit_privacy(payloads: List[Tuple[str, bytes]]) -> None:
    binary_exts = (".png", ".webp", ".npz", ".gz", ".whl", ".pt", ".jpg")
    local_user = Path.home().name
    bad: Dict[str, List[str]] = {}
    for arc_rel, data in payloads:
        if arc_rel.endswith(binary_exts):
            continue
        txt = data.decode("utf-8", errors="ignore")
        issues: List[str] = []
        if local_user and local_user in txt:
            issues.append(f"local_path:{local_user}")
        found = sorted(
            {
                m
                for m in EMAIL_RE.findall(txt)
                if m != ALLOWED_EMAIL and not re.search(r"@L\d+\.", m)
            }
        )
        issues.extend(found)
        if issues:
            bad[arc_rel] = issues
    if bad:
        raise RuntimeError(f"Privacy check failed — unexpected strings in archive candidates: {bad}")


def build_archives() -> Tuple[Path, Path, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any legacy competition-prefixed archives so only the clean <=100 MB packet exists
    for old in OUT_DIR.glob("dgg2_kaggle_gemma4_external_review*"):
        old.unlink()

    entries = gather_entries()
    payloads: List[Tuple[str, Path, bytes]] = [(arc_rel, src, sanitize_json_bytes(src, arc_rel)) for arc_rel, src in entries]
    audit_privacy([(arc_rel, data) for arc_rel, _, data in payloads])

    file_records = []
    sha_lines = []
    total_raw_bytes = 0

    for arc_rel, _, data in payloads:
        size = len(data)
        digest = sha256_bytes(data)
        total_raw_bytes += size
        file_records.append({"path": arc_rel, "size_bytes": size, "sha256": digest})
        sha_lines.append(f"{digest}  {arc_rel}")

    manifest_obj = {
        "archive_name": ARCHIVE_PREFIX,
        "project": "Latent Roadmaps: Soft-Token Re-Feeding in DiffusionGemma for Code Graphs",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": "bumblebeers",
        "repository_url": "https://github.com/bumblebeers/dgg2-latent-code-graphs",
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
        for arc_rel, src, data in payloads:
            ti = tarfile.TarInfo(name=f"{ARCHIVE_PREFIX}/{arc_rel}")
            ti.size = len(data)
            ti.mtime = int(src.stat().st_mtime)
            ti.mode = 0o755 if os.access(src, os.X_OK) else 0o644
            ti.uid = 0
            ti.gid = 0
            ti.uname = "bumblebeers"
            ti.gname = "bumblebeers"
            tf.addfile(ti, io.BytesIO(data))
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
        for arc_rel, src, data in payloads:
            zi = zipfile.ZipInfo(filename=f"{ARCHIVE_PREFIX}/{arc_rel}", date_time=dt_tuple)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = (0o755 if os.access(src, os.X_OK) else 0o644) << 16
            zf.writestr(zi, data)
        for meta_name, payload in [("MANIFEST.json", manifest_bytes), ("SHA256SUMS", sha_bytes)]:
            zi = zipfile.ZipInfo(filename=f"{ARCHIVE_PREFIX}/{meta_name}", date_time=dt_tuple)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, payload)

    for p in (tar_path, zip_path):
        sz = p.stat().st_size
        if sz > MAX_BYTES:
            raise RuntimeError(f"Archive {p} ({sz} bytes) exceeds 100 MB limit!")

    summary = {
        "tar_gz_path": str(tar_path),
        "tar_gz_mb": round(tar_path.stat().st_size / (1024 * 1024), 2),
        "tar_gz_sha256": sha256_file(tar_path),
        "zip_path": str(zip_path),
        "zip_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "zip_sha256": sha256_file(zip_path),
        "file_count": manifest_obj["file_count"],
        "uncompressed_mb": round(total_raw_bytes / (1024 * 1024), 2),
        "max_limit_mb": 100.0,
    }
    (OUT_DIR / "external_review_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return tar_path, zip_path, summary


if __name__ == "__main__":
    _, _, info = build_archives()
    print(json.dumps(info, indent=2))
