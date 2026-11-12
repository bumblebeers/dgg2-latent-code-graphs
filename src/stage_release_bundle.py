#!/usr/bin/env python3
"""Stage Release Bundle: Catalogs and stages checkpoints, datasets, libraries, and predictions.

Produces distribution manifests with SHA256 checksums, byte sizes, and staging targets
(GitHub, HuggingFace/Kaggle, Google Drive), and optionally stages assets to external NVMe storage.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "outputs" / "m4" / "staging_manifest.json"
DEFAULT_EXTERNAL_DIR = Path(
    os.environ.get("DGG2_STAGING_DIR", str(REPO_ROOT / "outputs" / "m4" / "external_stage"))
)
PARENT_REPO = Path(os.path.expanduser("~/workspaces/arc-agi-3"))


def compute_sha256(filepath: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 checksum of a file in streaming chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def catalog_asset(
    name: str,
    path: Path,
    category: str,
    staging_target: str,
    description: str,
) -> Optional[Dict[str, Any]]:
    """Generate catalog entry for a file asset."""
    if not path.exists():
        return None
    try:
        size = path.stat().st_size
        sha = compute_sha256(path)
        try:
            rel_path = str(path.relative_to(REPO_ROOT))
        except ValueError:
            rel_path = str(path)

        return {
            "dataset_name": name,
            "path": rel_path,
            "absolute_path": str(path.resolve()),
            "category": category,
            "staging_target": staging_target,
            "sha256": sha,
            "size_bytes": size,
            "description": description,
        }
    except Exception as e:
        print(f"Warning: Failed to catalog {path}: {e}", file=sys.stderr)
        return None


def collect_assets() -> List[Dict[str, Any]]:
    """Scan and catalog all core project assets."""
    assets: List[Dict[str, Any]] = []

    # 1. Datasets & Test Packs
    data_dir = REPO_ROOT / "data"
    if data_dir.exists():
        for fn in sorted(data_dir.glob("*.npz")):
            entry = catalog_asset(
                name=fn.stem,
                path=fn,
                category="dataset_pack",
                staging_target="huggingface_kaggle",
                description=f"Experiment A grid dataset pack: {fn.name}",
            )
            if entry:
                assets.append(entry)

        for fn in sorted(data_dir.glob("*.json.gz")):
            entry = catalog_asset(
                name=fn.name.split(".")[0],
                path=fn,
                category="dataset_pack",
                staging_target="huggingface_kaggle",
                description=f"Experiment B node dataset pack: {fn.name}",
            )
            if entry:
                assets.append(entry)

        for fn in sorted(data_dir.glob("*.nodes.json")):
            entry = catalog_asset(
                name=fn.stem,
                path=fn,
                category="metadata",
                staging_target="huggingface_kaggle",
                description=f"Node symbol metadata: {fn.name}",
            )
            if entry:
                assets.append(entry)

        for r_cand in [REPO_ROOT / "outputs" / "m1" / "reader.npz", data_dir / "reader.npz", REPO_ROOT / "reader.npz"]:
            if r_cand.exists():
                entry = catalog_asset(
                    name=f"cell_reader_probe_{r_cand.parent.name}" if r_cand.parent != REPO_ROOT else "cell_reader_probe",
                    path=r_cand,
                    category="probe_weights",
                    staging_target="huggingface_kaggle",
                    description="Gate A1 linear cell reader probe weights for 8 palette classes",
                )
                if entry and not any(a["path"] == entry["path"] for a in assets):
                    assets.append(entry)

        for emb_cand in [REPO_ROOT / "emb.npz", data_dir / "emb.npz"]:
            if emb_cand.exists():
                entry = catalog_asset(
                    name="all_minilm_embeddings",
                    path=emb_cand,
                    category="embeddings",
                    staging_target="huggingface_kaggle",
                    description="Gate B0 pre-computed symbol text embeddings with normalization stats",
                )
                if entry and not any(a["path"] == entry["path"] for a in assets):
                    assets.append(entry)

        hv_dir = data_dir / "hv"
        if hv_dir.exists():
            for fn in sorted(hv_dir.glob("*.npy")):
                entry = catalog_asset(
                    name=f"hv_{fn.stem}",
                    path=fn,
                    category="latent_cache",
                    staging_target="google_drive",
                    description=f"Precomputed SigLIP h_v latent state cache: {fn.name}",
                )
                if entry:
                    assets.append(entry)

        graphs_dir = data_dir / "graphs"
        if graphs_dir.exists():
            for fn in sorted(graphs_dir.glob("*.json")):
                entry = catalog_asset(
                    name=f"graph_{fn.stem}",
                    path=fn,
                    category="raw_graph",
                    staging_target="github",
                    description=f"Raw AST call graph representation: {fn.name}",
                )
                if entry:
                    assets.append(entry)

    # 2. Checkpoints and Adapters
    outputs_dir = REPO_ROOT / "outputs"
    if outputs_dir.exists():
        ckpt_paths = set(outputs_dir.rglob("*.pt")) | set(outputs_dir.rglob("*.pt.gz"))
        for pt in sorted(ckpt_paths):
            ckpt_name = pt.name[:-7] if pt.name.endswith(".pt.gz") else pt.stem
            entry = catalog_asset(
                name=ckpt_name,
                path=pt,
                category="checkpoint",
                staging_target="huggingface_kaggle",
                description=f"Trained model checkpoint: {pt.name}",
            )
            if entry and not any(a["path"] == entry["path"] for a in assets):
                assets.append(entry)

    # Parent frozen ImageDecoder weights
    parent_decoder = PARENT_REPO / "rendered" / "arc3" / "image_decoder" / "s2_vitconv" / "decoder.pt"
    if parent_decoder.exists():
        entry = catalog_asset(
            name="frozen_vitconv_decoder_s2",
            path=parent_decoder,
            category="checkpoint",
            staging_target="google_drive",
            description="Frozen 34.15M ViT+Conv ImageDecoder weights mapping h_v -> 768x768 RGB",
        )
        if entry:
            assets.append(entry)

    # 3. Evaluation Predictions & Verification Reports
    if outputs_dir.exists():
        for rpt in sorted(outputs_dir.rglob("*.json")):
            if rpt == DEFAULT_MANIFEST:
                continue
            entry = catalog_asset(
                name=f"report_{rpt.stem}",
                path=rpt,
                category="evaluation_report",
                staging_target="github",
                description=f"Experimental evaluation report: {rpt.name}",
            )
            if entry:
                assets.append(entry)

        for prd in sorted(outputs_dir.rglob("*.npy")):
            entry = catalog_asset(
                name=f"pred_{prd.stem}",
                path=prd,
                category="prediction_array",
                staging_target="huggingface_kaggle",
                description=f"Model predicted latent tensor dump: {prd.name}",
            )
            if entry:
                assets.append(entry)

    # 4. Standalone Packages & Tooling
    packages = [
        ("dgg_latent_pkg", REPO_ROOT / "src" / "dgg_latent" / "pyproject.toml", "Core continuous latent diffusion library"),
        ("dgg_codebook_pkg", REPO_ROOT / "src" / "dgg_codebook" / "pyproject.toml", "Vocabulary distribution derivation library"),
        ("dgg_vit_decoder_pkg", REPO_ROOT / "src" / "dgg_vit_decoder" / "pyproject.toml", "VisionStack and ViT+Conv decoder library"),
    ]
    for pkg_name, pkg_path, desc in packages:
        if pkg_path.exists():
            entry = catalog_asset(
                name=pkg_name,
                path=pkg_path,
                category="software_package",
                staging_target="github",
                description=desc,
            )
            if entry:
                assets.append(entry)

    # 4b. Distribution Wheels
    wheels_dir = outputs_dir / "m4" / "wheels"
    if wheels_dir.exists():
        for whl in sorted(wheels_dir.glob("*.whl")):
            entry = catalog_asset(
                name=whl.name,
                path=whl,
                category="distribution_wheel",
                staging_target="huggingface_kaggle",
                description=f"Built standalone wheel package: {whl.name}",
            )
            if entry and not any(a["path"] == entry["path"] for a in assets):
                assets.append(entry)

    # 5. Upstream Patches and Specifications
    patch_file = REPO_ROOT / "src" / "patches" / "0001-fix-diffusion-gemma-prefill-multimodal-masks.patch"
    if patch_file.exists():
        entry = catalog_asset(
            name="upstream_mask_patch",
            path=patch_file,
            category="patch",
            staging_target="github",
            description="Upstream Transformers PR patch fixing multimodal attention mask prefill bug",
        )
        if entry:
            assets.append(entry)

    shim_file = REPO_ROOT / "src" / "dg_attn_fix.py"
    if shim_file.exists():
        entry = catalog_asset(
            name="dg_attn_fix_shim",
            path=shim_file,
            category="patch_shim",
            staging_target="github",
            description="Runtime monkey-patch shim installing multimodal attention mask fix",
        )
        if entry:
            assets.append(entry)

    specs = ["PROJECT.md", "PREREG.json", "HANDOFF.md", "GOALS.md"]
    for spec_fn in specs:
        p = REPO_ROOT / spec_fn
        if p.exists():
            entry = catalog_asset(
                name=f"spec_{p.stem.lower()}",
                path=p,
                category="specification",
                staging_target="github",
                description=f"Project research specification: {spec_fn}",
            )
            if entry:
                assets.append(entry)

    return assets


def build_manifest(assets: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble master manifest dictionary."""
    total_bytes = sum(a["size_bytes"] for a in assets)
    by_target: Dict[str, int] = {}
    by_category: Dict[str, int] = {}

    for a in assets:
        by_target[a["staging_target"]] = by_target.get(a["staging_target"], 0) + 1
        by_category[a["category"]] = by_category.get(a["category"], 0) + 1

    return {
        "manifest_version": "1.0.0",
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project": "DiffusionGemma Latent Code Graphs",
        "competition_track": "Kaggle Best New Resource Track",
        "total_assets": len(assets),
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "total_size_gb": round(total_bytes / (1024 * 1024 * 1024), 3),
        "target_counts": by_target,
        "category_counts": by_category,
        "assets": assets,
    }


def stage_to_external(
    assets: List[Dict[str, Any]],
    stage_dir: Path,
    verify: bool = True,
) -> None:
    """Stage files to external NVMe drive with verification."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    print(f"Staging {len(assets)} assets to {stage_dir}...")

    manifest_copy = []

    for idx, asset in enumerate(assets, 1):
        src_path = Path(asset["absolute_path"])
        rel_path = asset["path"]
        dst_path = stage_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if not dst_path.exists() or dst_path.stat().st_size != asset["size_bytes"]:
            print(f"[{idx}/{len(assets)}] Copying {rel_path} ({asset['size_bytes'] / (1024 * 1024):.2f} MB)...")
            shutil.copy2(src_path, dst_path)
        else:
            print(f"[{idx}/{len(assets)}] Up to date: {rel_path}")

        if verify:
            staged_sha = compute_sha256(dst_path)
            if staged_sha != asset["sha256"]:
                raise RuntimeError(
                    f"Integrity check failed for staged file {dst_path}: expected {asset['sha256']}, got {staged_sha}"
                )

        staged_entry = dict(asset)
        staged_entry["staged_path"] = str(dst_path)
        manifest_copy.append(staged_entry)

    # Write external manifest
    ext_manifest_path = stage_dir / "manifest.json"
    manifest_data = build_manifest(manifest_copy)
    with open(ext_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    print(f"External staging manifest written to {ext_manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage release bundle and generate distribution manifest.")
    parser.add_argument("--manifest-out", "-o", type=Path, default=DEFAULT_MANIFEST, help="Output manifest path")
    parser.add_argument("--stage-dir", "-s", type=Path, default=DEFAULT_EXTERNAL_DIR, help="External staging directory")
    parser.add_argument("--sync", action="store_true", help="Copy files to external staging directory")
    parser.add_argument("--verify", action="store_true", help="Verify SHA256 checksums of staged files")
    args = parser.parse_args()

    print("Scanning project assets...")
    assets = collect_assets()
    manifest_data = build_manifest(assets)

    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Wrote master manifest to {args.manifest_out}:")
    print(f"  - Total assets: {manifest_data['total_assets']}")
    print(f"  - Total size:   {manifest_data['total_size_mb']} MB ({manifest_data['total_size_gb']} GB)")
    print(f"  - Target counts: {manifest_data['target_counts']}")

    if args.sync:
        if args.stage_dir.parent.exists() and os.access(args.stage_dir.parent, os.W_OK):
            stage_to_external(assets, args.stage_dir, verify=args.verify)
        else:
            print(f"Warning: External staging parent {args.stage_dir.parent} not writable or mounted. Skipping copy.", file=sys.stderr)


if __name__ == "__main__":
    main()
