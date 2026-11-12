#!/usr/bin/env python3
"""Upload raw .npy prediction arrays, decoded grid .npy arrays, emb_train.npz,
and an updated verifiable Dataset Card (README.md) to Hugging Face:
`bumblebeer/dgg2-code-graph-benchmarks`.

Supports incremental re-runs as final evaluation arms (click s1337, gemma4_ar_*,
Exp B s1337 / frozen_dg) complete.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
from huggingface_hub import CommitOperationAdd, HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "cg"))
from cg_hv_probe import load_model, read_grids  # noqa: E402

DATASET_REPO = "bumblebeer/dgg2-code-graph-benchmarks"

DATASET_CARD_MD = r"""---
license: apache-2.0
tags:
- diffusiongemma
- gemma4
- code-graphs
- code-refactoring
- latent-diffusion
- ast
pretty_name: DiffusionGemma 2 Latent Code Graph Benchmarks & Raw Predictions
size_categories:
- 1K<n<10K
---

# DiffusionGemma 2: Latent Code-Graph Benchmarks & Raw Predictions

Standardized AST code-graph benchmark packs, continuous symbol embeddings, pre-registration (`PREREG.json`), linear evaluation probes, and **raw `.npy` model prediction arrays** for independent CPU rescoring of ***Latent Roadmaps: DiffusionGemma State Transitions for Code-Graph Refactoring***.

- **Author:** `bumblebeers` ([GitHub](https://github.com/bumblebeers))
- **License:** Apache-2.0
- **Code & Scoring Harness:** [bumblebeers/dgg2-latent-code-graphs](https://github.com/bumblebeers/dgg2-latent-code-graphs)
- **Trained LoRA Adapters & Readout Heads:** [bumblebeer/dgg2-latent-code-graphs-adapters](https://huggingface.co/bumblebeer/dgg2-latent-code-graphs-adapters)
- **Standalone Libraries:** [dgg-latent](https://github.com/bumblebeers/dgg-latent) | [dgg-codebook](https://github.com/bumblebeers/dgg-codebook) | [dgg-vit-decoder](https://github.com/bumblebeers/dgg-vit-decoder)

---

## Repository Contents

### 1. Experiment A: 16×16 Visual Adjacency Grid Benchmark Packs (`data/gridA_*`)
- **`data/gridA_train1608.npz`**: 1,608 16-symbol Python AST subgraphs extracted across 3 training repositories (`fastapi`, `requests`, `rich`) with 22 fixed candidate states and 3 open-ended edit families.
- **`data/gridA_test360_httpx.npz`**: 360 held-out 16-symbol AST subgraphs from `encode/httpx` across seen, shuffled, unseen paraphrase, indirect intent, held-out single-pass, 2-step re-feeding, 3-step chain, and open-ended manifests.
- **`data/gridA_test360_click.npz`**: 360 held-out 16-symbol AST subgraphs from `pallets/click` for zero-shot cross-repository transfer.
- **`data/gridA_n32_*.npz` & `data/gridA_n48_*.npz`**: Multi-resolution 32×32 and 48×48 subgraph packs testing sub-patch packing limits (Gate A1).
- **`data/*.nodes.json`**: Per-subgraph AST symbol and line metadata.
- **`reader.npz`** (`outputs/m1/reader.npz`): Gate A1 closed-form Ridge probe mapping clean SigLIP latents $h_v \in \mathbb{R}^{256 \times 1152}$ to the 8 discrete palette classes (`1.000` clean cell accuracy across all 8 classes).

### 2. Experiment B: 256-Symbol Continuous Embedding Packs (`data/nodeB_*`)
- **`data/nodeB_train1608.json.gz`**: 1,608 256-symbol AST windows for single-site and multi-site symbol refactoring.
- **`data/nodeB_test360_httpx.json.gz`** & **`data/nodeB_test360_click.json.gz`**: Held-out `httpx` ($N=360$) and zero-shot `click` ($N=360$) evaluation packs (`chance = 12.5%`, `mean-vector floor = 25.0%`).
- **`data/nodeB_test360_httpx_unseenargs.json.gz`** & **`data/nodeB_test180_httpx_renameclass.json.gz`**: Unseen argument and class-rename splits.
- **`data/nodeB_multi_test360_httpx_k2.json.gz`** & **`k4.json.gz`**: 2-site and 4-site simultaneous and re-fed symbol-editing splits.
- **`emb.npz`** & **`emb_train.npz`**: Frozen `sentence-transformers/all-MiniLM-L6-v2` ($d=384$) symbol embeddings and train-only standardization statistics ($\mu, \sigma$).

### 3. Raw Model Prediction Arrays (`predictions/` & `outputs/m2/decoded_grids_all_arms.npz`)
To enable full independent verification without requiring a 26B GPU, this dataset hosts **both** continuous latent `.npy` outputs and Ridge-decoded discrete `(M, 16, 16)` `uint8` grid `.npy` files:
- **`predictions/expA_hv/<arm>/preds_*.npy`**: Raw continuous predicted SigLIP latent tensors $\hat{h}_v \in \mathbb{R}^{360 \times 256 \times 1152}$ (`float32`) emitted by `stock_generate_with_readout` for `cg_lora_dg_seed42_ep3`, `cg_lora_dg_seed1337_ep3`, `cg_frozen_dg_ep3`, zero-shot `click`, and `NonDGTransformBaseline`.
- **`predictions/expA_grids/<arm>/*.npy`** & **`outputs/m2/decoded_grids_all_arms.npz`**: Decoded `(360, 16, 16)` `uint8` palette grids (`0..7`) across all evaluated arms (including LoRA DG, Frozen DG, Non-DG Frozen/EncLoRA, and Gemma 4 AR baselines as completed) for instant lightweight CPU rescoring.
- **`predictions/expB_vec/<arm>/*.npy`**: Raw standardized continuous symbol-vector predictions $\hat{z} \in \mathbb{R}^{M \times 256 \times 384}$ (`float32`) for Experiment B (`cg_expB_lora_dg_*` and `NonDGTransformBaseline`).

---

## Independent CPU Rescoring Quickstart

Clone the scoring harness from [bumblebeers/dgg2-latent-code-graphs](https://github.com/bumblebeers/dgg2-latent-code-graphs) and download this dataset (or individual `.npy` files):

```bash
git clone https://github.com/bumblebeers/dgg2-latent-code-graphs.git
cd dgg2-latent-code-graphs

# Rescore Experiment A directly from raw continuous h_v (.npy, shape 360x256x1152) + Ridge probe:
python3 cg/cg_score_preds.py \\
  --pack data/gridA_test360_httpx.npz \\
  --reader reader.npz \\
  --manifest seen_bal360 \\
  --pred predictions/expA_hv/cg_lora_dg_seed42_ep3/preds_cg_lora_dg_seed42_ep3_seen_bal360.npy

# Or rescore instantly from the decoded (360, 16, 16) uint8 palette grid (.npy):
python3 cg/cg_score_preds.py \\
  --pack data/gridA_test360_httpx.npz \\
  --manifest refeed_heldout_comp_bal360 \\
  --grids predictions/expA_grids/cg_lora_dg_seed42_ep3/preds_cg_lora_dg_seed42_ep3_refeed_heldout_comp_bal360.npy

# Rescore Experiment B raw (360, 256, 384) symbol vectors (.npy):
python3 -c "
import gzip, json, numpy as np, sys
sys.path.insert(0, 'cg')
from cg_node_gate import Emb
from cg_node_metrics import Evaluator, default_items
ds = json.load(gzip.open('data/nodeB_test360_httpx.json.gz', 'rt'))
ev = Evaluator(ds, Emb.load('emb.npz'), Emb.load('emb_train.npz').vec)
pred = np.load('predictions/expB_vec/cg_expB_lora_dg_seed42_ep3/preds_cg_expB_lora_dg_seed42_ep3_instr_seen.npy')
res = ev.score(pred, default_items(ds))
print({k: round(res[k], 4) for k in ('site_hit', 'all_sites_hit', 'top_is_site', 'drift')})
"
```
"""


def build_decoded_grid_npys(grid_stage_dir: Path) -> List[Tuple[Path, str]]:
    """Decode every continuous Exp A .npy into a lightweight (M, 16, 16) uint8 .npy file,
    and gather Non-DG / Gemma 4 AR grid .npy files."""
    reader = load_model(str(REPO_ROOT / "reader.npz"))
    ops: List[Tuple[Path, str]] = []

    # 1. DG arms on httpx
    for arm_dir in sorted((REPO_ROOT / "outputs" / "m2" / "arms").glob("*")):
        if not arm_dir.is_dir():
            continue
        out_arm = grid_stage_dir / arm_dir.name
        out_arm.mkdir(parents=True, exist_ok=True)
        for npy in sorted(arm_dir.glob("*.npy")):
            arr = np.load(npy, mmap_mode="r")
            out_f = out_arm / npy.name
            if arr.ndim == 3 and arr.shape[1:] == (256, 1152):
                if not out_f.exists():
                    g = read_grids(reader, np.asarray(arr)).astype(np.uint8)
                    np.save(out_f, g)
                ops.append((out_f, f"predictions/expA_grids/{arm_dir.name}/{npy.name}"))
            elif arr.ndim == 3 and arr.shape[1:] == (16, 16):
                ops.append((npy, f"predictions/expA_grids/{arm_dir.name}/{npy.name}"))

    # 2. DG arms on click
    click_dir = REPO_ROOT / "outputs" / "m2" / "click"
    if click_dir.exists():
        out_click = grid_stage_dir / "click_dg"
        out_click.mkdir(parents=True, exist_ok=True)
        for npy in sorted(click_dir.glob("preds_*.npy")):
            out_f = out_click / npy.name
            if not out_f.exists():
                arr = np.load(npy)
                g = read_grids(reader, arr).astype(np.uint8)
                np.save(out_f, g)
            ops.append((out_f, f"predictions/expA_grids/click_dg/{npy.name}"))

    # 3. Non-DG grid .npy files (httpx & click)
    m2_preds = REPO_ROOT / "outputs" / "m2" / "preds"
    if m2_preds.exists():
        for npy in sorted(m2_preds.glob("pred_grids_*.npy")):
            if "_hel_heldout_" in npy.name:
                continue
            ops.append((npy, f"predictions/expA_grids/nondg/{npy.name}"))

    return ops


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["grids_and_card", "headline", "all"], default="grids_and_card")
    args = ap.parse_args()

    token = Path(os.path.expanduser("~/.tokens/HF_TOKEN")).read_text().strip()
    api = HfApi(token=token)

    existing = {s.rfilename for s in api.dataset_info(DATASET_REPO).siblings}
    print(f"[HF] Currently {len(existing)} files on {DATASET_REPO}")

    tmp_dir = Path("/tmp/dgg2_codegraph/hf_pred_stage")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    readme_path = tmp_dir / "README.md"
    readme_path.write_text(DATASET_CARD_MD)

    grid_ops = build_decoded_grid_npys(tmp_dir / "expA_grids")
    print(f"[Stage] Prepared {len(grid_ops)} lightweight (M,16,16) grid .npy files")

    # Commit 1: README.md + emb_train.npz + decoded_grids_all_arms.npz + all expA_grids/*.npy
    # + headline continuous .npy files (seen, shuf, unseen_para, unseen_infer, refeed, chain, click, Exp B DG)
    c1_ops: List[CommitOperationAdd] = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(readme_path)),
    ]
    emb_train = REPO_ROOT / "emb_train.npz"
    if emb_train.exists() and "emb_train.npz" not in existing:
        c1_ops.append(CommitOperationAdd(path_in_repo="emb_train.npz", path_or_fileobj=str(emb_train)))

    dec_npz = REPO_ROOT / "outputs" / "m2" / "decoded_grids_all_arms.npz"
    if dec_npz.exists():
        c1_ops.append(CommitOperationAdd(path_in_repo="outputs/m2/decoded_grids_all_arms.npz", path_or_fileobj=str(dec_npz)))

    for local_p, repo_p in grid_ops:
        if repo_p not in existing:
            c1_ops.append(CommitOperationAdd(path_in_repo=repo_p, path_or_fileobj=str(local_p)))

    headline_tags = (
        "seen_bal360", "seen_shuf360", "unseen_para_bal360", "unseen_infer_360",
        "refeed_heldout_comp_bal360", "refeed_chain_heldout_ops_360",
    )
    c2_candidates: List[Tuple[Path, str]] = []

    if args.stage != "grids_and_card":
        # Exp A continuous h_v .npy files
        for arm_dir in sorted((REPO_ROOT / "outputs" / "m2" / "arms").glob("cg_*")):
            for npy in sorted(arm_dir.glob("preds_*.npy")):
                repo_p = f"predictions/expA_hv/{arm_dir.name}/{npy.name}"
                if repo_p in existing:
                    continue
                if any(t in npy.name for t in headline_tags):
                    c1_ops.append(CommitOperationAdd(path_in_repo=repo_p, path_or_fileobj=str(npy)))
                else:
                    c2_candidates.append((npy, repo_p))

        # Click continuous h_v .npy files
        for npy in sorted((REPO_ROOT / "outputs" / "m2" / "click").glob("preds_*.npy")):
            repo_p = f"predictions/expA_hv/click_dg/{npy.name}"
            if repo_p not in existing:
                c1_ops.append(CommitOperationAdd(path_in_repo=repo_p, path_or_fileobj=str(npy)))

        # Non-DG continuous sample h_v .npy
        nondg_hv = REPO_ROOT / "outputs" / "m2" / "preds" / "nondg_enc_lora_s42_seen_bal360.npy"
        if nondg_hv.exists() and "predictions/expA_hv/nondg/nondg_enc_lora_s42_seen_bal360.npy" not in existing:
            c1_ops.append(CommitOperationAdd(
                path_in_repo="predictions/expA_hv/nondg/nondg_enc_lora_s42_seen_bal360.npy",
                path_or_fileobj=str(nondg_hv),
            ))

        # Exp B DG continuous symbol vector .npy files
        for arm_dir in sorted((REPO_ROOT / "outputs" / "m3" / "arms").glob("cg_expB_*")):
            for npy in sorted(arm_dir.glob("preds_*.npy")):
                if npy.is_symlink():
                    continue
                repo_p = f"predictions/expB_vec/{arm_dir.name}/{npy.name}"
                if repo_p in existing:
                    continue
                c1_ops.append(CommitOperationAdd(path_in_repo=repo_p, path_or_fileobj=str(npy)))

        # Exp B Non-DG continuous symbol vector .npy files (go in c2 batches)
        for npy in sorted((REPO_ROOT / "outputs" / "m3" / "preds").glob("pred_vecs_nondg_*.npy")):
            repo_p = f"predictions/expB_vec/nondg/{npy.name}"
            if repo_p not in existing:
                c2_candidates.append((npy, repo_p))

    print(f"[HF] Committing Stage 1 ({len(c1_ops)} operations: Dataset Card + grid .npy + headline continuous .npy + Exp B DG .npy)...")
    api.create_commit(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        operations=c1_ops,
        commit_message="Upload raw .npy continuous & decoded grid predictions, emb_train.npz, and updated rescoring Dataset Card",
    )
    print("[HF] Stage 1 commit LIVE!")

    if args.stage == "all" and c2_candidates:
        batch_size = 20
        for b_start in range(0, len(c2_candidates), batch_size):
            chunk = c2_candidates[b_start : b_start + batch_size]
            ops = [CommitOperationAdd(path_in_repo=rp, path_or_fileobj=str(lp)) for lp, rp in chunk]
            print(f"[HF] Committing Stage 2 batch {b_start // batch_size + 1} ({len(ops)} .npy files)...")
            api.create_commit(
                repo_id=DATASET_REPO,
                repo_type="dataset",
                operations=ops,
                commit_message=f"Upload remaining held-out, open-ended, and baseline raw .npy predictions (batch {b_start // batch_size + 1})",
            )
        print("[HF] All Stage 2 batches LIVE!")


if __name__ == "__main__":
    main()
