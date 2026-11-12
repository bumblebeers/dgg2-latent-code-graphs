# External Review Archive — Latent Roadmaps: DiffusionGemma World Models for Code-Graph Refactoring

- **Track:** Kaggle Gemma 4 Developer Agent Paper Track (`https://www.kaggle.com/competitions/gemma-4-developer-agent-paper`)
- **Author Handle:** `bumblebeers`
- **Hard Competition Deadline:** `2026-11-12T23:59:00Z` (`17:59 CST`)
- **Snapshot Timestamp (UTC):** `2026-11-12T19:50:00Z`

---

## 1. Quick-Start Orientation for Reviewers

1. **Read the Competition Rules, Rubric & Review Questions First (`5 min`):**
   - [`KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md`](KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md) (also in [`docs/KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md`](docs/KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md)) — Verbatim Kaggle competition scope, the 5 equally weighted `0–5` judging criteria (`Novelty`, `Quality`, `Relevance`, `Verifiability`, `Clarity`), the 3 prize categories (`Overall Best Paper`, `Best New Resource`, `Best New Application`), UI character/pixel limits, and 5 targeted questions where reviewer feedback will help most.
2. **Review the Submission Writeup & Focal Visuals (`15 min`):**
   - **Main Submission Writeup:** [`docs/paper_draft.md`](docs/paper_draft.md) (`2,405` whitespace words; `2,978 / 3,000` strict `\b\w+\b` regex word tokens).
   - **Kaggle `560 × 280` Card & Thumbnail Image:** [`docs/figures/kaggle_card_thumbnail_560x280.png`](docs/figures/kaggle_card_thumbnail_560x280.png).
   - **Paper Hero Figure (`1920 × 1080`):** [`docs/figures/figure1_hero_multimodal_roadmap.png`](docs/figures/figure1_hero_multimodal_roadmap.png).
   - **Complete Tabular & Statistical Ledger:** [`docs/PAPER_DATA_SUMMARY.md`](docs/PAPER_DATA_SUMMARY.md).
3. **Audit Empirical Evidence, Code, Libraries & Tests (`15–30 min`):**
   - Every number in [`docs/paper_draft.md`](docs/paper_draft.md) traces directly to machine-readable JSON reports in `outputs/` and `parent_sec26_27_reference/` (see **Section 3** below).
   - All pre-registration hashes, unit tests, E2E rubric checks, and demo notebook generation checks run locally on CPU in `< 5 seconds` (see **Section 5** below).

---

## 2. Live Snapshot Status (Settled vs. Actively Completing on GPU Fleet)

### 100% Complete, Offline-Verified & Included in This Archive
1. **Natural-Image Foundation (`§3`, `N=300` Held-Out DOCCI, Stock `last_sc` Generation Readout):**
   - All positive claims (`C01–C05`, `C08–C11a`, `C12a`, `C12c`, `C14`) and negative/retracted claims (`C06` sub-patch invariance, `C07` single-pass geo×geo collapse, `C11b` `darken` failure, `C13` luminance LayerNorm failure) included in `parent_sec26_27_reference/paper_data_package/`.
2. **Pre-Registered Empirical Gates (`§4.1`, `PREREG.json`):**
   - Gate A1 (`16² = 100.0%` all 8 classes; `32²` & `48²` sparse-class degradation `selected = 13.15%`), Gate A2 (`0.8757` median max candidate cosine), Gate A3 (`90.0%` changed-cell Non-DG EncLoRA), Gate B0 (`all-MiniLM-L6-v2` `PR = 47.83`, `cent_cos = -0.0015`, `Oracle = 1.0`), and Gate B1 (`52.22%` Non-DG delete slot hit).
3. **Experiment A — AST Visual Adjacency Grids (`§5.1–§5.4`, `httpx` `N=360`, 16 Manifests):**
   - **100% complete across both LoRA DG seeds (`42` & `1337`), Frozen DG, and 2-seed Non-DG controls (`Frozen Instr` & `Enc LoRA`), plus 2-seed Non-DG secondary-repo `click` (`N=360`):**
     - `seen_bal360`: **`91.94 ± 1.11%`** vs. Non-DG EncLoRA `73.33 ± 0.00%` ($p < 10^{-8}$) and Frozen DG `0.28%`.
     - `seen_shuf360` (Shuffled Control): **`0.56 ± 0.28%`**.
     - `unseen_para_bal360`: **`78.47 ± 3.19%`** vs. `49.17 ± 3.61%` ($p < 10^{-15}$).
     - `unseen_infer_360` (Indirect Developer Intent): **`61.39 ± 2.50%`** vs. `23.19 ± 1.25%` ($p < 10^{-31}$).
     - Commuting 1-pass compounds: **`99.31 ± 0.69%`** (`reverse_order+delete_selected`, $p < 10^{-89}$) and **`98.61 ± 1.39%`** (`reverse_order+delete_tests`) vs. Non-DG `8.06%` / `6.53%`.
     - 2-step soft-token re-feeding (`refeed_heldout_comp_bal360`): **`82.08 ± 0.97%`** (up from `48.00 ± 1.67%` 1-pass; `reverse_edges+keep_callees = 100.0 ± 0.0%`) vs. Non-DG `34.58 ± 2.36%` ($p < 10^{-41}$).
     - 3-step mirror chain (`refeed_chain_heldout_ops_360`): **`90.83 ± 4.17%`** on unseen `keep_callers` (`45.42 ± 2.08%` overall with `mark_ancestors = 0.00%`) vs. **`0.00%`** across all Non-DG and Frozen DG arms ($p < 10^{-51}$).
     - Open-ended refactorings (4 sampler seeds per model seed): **`55.00 ± 5.83%`** seen / **`56.11 ± 5.28%`** unseen vs. Non-DG `25.97 ± 0.42%` / `11.39 ± 1.11%`.
4. **Experiment B — Continuous Symbol Latents (`§5.5`, `httpx` & `click` `N=360`):**
   - Full 2-seed Non-DG baselines across all 9 manifests on `httpx` (`47.4 ± 1.3%` seen, `41.3 ± 4.3%` unseen, $k=2$ re-feed `5.6% → 40.9%`) and `click` (`39.2 ± 3.1%` / `32.8 ± 3.1%`).
   - Seed-42 LoRA DG (`modified model: trained input adapter`) scored across 6 manifests (`outputs/m3/arms/cg_expB_lora_dg_seed42_ep3/expB_scoring_report.json`), establishing the architectural boundary (`98.3%` discrete 3-class canvas accuracy, `100.0%` `delete` recall vs. `2.2%` seen / `4.8%` unseen non-delete accuracy due to `18.0×` unchanged-slot reconstruction drift without a residual skip).
5. **Reusable Engineering Deliverables (`§7`):**
   - Built `.whl` packages (`outputs/m4/wheels/dgg_{latent,codebook,vit_decoder}-0.1.0-py3-none-any.whl`) + full source under `src/dgg_*/`.
   - Formal upstream `transformers` git patch (`src/patches/0001-fix-diffusion-gemma-prefill-multimodal-masks.patch`), runtime shim (`src/dg_attn_fix.py`), and technical note (`docs/UPSTREAM_PR_MASK_FIX.md`).
   - Reproducible Kaggle demo notebook (`notebooks/dgg2_latent_code_graphs_demo.ipynb` built via `src/build_review_notebook.py`).

### Only 4 Narrow `[PENDING: ...]` Slots Remain in `docs/paper_draft.md` (Actively Finishing on Colab GPUs Now)
1. `Table 4, L109`: Frozen DG + Head on Experiment B (`[PENDING: G4]`).
2. `Table 4, L110`: LoRA DG Seed `1337` on Experiment B (`[PENDING: s1337]`).
3. `§5.1, L115`: Zero-shot LoRA DG transfer on secondary repository `click` (`[PENDING: zero-shot DG click eval]`).
4. `§5.5, L127`: Experiment B LoRA DG $k=2,4$ multi-site re-feeding + Seed `1337` + Frozen DG (`[PENDING: k=2/k=4 refeed, s1337, Frozen DG]`).

---

## 3. Claim-to-Artifact Traceability Matrix

| Paper Section / Table | Claim / Metric | Verification Artifact(s) in Archive | Source / Scoring Script |
| :--- | :--- | :--- | :--- |
| **§2 & Stock Contract** | Canvas=256, SC on, frozen SigLIP/`embed_vision`, final-SC readout, `>=6` adapter labels | `parent_sec26_27_reference/DG-STOCK-REFERENCE.md` | `src/cg_colab_runner.py`, `tests/e2e/test_tier4_scenarios.py` |
| **§3 & Table 2** | Natural-image §26–§27 (`N=300` DOCCI, C01–C14, `last_sc` only) | `parent_sec26_27_reference/paper_data_package/claims_ledger.md`, `paper_data_master.json`, `tables/final_headline_tables_n300.md` | Parent `dgg2_sample_stock.py` |
| **§4.1 & Table 3** | Pre-registered Gates A1 (`16/32/48²`), A2, A3, B0, B1 | `PREREG.json`, `outputs/m1/*.json`, `outputs/m2/gateA3_report.json`, `outputs/m2/baselines/nondg_expB/gateB1_report.json` | `cg/cg_prereg.py`, `cg/cg_ridge_reader.py`, `cg/cg_node_metrics.py` |
| **§5.1–§5.3 & Table 4** | Exp A 2-seed LoRA DG (`91.9%` seen, `78.5%` unseen, `61.4%` infer, `82.1%` 2-step, `90.8%` `keep_callers`) | `outputs/m2/offline_verification_report.json`, `outputs/m2/arms/*/cg_*_summary.json`, `outputs/m2/nondg_baseline_manifest_summary.json`, `outputs/m2/nondg_click_summary.json` | `src/cg_colab_runner.py`, `cg/cg_eval_Lock.py`, `src/cg_eval_nondg.py` |
| **§5.4** | Open-ended refactorings across 4 sampler seeds (`55.0%` seen / `56.1%` unseen vs. `26.0%` / `11.4%`) | `outputs/m2/open_predictions_scoring_report.json` | `cg/cg_eval_Lock.py` |
| **§5.5 & Table 4** | Exp B Non-DG (`47.4%` seen; $k=2$ re-feed `40.9%`) vs. DG cold-start drift (`18.0×`, `100%` delete / `2.2%` non-delete) | `outputs/m3/nondg_expB_manifest_summary.json`, `outputs/m3/nondg_expB_click_summary.json`, `outputs/m3/arms/cg_expB_lora_dg_seed42_ep3/expB_scoring_report.json` | `cg/cg_node_metrics.py`, `src/cg_train_nondg_expB.py`, `src/cg_colab_runner.py` |
| **§7** | `dgg_latent`, `dgg_codebook`, `dgg_vit_decoder`, `dg_attn_fix`, 931-asset release manifest | `outputs/m4/wheels/*.whl`, `src/dgg_*/`, `src/patches/0001-*.patch`, `src/dg_attn_fix.py`, `outputs/m4/staging_manifest.json` | `src/stage_release_bundle.py`, `tests/unit/test_dgg_*.py` |

---

## 4. Public External Release Status (Pre-Staged vs. Held Pending Final Sign-Off)

- **Live / Publicly Pre-Staged Now:**
  - **Hugging Face Benchmark Dataset:** `https://huggingface.co/datasets/bumblebeer/dgg2-code-graph-benchmarks` (34 static benchmark packs + Dataset Card)
  - **Kaggle Benchmark Dataset:** `https://www.kaggle.com/datasets/unclebumblebeer/dgg2-code-graph-benchmarks` (`status: ready`)
  - **Hugging Face Adapters & Readouts Repo:** `https://huggingface.co/bumblebeer/dgg2-latent-code-graphs-adapters` (uploading final LFS checkpoint pack)
  - **GitHub `transformers` Fork Branch:** `https://github.com/bumblebeers/transformers/tree/fix/diffusion-gemma-prefill-multimodal-masks`
- **Prepared Locally & Held Until Final Run Harvest + Author Go-Ahead:**
  - **GitHub Source Repository & `v0.1.0` Wheels Release:** `https://github.com/bumblebeers/dgg2-latent-code-graphs`
  - **Kaggle Public Demo Notebook:** `https://www.kaggle.com/code/unclebumblebeer/dgg2-latent-code-graphs-demo` (`notebooks/dgg2_latent_code_graphs_demo.ipynb`)
  - **Upstream Pull Request to `huggingface/transformers`:** Prepared in `docs/UPSTREAM_PR_MASK_FIX.md` + `src/patches/0001-fix-diffusion-gemma-prefill-multimodal-masks.patch`.

---

## 5. Offline CPU Verification Commands (Run Inside Unpacked Root)

No GPU or 26B model weights are needed to verify pre-registration hashes, unit/E2E tests, rubric word-count constraints, notebook determinism, or archive checksums:

```bash
# 1. Verify SHA-256 integrity of every file in this archive
sha256sum -c SHA256SUMS

# 2. Verify cryptographic lock on pre-registered hypotheses & gates
python3 cg/cg_prereg.py --check

# 3. Verify reproducible Kaggle demo notebook matches generator byte-for-byte
python3 src/build_review_notebook.py --check

# 4. Run unit & end-to-end test suite (includes <=3000 word-count, citation & stock-contract checks)
pytest -q tests/unit tests/e2e
```

---

## 6. Archive Directory Layout

- `README_EXTERNAL_REVIEW.md` — This guide.
- `KAGGLE_COMPETITION_BRIEF_AND_RUBRIC.md` — Official Kaggle competition brief, 5-criterion rubric, 3 prize tracks, UI constraints, and review questions.
- `docs/` — `paper_draft.md` (primary submission writeup), `figures/` (`kaggle_card_thumbnail_560x280.png`, `figure1_hero_multimodal_roadmap.png`), `PAPER_DATA_SUMMARY.md`, `PAPER_BLUEPRINT.md`, `RELATED_WORKS.md`, `UPSTREAM_PR_MASK_FIX.md`.
- `PREREG.json`, `PROJECT.md`, `GOALS.md`, `DEAD_ENDS.md`, `TEST_INFRA.md`, `AGENTS.md`, `pyproject.toml` — Pre-registration, design records, dead-ends ledger, and repo metadata.
- `cg/` — Core AST graph extraction, deterministic 16×16 palette renderer, Ridge cell reader, anisotropy-aligned candidate scorer, and pre-registration checker.
- `src/` — Standalone packages (`dgg_latent/`, `dgg_codebook/`, `dgg_vit_decoder/`), upstream patch & runtime shim (`patches/`, `dg_attn_fix.py`), Colab/Non-DG training & evaluation runners, figure builders, notebook builder, and release bundler.
- `tests/` — Unit (`tests/unit/`) and end-to-end rubric/pipeline tests (`tests/e2e/`).
- `notebooks/` — Generated Kaggle public demo notebook (`dgg2_latent_code_graphs_demo.ipynb`) and `kernel-metadata.json`.
- `data/` — Compact AST source subgraphs (`data/graphs/`), Experiment A 16²/32²/48² grid packs (`data/gridA_*`), and Experiment B 256-symbol packs (`data/nodeB_*.json.gz`). *(Multi-GB cached `h_v` tensors and raw `.npy` prediction arrays are omitted to keep the archive `< 100 MB`; their SHA-256 digests are cataloged in `outputs/m4/staging_manifest.json`.)*
- `outputs/` — All Milestone 1–4 JSON evaluation summaries, gate reports, offline verification reports, `.whl` artifacts (`outputs/m4/wheels/`), and `reader*.npz` probes.
- `parent_sec26_27_reference/` — Read-only snapshot of `DG-STOCK-REFERENCE.md` and the natural-image `paper_data_package/` (`claims_ledger.md`, `not_run_and_failures.md`, `paper_data_master.json`, `tables/`, `figures/`) backing §2–§3.
- `MANIFEST.json` & `SHA256SUMS` — Machine-readable inventory and cryptographic SHA-256 checksums of all archived files.
