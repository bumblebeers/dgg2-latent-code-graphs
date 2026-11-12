# Title: Latent Roadmaps: DiffusionGemma State Transitions for Code-Graph Refactoring

### Subtitle: We adapt DiffusionGemma's bidirectional canvas to execute multi-step AST code-graph refactoring transitions in latent space.

**Author:** `bumblebeers` | **Track:** Kaggle Gemma 4 Developer Agent Paper Track

---

## 0. Abstract

Autoregressive coding agents emit tokens left-to-right before downstream call-graph consequences are resolved. Toward structural pre-computation, we study **Latent Roadmaps**: executing user-specified AST call-graph transitions on DiffusionGemma's (DG 26B-A4B) bidirectional canvas, evaluating graph fidelity without syntax emission or autonomous decomposition. Extending a natural-image study ($N=300$ DOCCI, 91.3% seen / 90.5% unseen 24-way), we test (1) 16-symbol AST call graphs rendered as 16×16 palette grids (768² RGB, 1 cell = 1 soft token) and (2) 256-symbol windows embedded via frozen `all-MiniLM-L6-v2`. Reporting **22-way nearest-candidate match** and **exact 256-cell grid match** on held-out `httpx` ($N=360$), 2-seed LoRA DG achieves 91.9±1.1% 22-way (62.2±4.2% exact) seen, 78.5±3.2% (47.1±3.8% exact) on paraphrases, and 61.4±2.5% (41.2±1.2% exact, bimodal) on indirect intent—beating the strongest scratch ~50M Non-DG transformer per split by +16.2 to +38.2 pp, while Non-DG's seen exact match (7.2%) equals unmodified source copy (7.2%). Under caller-supplied decomposition, **soft-token re-feeding** reaches 82.1±1.0% 22-way (47.8±2.2% exact, 98.0% edge F1) on 2-step compounds and 90.8±4.2% 22-way (0.0% exact, 73.1% edge F1) on 3-step `keep_callers` chains (Non-DG: 0.0%). Zero-shot `click` transfer ($N=360$, s42) reaches 88.3% seen (61.4% exact) and 72.5% unseen. On ambiguous edits, DG commits deterministically (~1.01 distinct grids across 4 seeds; 56.1% unseen vs. Non-DG 11.4%). Experiment B suggests reconstructing 384-d symbol vectors across a 3-symbol canvas vocabulary (`_`/`+`/`-`) without a residual skip causes 18.0× unchanged-slot drift (s42: 26.7% seen; 100% delete vs. 2.2% non-delete).

---

## 1. Introduction & Scope

Autoregressive code models generate syntax left-to-right under causal masking, committing to early signatures before downstream callers and transitive dependencies are updated. Engineers instead reason over dependency graphs—identifying which nodes to remove, redirect, or isolate—as a structural specification.

We test whether a discrete diffusion language model can execute **structural state transitions** directly on a 256-token latent canvas: given a source call graph (or symbol-embedding window) and an instruction, the model predicts the updated latent state. To keep scope exact: **our system executes caller-specified transitions rather than planning decompositions autonomously, and evaluates latent graph states rather than synthesizing Python code.**

![Figure 1: Shared 256-Token Canvas](figures/figure1_hero_multimodal_roadmap.png)

*Figure 1: Shared 256-token SigLIP $h_v$ canvas across natural images (Panel A) and 16×16 AST call graphs (Panel B), caller-decomposed soft-token re-feeding (Panel C), and the Experiment A vs. B architectural boundary (Panel D).*

DG 26B-A4B refines all 256 canvas tokens under **bidirectional self-attention** across up to 48 denoising steps, letting adjacency rows and columns constrain each other jointly and supporting multi-step transitions via soft-token re-feeding.

---

## 2. Background & Representation Architecture

### 2.1 DiffusionGemma Stock Contract & Dual Representations
DG 26B-A4B (3.8B active / 25.2B total MoE) denoises a 256-token canvas under multinomial corruption, self-conditioning ($p=0.5$), block-bidirectional multimodal prefill (`dg_attn_fix`), and final self-conditioned step readout (`stock_generate_with_readout`).
- **Experiment A (Visual Adjacency Grids):** 16-symbol Python AST call graphs render as 16×16 8-color grids at 768² RGB (48 px/cell), yielding $h_v \in \mathbb{R}^{256 \times 1152}$ via frozen SigLIP and `embed_vision`. Pointwise readouts predict $\hat{h}_v$, decoded into 8 cell classes (`background`, `edge`, `func`, `method`, `class`, `test`, `selected`, `marked`) by a closed-form Ridge reader fitted on clean $h_v$.
- **Experiment B (Continuous Symbol Latents):** 256-symbol AST stubs embedded via frozen `all-MiniLM-L6-v2` ($d=384$) enter DG through a $\text{LayerNorm}(384) + \text{Linear}(384, 2816)$ projector (**modified model: trained input adapter**, our sole non-stock input interface), paired with a 3-symbol canvas vocabulary (`_` unchanged, `+` edited, `-` deleted) and pointwise 384-d readout.

### 2.2 Anisotropy Alignment
Because raw SigLIP $h_v$ and pooled states occupy narrow cones ($\cos \approx 0.30\text{–}0.99$, participation ratio $\text{PR} \approx 2.4$), we subtract the training-set mean (`cent_cos`) and Z-standardize channels before candidate lookups.

|Dimension|Natural Images|Exp A: Adjacency Grids|Exp B: Symbol Latents|
|:---|:---|:---|:---|
|Layout|DOCCI 768² (16×16 patches)|16-node AST (16×16 cells)|256-symbol AST window|
|Encoder|Frozen SigLIP|Frozen SigLIP|Frozen `all-MiniLM-L6-v2`|
|Latents|256×1152 ($h_v$) → 2816|256×1152 ($h_v$) → 2816|256×384 ($z$) → 2816|
|Input Path|Frozen `embed_vision`|Frozen `embed_vision`|Trained `LayerNorm+Linear`|
|Evaluation|24-way $h_v$ cosine & PSNR|22-way & Exact-grid match|8-way centered cosine|

*Table 1: Cross-domain representation and evaluation summary.*

---

## 3. Empirical Foundation: Natural-Image Patch Transitions

We first benchmarked DG on $N=300$ held-out DOCCI photographs across a 24-state geometric and photometric pool (`chance = 4.2%`).

### 3.1 Natural-Image Transition Results
2-seed LoRA DG ($r=64$, ~150M trainable parameters) outperforms scratch 53.6M Non-DG transformers on identity reconstruction (17.28 dB PSNR, 100.0% R@1 vs. Frozen DG 12.41 dB, 7.3%), 6-way geometry (100.0% seen, 89.5% unseen vs. 55.8%), mixed 24-way edits, and held-out `blue_tint` interpolation:

|Manifest ($N=300$, $K=24$)|Dec Ceil|Copy|Frozen DG|Non-DG Frz|Non-DG EncLoRA|**LoRA DG (2-Seed)**|McNemar|
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|`seen_bal300`|100.0%|4.7%|12.7%|74.5 ± 1.5%|74.8 ± 0.5%|**91.3 ± 0.0%**|$p < 10^{-11}$|
|`seen_shuf300` (Shuffled)|100.0%|4.7%|3.0%|2.2 ± 0.5%|1.8 ± 0.2%|**0.0 ± 0.0%**|—|
|`unseen_para_bal300`|100.0%|4.7%|12.3%|32.7 ± 0.3%|61.3 ± 0.3%|**90.5 ± 0.5%**|$p < 10^{-20}$|
|`heldout_blue_tint_300`|100.0%|0.0%|3.7%|0.0 ± 0.0%|0.0 ± 0.0%|**60.2 ± 5.8%**|$p < 10^{-59}$|
|`heldout_flip_h_then_rot90`|100.0%|0.0%|2.3%|0.3 ± 0.3%|0.0 ± 0.0%|**0.5 ± 0.2%**|—|
|`refeed_geogeo_bal300` (2-Step)|100.0%|0.0%|3.7%|67.7 ± 1.3%|65.2 ± 0.8%|**88.3 ± 1.7%**|$p < 10^{-17}$|

*Table 2: Natural-image 24-way nearest-candidate accuracy ($N=300$ DOCCI).*

### 3.2 Structural Boundaries Motivating Experiment A
1. **Sub-Patch Orientation Invariance:** Intact vs. internally rotated patches ($N=250$) yield tied cosines ($0.3291$ vs. $0.3297$, $p = 0.68$), showing macroscopic 16×16 patch routing without sub-patch synthesis and motivating 1-cell-per-patch grids.
2. **Single-Pass Chaining & Opposites:** Unfactored two-step geometric prompts collapse onto the final edit (0.5%), and `brighten`-only training fails on unseen `darken` (8.2%).
3. **Caller-Decomposed Re-Feeding:** Re-feeding $\hat{h}_v^{(1)}$ through frozen `embed_vision` resolves non-commuting pairs (88.3 ± 1.7% geo×geo; 92.0 ± 0.3% geo×photo).

---

## 4. Methods: Code Graphs, Baselines & Metrics

### 4.1 Pre-Registered Empirical Gates (`PREREG.json`)

|Gate|Domain|Metric|Rule|Outcome|Status|
|:---|:---|:---|:---|:---|:---:|
|**Gate A1 (16²)**|Exp A (16²)|Clean-$h_v$ Ridge cell acc|≥0.99 cell, ≥0.95/class|**1.000 all 8 classes**|**PASS**|
|**Gate A1 (32/48²)**|Exp A (32/48²)|Sub-patch packing probe|≥0.99 cell, ≥0.95/class|Cell >0.99, `selected` **0.132**|**FAIL**|
|**Gate A2**|Exp A (Pool)|Max off-target centered cos|Report fraction >0.85|Median 0.8757 (**72.8% >0.85**)|**Measured**|
|**Gate A3**|Exp A (Control)|Non-DG EncLoRA `seen_bal360`|Changed-cell acc >0.0|**90.0% changed** (73.3% 22-way)|**PASS**|
|**Gate B0**|Exp B (Encoder)|`all-MiniLM-L6-v2` isotropy|$\text{PR} \ge 20$, $\cos \le 0.20$|**PR=47.8, cos=-0.0015, Oracle=1.0**|**PASS**|
|**Gate B1**|Exp B (Control)|Non-DG delete slot hit|≥50% (chance 0.39%)|**52.2% (47/90)**|**PASS**|

*Table 3: Pre-registered gate verification prior to DG training.*

Gate A1 confirms 16×16 grids decode losslessly, whereas packing 32² or 48² cells into 256 tokens crushes sparse node classes (`selected`: 13.2%). Categorical 8-color separation also bypasses SigLIP LayerNorm luminance loss (where brightness conditionals scored 48.2% / 23.2%, vs. 83.3 ± 2.7% on palette conditionals, $p < 10^{-12}$).

### 4.2 Metrics & Baseline Definitions
- **Experiment A Metrics:** From the Ridge-decoded 16×16 grid, we report **22-way nearest-candidate accuracy** (strictly smallest Hamming distance among the source graph's 22 pre-registered candidate states; ties wrong; chance $4.5\%$), **exact-grid match** (all 256 cells correct simultaneously), and off-diagonal call **edge F1**.
- **Experiment B Metrics:** Centered cosine nearest-neighbor lookup at edited slots over an 8-candidate AST edit pool (`chance = 12.5%`; `mean-vector floor = 25.0%`, as predicting near zero selects `delete` on all delete items).
- **"Non-DG" Architecture & Scope Limits:** `NonDGTransformBaseline` is a **4-layer ($d=768$, 12-head) spatial cross-attention transformer trained from scratch (`51.7M` params in Exp A, `49.9M` in Exp B, `53.6M` on images)** conditioned on `all-MiniLM-L6-v2` (22.7M) instruction tokens (**Frozen Instr** or $r=16$ **Enc LoRA**) with slot residuals ($x \leftarrow x + \text{SA} + \text{CA} + \text{FFN}$). This scratch ~50M model is **not parameter- or pretraining-matched** to 26B pretrained DG, and no autoregressive LLM baseline is evaluated here.
- **Training & Cutoff:** Trained arms run 4,824 steps (seeds `42`, `1337`) on 1,608 subgraphs across 8 Python repos and evaluate zero-shot on held-out `httpx` ($N=360$) and `click` ($N=360$). Runs incomplete at our cutoff are marked `not run`.

---

## 5. Experimental Results & Analysis

|Model Arm|Natural Images ($K=24$)|Exp A: Code Graphs ($K=22$)|Exp B: Symbol Latents ($K=8$)|Architecture|
|:---|:---:|:---:|:---:|:---|
|**Oracle Ceiling**|100.0% (23.3 dB)|100.0% (100.0% Exact)|100.0% (1.000 Cos)|Ground-truth latent|
|**Chance / Mean Floor**|4.2% chance|4.5% chance|12.5% ch. / 25.0% mean|Analytical floor|
|**Source Copy**|4.7%|7.2% (7.2% Exact)|0.0% hit (0.0 drift)|Unmodified input|
|**Non-DG + Frozen Instr**|74.5% / 32.7%|75.7 ± 4.6% / 39.6 ± 0.4%|46.1 ± 3.9% / 41.7 ± 2.8%|Scratch ~50M + Frz Text|
|**Non-DG + Enc LoRA**|74.8% / 61.3%|73.3 ± 0.0% / 49.2 ± 3.6%|**47.4 ± 1.3%** / **41.3 ± 4.3%**|Scratch ~50M + Enc LoRA|
|**Frozen DG + Head**|12.7% / 12.3%|0.3% / 0.0%|not run|Frozen 26B + Readout|
|**LoRA DG**|**91.3%** / **90.5%**|**91.9 ± 1.1%** / **78.5 ± 3.2%**|26.7% / 28.6% (s42; s1337 not run)|26B + LoRA ($r=64$) + Head|

*Table 4: Seen / unseen paraphrase K-way nearest-candidate accuracy (`httpx` $N=360$; DOCCI $N=300$).*

### 5.1 22-Way Selection vs. Exact-Grid Match, Indirect Intent & Cross-Repo Transfer

|Manifest (`httpx`, $N=360$)|LoRA DG: 22-Way|LoRA DG: Exact (Edge F1)|Best Non-DG: 22-Way|Best Non-DG: Exact (Edge F1)|Copy Exact|
|:---|:---:|:---:|:---:|:---:|:---:|
|`seen_bal360`|**91.9 ± 1.1%**|**62.2 ± 4.2%** (94.6%)|75.7 ± 4.6% (Frz)|**7.2 ± 0.0%** (59.7%)|7.2%|
|`unseen_para_bal360`|**78.5 ± 3.2%**|**47.1 ± 3.8%** (84.1%)|49.2 ± 3.6% (LoRA)|**3.3 ± 1.4%** (47.7%)|7.2%|
|`unseen_infer_360`|**61.4 ± 2.5%**|**41.2 ± 1.2%** (79.8%)|23.2 ± 1.2% (LoRA)|**0.0 ± 0.0%** (34.7%)|11.1%|
|`refeed_heldout_comp` (2-Step)|**82.1 ± 1.0%**|**47.8 ± 2.2%** (98.0%)|34.6 ± 2.4% (LoRA)|**0.0 ± 0.0%** (40.9%)|0.0%|
|3-Step `keep_callers` ($n=180$)|**90.8 ± 4.2%**|**0.0 ± 0.0%** (73.1%)|0.0 ± 0.0%|0.0 ± 0.0%|0.0%|

*Table 5: 22-way nearest-candidate accuracy vs. 256-cell exact-grid match and call-edge F1 on `httpx`.*

- **Headline Gaps & Exact-Grid Fidelity:** LoRA DG beats the strongest Non-DG arm per split by **+16.2 pp** seen (91.9% vs. 75.7% Frozen Instr; +18.6 pp vs. 73.3% Enc LoRA, $p < 10^{-8}$), **+29.3 pp** on paraphrases ($p < 10^{-15}$), and **+38.2 pp** on indirect intent ($p < 10^{-31}$). Non-DG's seen exact-grid match (**7.2%**) equals source copy (**0.0% exact on non-identity ops**), whereas LoRA DG reaches **62.2%** exact seen and **47.1%** exact unseen.
- **Shuffled Control Below Chance:** On `seen_shuf360` (pairing source $A$ with another item's instruction), LoRA DG scores **0.6±0.3%** (below 4.5% chance) because it applies the mismatched edit faithfully, moving predictions away from the unshuffled target.
- **Bimodal Indirect Intent:** `unseen_infer_360` (61.4±2.5%) is **bimodal**: `reverse_order`, `delete_selected`, `keep_callees`, and `selected_first` score **100.0% (40/40)** on both seeds (`reverse_edges`: 45.0%/100.0%; `mark_reachable`: 67.5%/65.0%), while `identity` (**0/40**), `delete_tests` (**1/40–2/40**), and `delete_classes` (**2/40–6/40**) fail without lexical overlap with training verbs.
- **Zero-Shot `click` Transfer ($N=360$):** Seed-42 LoRA DG reaches **88.3% 22-way / 61.4% exact** seen, **72.5% / 38.6% exact** on paraphrases, and **58.9% / 38.3% exact** on indirect intent vs. best 2-seed Non-DG `72.1%`, `46.1%`, and `24.4%` (`7.4%`, `3.8%`, `0.0%` exact; s1337 `click` DG `not run`).

### 5.2 Held-Out Node Categories & Commuting Compounds
Zero-shot node-category transfer **fails**: held-out `delete_methods` scores 13.2±2.9% 22-way (**0.0% exact**) on LoRA DG and 6.9±2.8% on Non-DG, and 1-pass `keep_callers` / `mark_ancestors` score 0.0%. On order-invariant **commuting** compounds, 1-pass LoRA DG reaches **99.3±0.7%** on `reverse_order + delete_selected` ($p < 10^{-89}$) and **98.6±1.4%** on `reverse_order + delete_tests` vs. Non-DG 8.1% and 6.5%.

### 5.3 Caller-Decomposed Re-Feeding & 3-Step Mirror Chains
With caller-supplied step prompts and $\hat{h}_v^{(t)}$ re-fed through `embed_vision`, 5-compound accuracy rises from 48.0±1.7% (1-pass) to **82.1±1.0% 22-way** and **47.8±2.2% exact** (98.0% edge F1; Non-DG: 34.6% / **0.0% exact**, $p < 10^{-41}$), lifting `reverse_edges + keep_callees` from 0.3% to **100.0±0.0%**, `delete_tests + mark_reachable` to **70.1±2.1%**, and `reverse_edges + mark_reachable` to **40.3±2.8%**. A caller-specified **3-step mirror chain** ($\text{reverse\_edges} \to \text{keep\_callees} \to \text{reverse\_edges}$) reaches **90.8±4.2% 22-way** on unseen `keep_callers` (45.4±2.1% chain-wide; `mark_ancestors`: 0.0%; Non-DG/Frozen DG: 0.0%, $p < 10^{-51}$), though **exact-grid match is 0.0% on both seeds** (97.2% cell acc, 73.1% `keep_callers` edge F1, 84.5% chain edge F1) from accumulated single-cell drift.

### 5.4 Open-Ended Edits: Branch Commitment, Not Diversity
On ambiguous edits with 2–3 valid targets (`cut_one_call`, `delete_one_callee`, `delete_one_test`), continuous MSE averages branches (Non-DG: 26.0% seen, 11.4% unseen). LoRA DG lands on a valid branch on **49.5±7.0%** (seen) and **50.4±5.2%** (unseen) per seed (**55.0±5.8%** / **56.1±5.3%** across 4 sampler seeds; first-valid 39.8% vs. 42.5% uniform). However, **all 4 sampler seeds return nearly identical grids (`1.005–1.035` distinct valid answers/item)**, showing deterministic commitment rather than diverse sampling.

### 5.5 Experiment B: Cold-Start Continuous Symbol Injection Boundary
Scratch `NonDGTransformBaseline` (49.9M) reaches 47.4 ± 1.3% seen and 41.3 ± 4.3% unseen (`click`: 39.2% / 32.8%), with re-feeding lifting $k=2$ all-sites accuracy from 5.6% to 40.9% ($k=4$: 1.3% $\to$ 34.2%). Seed-42 LoRA DG reaches 98.3% 3-class (`_`/`+`/`-`) canvas accuracy yet scores **26.7% seen, 28.6% unseen, 25.3% shuffled, and 50.0% `renameclass`** ($k=2$ 1-pass: 5.3%, $k=2$ re-feed: 5.3%, `drift: 18.1× → 18.7×`; $k=4$ 1-pass: 0.6%; $k=4$ re-feed, Seed 1337, and Frozen DG `not run`), with **100.0% `delete` recall** vs. **2.2% seen / 4.8% unseen** non-delete accuracy (`top_is_site = 0.6%`), matching the 25.0% / 50.0% mean-vector floor. Subject to single-seed DG evidence and unmatched architectures, this contrast **suggests** that reconstructing all $256 \times 384$ vectors across a 3-symbol canvas vocabulary without a slot-wise residual skip lets unchanged-slot drift swamp single-symbol edits—a hypothesis whose gated-residual fix remains untested.

---

## 6. Related Works

1. **Latent World Models & Continuous Reasoning:** LPN [10] performs gradient search in continuous program latents; Tycho [16] and DreamTeam [21] maintain structured agent states. Coconut [13] passes continuous *text-latent* hidden states across steps, whereas COVT [20] and Latent Sketchpad [24] use visual-latent tokens, and DINO-WM [25], LeWorldModel [18], and V-JEPA 2 [3] model dynamics in frozen visual features.
2. **Code Graphs & Code LLMs:** Code Property Graphs [23], Allamanis et al. [1], GraphCodeBERT [11], and ProGraML [6] represent AST, data-flow, and call graphs for GNNs, whereas autoregressive code LLMs like StarCoder [26] and DeepSeek-Coder [12] linearize files into token streams.
3. **Diffusion Language Models:** LLaDA [19] and Scaling-DLM [9] train masked diffusion LMs; DiffusionGemma [7] denoises a 256-token Gemma-4 canvas whose commitment order and bottlenecks are analyzed by Asaria et al. [2] and Engels et al. [8], alongside multimodal Transfusion [27].
4. **Perceptual Token Grounding & Multimodal Projection:** PaLM-E [28], RT-2 [5], and GILL [14] project frozen visual latents into LM trunks for robotics and image generation; we adapt frozen `SigLIP + embed_vision` to discrete AST adjacency grids.
5. **Anisotropy & Alignment:** Gao et al. [29] and Ethayarajh [30] document narrow-cone embedding degeneration, while REPA [22], GAP [17], LatentLens [15], and LeJEPA [4] regularize latent geometry.

---

## 7. Reusable Resources & Verifiable Artifacts

- **Standalone Python Libraries (`v0.1.0`):** [`dgg-latent`](https://github.com/bumblebeers/dgg-latent) (continuous latent diffusion losses, readout heads, and `stock_generate_with_readout`), [`dgg-codebook`](https://github.com/bumblebeers/dgg-codebook) (continuous vocabulary chords, de-biasing, and spatial teachers), and [`dgg-vit-decoder`](https://github.com/bumblebeers/dgg-vit-decoder) (572.8M SigLIP vision-stack loader and 34.15M ViT+Conv $h_v \to 768^2$ decoder).
- **Upstream `transformers` Bugfix (`dg_attn_fix`):** Fixes dropped `create_masks_for_generate(...)` assignment in `DiffusionGemmaEncoderModel.forward` under 2D `attention_mask` ([`bumblebeers/transformers` branch `fix/diffusion-gemma-prefill-multimodal-masks`](https://github.com/bumblebeers/transformers/tree/fix/diffusion-gemma-prefill-multimodal-masks)).
- **931-Asset Open Bundle & Raw Predictions:** SHA-256 cataloged across [`bumblebeers/dgg2-latent-code-graphs`](https://github.com/bumblebeers/dgg2-latent-code-graphs) (code, `PREREG.json`, offline scorers `cg/cg_score_preds.py` & `cg/cg_node_metrics.py`), [`bumblebeer/dgg2-code-graph-benchmarks`](https://huggingface.co/datasets/bumblebeer/dgg2-code-graph-benchmarks) and [`unclebumblebeer/dgg2-code-graph-benchmarks`](https://www.kaggle.com/datasets/unclebumblebeer/dgg2-code-graph-benchmarks) (benchmark packs & raw `.npy` predictions), and [`bumblebeer/dgg2-latent-code-graphs-adapters`](https://huggingface.co/bumblebeer/dgg2-latent-code-graphs-adapters) (LoRA adapters, readout heads, and `ImageDecoder` weights).

---

## 8. Limitations & Future Work

1. **Unmatched Controls & No Autoregressive Baseline:** Scratch ~50M `NonDGTransformBaseline` is not scale- or pretraining-matched to 26B DG, and we did not evaluate an autoregressive LLM editing graphs or Python source directly.
2. **Caller Decomposition & No Code Synthesis:** Multi-step re-feeding uses caller-supplied step prompts, and predicted 16×16 graphs are not yet compiled into source diffs.
3. **Resolution, 3-Step Drift & Cold-Start Adapters:** Sub-patch packing (32²/48²) crushes sparse classes (`selected`: 13.2%), 3-step chains hit 0.0% exact match (`mark_ancestors`: 0.0%), and Exp B incurs 18.0× drift without a tested gated residual skip $\hat{z}_i = g_i \hat{z}_i^{\text{dec}} + (1 - g_i) z_i^{\text{src}}$.

---

## References

[1] M. Allamanis et al., "Learning to Represent Programs with Graphs," ICLR, 2018.  
[2] C. Asaria et al., "Neither Parallel Nor Sequential: How DiffusionGemma Actually Commits Tokens," arXiv:2606.14620, 2026.  
[3] M. Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning," arXiv:2506.09985, 2025.  
[4] R. Balestriero & Y. LeCun, "LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics," arXiv:2511.08544, 2025.  
[5] A. Brohan et al., "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control," arXiv:2307.15818, 2023.  
[6] C. Cummins et al., "ProGraML: A Graph-based Program Representation for Data Flow Analysis," ICML, 2021.  
[7] DiffusionGemma Team, "DiffusionGemma Technical Report," arXiv:2608.00146, 2026.  
[8] J. Engels et al., "How Transparent is DiffusionGemma?," arXiv:2606.20560, 2026.  
[9] S. Gong et al., "Scaling Diffusion Language Models via Adaptation from Autoregressive Models," ICLR, 2025.  
[10] M. V. Macfarlane & C. Bonnet, "Searching Latent Program Spaces," NeurIPS, 2025.  
[11] D. Guo et al., "GraphCodeBERT: Pre-training Code Representations with Data Flow," ICLR, 2021.  
[12] D. Guo et al., "DeepSeek-Coder: When the Large Language Model Meets Programming," arXiv:2401.14196, 2024.  
[13] S. Hao et al., "Training Large Language Models to Reason in a Continuous Latent Space," arXiv:2412.06769, 2024.  
[14] J. Y. Koh et al., "Generating Images with Multimodal Language Models," arXiv:2305.17216, 2023.  
[15] B. Krojer et al., "Revealing Highly Interpretable Visual Tokens in LLMs," arXiv:2602.00462, 2026.  
[16] J. Lehmann et al., "Tycho: Active Abstraction with Programmatic World Models for ARC-AGI-3," arXiv:2607.28287, 2026.  
[17] Y. Miao et al., "Fill the GAP: Granular Alignment Paradigm for Visual Reasoning," arXiv:2605.12374, 2026.  
[18] Q. Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture," arXiv:2603.19312, 2026.  
[19] S. Nie et al., "Large Language Diffusion Models," ICML, 2025.  
[20] Y. Qin et al., "Chain-of-Visual-Thought: Continuous Visual Tokens," arXiv:2511.19418, 2025.  
[21] E. Sarafian et al., "Workspace Optimization: How to Train Your Agent," arXiv:2605.09650, 2026.  
[22] S. Yu et al., "Representation Alignment for Generation," ICLR, 2025.  
[23] F. Yamaguchi et al., "Modeling and Discovering Vulnerabilities with Code Property Graphs," IEEE S&P, 2014.  
[24] H. Zhang et al., "Latent Sketchpad: Sketching Visual Thoughts in MLLMs," arXiv:2510.24514, 2025.  
[25] G. Zhou et al., "DINO-WM: World Models on Pre-trained Visual Features enable Zero-shot Planning," arXiv:2411.04983, 2024.  
[26] R. Li et al., "StarCoder: may the source be with you!," TMLR, 2023.  
[27] C. Zhou et al., "Transfusion: Predict the Next Token and Diffuse Images with One Model," arXiv:2408.11039, 2024.  
[28] D. Driess et al., "PaLM-E: An Embodied Multimodal Language Model," ICML, 2023.  
[29] J. Gao et al., "Representation Degeneration Problem in Language Modeling," ICLR, 2019.  
[30] K. Ethayarajh, "How Contextual are Contextualized Word Representations?," EMNLP, 2019.  
