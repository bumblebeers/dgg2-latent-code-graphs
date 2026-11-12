# dgg-latent: Continuous Latent Diffusion, Losses, and Readout Heads for DiffusionGemma

`dgg-latent` provides clean, standalone implementations of core mechanisms for adapting DiffusionGemma (DG 26B-A4B) to continuous latent representations, code graph embeddings, and multi-modal canvas decoders.

## Modules

- **`dgg_latent.readout_heads`**: Pointwise single-canvas and two-canvas readout heads mapping decoder hidden states and continuous vocabulary chords to visual/latent spaces:
  - `SoftCanvasHead`: 3.64M parameter pointwise MLP ($H \in \mathbb{R}^{B \times 256 \times 2816} \rightarrow h_v \in \mathbb{R}^{B \times 256 \times 1152}$).
  - `DualReadoutSoftCanvasHead`: 7.00M parameter dual-norm gated residual head fusing unconstrained decoder hidden states $h_{\text{dec}}$ with expected vocabulary chord $e_{\text{soft}}$ and detached confidence summary $c_z$.
  - `CrossSoftCanvasHead`: 10.33M parameter cross-attention head attending across structural and photometric canvas hidden states.
  - `compute_readout_input`: Computes inputs for single, logits-based, or dual-gated readout heads.
  - `build_single_canvas_head`: Extensible factory function.

- **`dgg_latent.losses`**: Numerically stable and stock-compliant loss functions:
  - `distribution_ce`: Multi-modal vocabulary distribution cross-entropy with rank-1 mode crash-guard.
  - `loss_hv`: SmoothL1 + Cosine similarity loss in continuous latent space.
  - `loss_pixel`: Unclamped pixel reconstruction loss through differentiable image decoders.
  - `per_example_reliability_weight`: D3PM per-example noise reliability weighting $w(t) = \text{clamp}(1 - t, 0, 1)$ preserving Jensen's inequality.
  - `balanced_slot_loss`: Equalized loss weighting across changed and unchanged slots for structured graph refactoring.
  - `PhaseController`: Annealing schedule manager for multi-phase training conditions.

- **`dgg_latent.sampling`**: Continuous latent diffusion sampling routines and utilities:
  - `ContinuousLatentSampler`: Multi-step continuous latent refinement with self-conditioning.
  - `corrupt_discrete_canvas`: D3PM multinomial corruption scheduler.
  - `slerp`: Spherical linear interpolation between continuous latent representations.
  - `cosine_corruption_schedule` & `linear_corruption_schedule`.

## Installation

```bash
pip install -e src/dgg_latent
```
