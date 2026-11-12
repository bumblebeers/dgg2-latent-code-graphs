# dgg-vit-decoder: Standalone Vision Stack Loader and ViT+Conv Decoder for DiffusionGemma

`dgg-vit-decoder` extracts DiffusionGemma's vision components into a lightweight, standalone library that runs without loading the 49GB / 26B language model.

## Features

1. **`VisionStack` Loader (572M Parameters)**:
   Extracts and runs the SigLIP vision tower and `embed_vision` projector independently from the 26B checkpoint:
   - Takes $768 \times 768$ RGB image pixel values $\rightarrow$ emits continuous latent tokens $h_v \in \mathbb{R}^{256 \times 1152}$ and soft tokens in $\mathbb{R}^{256 \times 2816}$.
   - Requires zero language model memory (fits easily on local consumer or edge GPUs with 8-16 GB VRAM).
2. **`ImageDecoder` Architecture (34.15M Parameters)**:
   The frozen ViT+Conv decoder architecture mapping continuous visual latent tokens $h_v$ back to full-resolution $768 \times 768 \times 3$ RGB images:
   - **`TokenMixer`**: LayerNorm + 1x1 conv projection + 2D positional embeddings + 4 pre-LN ViT self-attention blocks with zero-init projection.
   - **`ConvDecoder`**: 5-stage convolutional PixelShuffle upsampler ($16 \rightarrow 48 \rightarrow 96 \rightarrow 192 \rightarrow 384 \rightarrow 768$) with residual blocks at each resolution stage.
   - Zero-initialized final conv head ensuring initial output is exact mid-grey ($\sigma(0) = 0.5$).
3. **Discrete Palette Grid Recovery**:
   - `pred_to_grids`: Maps decoded continuous RGB outputs back to discrete palette classes.
4. **Perceptual & Reconstruction Metrics**:
   - `psnr`: Peak Signal-to-Noise Ratio.
   - `ssim`: Structural Similarity Index Measure with Gaussian weighting window.

## Installation

```bash
pip install -e src/dgg_vit_decoder
```
