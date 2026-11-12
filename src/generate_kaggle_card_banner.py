#!/usr/bin/env python3
"""
Generate 560x280 Card & Thumbnail Banner (exact 560x280 RGB PNG).

Specs:
  - Exact dimensions: 560x280 pixels
  - Subject: 1 Source Grid -> arrow labeled with instruction (`keep_callees`) -> 1 Predicted Grid
    (seen_bal360 idx 4, Seed 42 genuine prediction, Exact: HIT)
  - Title: "Latent Roadmaps" + at most one headline number ("91.9% 22-Way")
  - 4x Supersampling (2240x1120 down to 560x280 via Lanczos)
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cg"))

from cg import cg_ops as O
from cg import cg_sec27_spec as SP


def render_grid_u8(grid: np.ndarray, cell_px: int = 20) -> np.ndarray:
    """Render 16x16 grid with official cg_ops.PALETTE_RGB and subtle grid lines."""
    h, w = grid.shape
    H, W = h * cell_px, w * cell_px
    arr = np.zeros((H, W, 3), dtype=np.uint8)

    for r in range(h):
        for c in range(w):
            color = O.PALETTE_RGB[int(grid[r, c])]
            arr[r * cell_px : (r + 1) * cell_px, c * cell_px : (c + 1) * cell_px] = color

    line_col = np.array([215, 220, 228], dtype=np.uint8)
    for r in range(1, h):
        arr[r * cell_px, :] = line_col
    for c in range(1, w):
        arr[:, c * cell_px] = line_col
    return arr


def create_kaggle_card_banner(out_path: str = "docs/figures/kaggle_card_thumbnail_560x280.png") -> None:
    out_file = REPO_ROOT / out_path
    out_file.parent.mkdir(parents=True, exist_ok=True)

    S = 4
    W, H = 560 * S, 280 * S

    # Deep slate gradient background
    grad = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, W)[None, :]
    ys = np.linspace(0.0, 1.0, H)[:, None]
    mix = 0.65 * xs + 0.35 * ys
    c0 = np.array([11, 17, 32], dtype=np.float32)
    c1 = np.array([22, 38, 76], dtype=np.float32)
    for ch in range(3):
        grad[:, :, ch] = np.clip(c0[ch] * (1.0 - mix) + c1[ch] * mix, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(grad, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    # Subtle background grid
    grid_step = 28 * S
    for gx in range(0, W, grid_step):
        draw.line([(gx, 0), (gx, H)], fill=(23, 38, 72), width=2)
    for gy in range(0, H, grid_step):
        draw.line([(0, gy), (W, gy)], fill=(23, 38, 72), width=2)

    font_bold = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
    font_reg = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"

    f_h1 = ImageFont.truetype(font_bold, 30 * S)
    f_sub = ImageFont.truetype(font_reg, 12.5 * S if False else 13 * S)
    f_metric = ImageFont.truetype(font_bold, 12 * S)
    f_cap = ImageFont.truetype(font_bold, 11 * S)
    f_instr = ImageFont.truetype(font_bold, 9 * S)

    # ------------------------------------------------------------------
    # Left Column: Title + Subtitle + Single Re-Feeding Metric
    # ------------------------------------------------------------------
    lx = 18 * S
    draw.text((lx, 42 * S), "LATENT", font=f_h1, fill=(255, 255, 255))
    draw.text((lx, 79 * S), "ROADMAPS", font=f_h1, fill=(96, 165, 250))

    draw.text((lx, 132 * S), "Soft-Token Re-Feeding", font=f_sub, fill=(226, 232, 240))
    draw.text((lx, 151 * S), "in DiffusionGemma", font=f_sub, fill=(148, 163, 184))

    # Re-feeding focal badge
    m_w = 196 * S
    draw.rounded_rectangle(
        [lx, 192 * S, lx + m_w, 240 * S],
        radius=10 * S,
        fill=(13, 24, 44),
        outline=(74, 222, 128),
        width=2 * S,
    )
    draw.text(
        (lx + m_w // 2, 216 * S),
        "1-pass 0.3% -> re-feed 100%",
        font=f_metric,
        fill=(134, 239, 172),
        anchor="mm",
    )

    # ------------------------------------------------------------------
    # Right Column: Source Grid -> Re-Feed Arrow -> Predicted Grid
    # ------------------------------------------------------------------
    pack = SP.load_pack(str(REPO_ROOT / "data" / "gridA_test360_httpx.npz"))
    item_idx = 72  # First reverse_edges+keep_callees item in refeed_heldout_comp_bal360 (Exact: HIT)
    img_idx = 72

    z_pred = np.load(REPO_ROOT / "outputs" / "m2" / "decoded_grids_all_arms.npz")
    src_grid = pack["src_grid"][img_idx]
    pred_grid = z_pred["arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_refeed_heldout_comp_bal360"][item_idx]

    im_src = Image.fromarray(render_grid_u8(src_grid, cell_px=20))
    im_pred = Image.fromarray(render_grid_u8(pred_grid, cell_px=20))

    box_sz = 110 * S
    y_top = 58 * S
    x_src = 232 * S
    x_pred = 428 * S

    # Source Grid Card
    draw.rounded_rectangle(
        [x_src - 4 * S, y_top - 4 * S, x_src + box_sz + 4 * S, y_top + box_sz + 26 * S],
        radius=8 * S,
        fill=(15, 23, 42),
        outline=(148, 163, 184),
        width=2 * S,
    )
    canvas.paste(im_src.resize((box_sz, box_sz), Image.Resampling.NEAREST), (x_src, y_top))
    draw.text(
        (x_src + box_sz // 2, y_top + box_sz + 12 * S),
        "Source Graph",
        font=f_cap,
        fill=(203, 213, 225),
        anchor="mm",
    )

    # Labeled Instruction Arrow between Source & Predicted
    ax0 = x_src + box_sz + 10 * S
    ax1 = x_pred - 10 * S
    ay = y_top + box_sz // 2 + 6 * S
    acx = (ax0 + ax1) // 2

    # Instruction pill above arrow
    pill_hw = 40 * S
    draw.rounded_rectangle(
        [acx - pill_hw, ay - 28 * S, acx + pill_hw, ay - 10 * S],
        radius=6 * S,
        fill=(30, 58, 138),
        outline=(96, 165, 250),
        width=int(1.5 * S),
    )
    draw.text(
        (acx, ay - 19 * S),
        "2-step re-feed",
        font=f_instr,
        fill=(219, 234, 254),
        anchor="mm",
    )

    # Crisp vector arrow shaft + head
    head_len = 10 * S
    head_half = 6 * S
    draw.line([(ax0, ay), (ax1 - head_len + 2 * S, ay)], fill=(96, 165, 250), width=3 * S)
    draw.polygon(
        [(ax1, ay), (ax1 - head_len, ay - head_half), (ax1 - head_len, ay + head_half)],
        fill=(96, 165, 250),
    )

    # Predicted Grid Card
    draw.rounded_rectangle(
        [x_pred - 4 * S, y_top - 4 * S, x_pred + box_sz + 4 * S, y_top + box_sz + 26 * S],
        radius=8 * S,
        fill=(15, 23, 42),
        outline=(74, 222, 128),
        width=2 * S,
    )
    canvas.paste(im_pred.resize((box_sz, box_sz), Image.Resampling.NEAREST), (x_pred, y_top))
    draw.text(
        (x_pred + box_sz // 2, y_top + box_sz + 12 * S),
        "Predicted Graph",
        font=f_cap,
        fill=(134, 239, 172),
        anchor="mm",
    )

    final_img = canvas.resize((560, 280), Image.Resampling.LANCZOS)
    assert final_img.size == (560, 280), f"Expected (560, 280), got {final_img.size}"
    final_img.save(out_file, format="PNG", optimize=True)
    print(f"[Done] Saved Thumbnail: {out_file} ({final_img.size[0]}x{final_img.size[1]})")


if __name__ == "__main__":
    create_kaggle_card_banner()
