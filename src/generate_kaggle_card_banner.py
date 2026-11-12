#!/usr/bin/env python3
"""Generate Kaggle Writeup Card & Thumbnail Banner (exact 560x280 RGB PNG).

Renders at 4x supersampling (2240x1120) and Lanczos-downsamples to 560x280
for crisp typography and clean visual isomorphism between natural-image
patch latents (qual_dev_000031) and 16x16 AST call-graph grids (httpx).
"""
from __future__ import annotations

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath("."))
from cg.cg_ops import render


def create_kaggle_card_banner(out_path: str = "docs/figures/kaggle_card_thumbnail_560x280.png") -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    S = 4
    W, H = 560 * S, 280 * S

    # Diagonal midnight-to-royal-navy gradient background
    grad = np.zeros((H, W, 3), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, W)[None, :]
    ys = np.linspace(0.0, 1.0, H)[:, None]
    mix = 0.65 * xs + 0.35 * ys
    c0 = np.array([10, 15, 29], dtype=np.float32)
    c1 = np.array([23, 39, 84], dtype=np.float32)
    for ch in range(3):
        grad[:, :, ch] = np.clip(c0[ch] * (1.0 - mix) + c1[ch] * mix, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(grad, mode="RGB")
    draw = ImageDraw.Draw(canvas)

    # Subtle background blueprint grid
    grid_step = 28 * S
    for gx in range(0, W, grid_step):
        draw.line([(gx, 0), (gx, H)], fill=(24, 40, 78), width=2)
    for gy in range(0, H, grid_step):
        draw.line([(0, gy), (W, gy)], fill=(24, 40, 78), width=2)

    font_bold = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
    font_reg = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"

    f_badge = ImageFont.truetype(font_bold, 10 * S)
    f_h1 = ImageFont.truetype(font_bold, 34 * S)
    f_sub = ImageFont.truetype(font_reg, 13 * S)
    f_metric = ImageFont.truetype(font_bold, 12 * S)
    f_cap = ImageFont.truetype(font_bold, 11 * S)
    f_pill = ImageFont.truetype(font_bold, 11 * S)

    # ------------------------------------------------------------------
    # Left Column: Focal Branding & Headline Metrics
    # ------------------------------------------------------------------
    lx = 20 * S
    pill_w = 216 * S
    draw.rounded_rectangle(
        [lx, 20 * S, lx + pill_w, 42 * S],
        radius=11 * S,
        fill=(15, 29, 64),
        outline=(56, 189, 248),
        width=2 * S,
    )
    draw.text(
        (lx + pill_w // 2, 31 * S),
        "GEMMA 4 DEVELOPER AGENT",
        font=f_badge,
        fill=(125, 211, 252),
        anchor="mm",
    )

    draw.text((lx, 54 * S), "LATENT", font=f_h1, fill=(255, 255, 255))
    draw.text((lx, 91 * S), "ROADMAPS", font=f_h1, fill=(96, 165, 250))

    draw.text((lx, 139 * S), "DiffusionGemma 26B", font=f_sub, fill=(226, 232, 240))
    draw.text((lx, 158 * S), "Latent Graph Transitions", font=f_sub, fill=(148, 163, 184))

    m_w = 224 * S
    draw.rounded_rectangle(
        [lx, 196 * S, lx + m_w, 258 * S],
        radius=10 * S,
        fill=(13, 22, 40),
        outline=(74, 222, 128),
        width=2 * S,
    )
    draw.text(
        (lx + m_w // 2, 214 * S),
        "91.9% Seen  |  78.5% Unseen",
        font=f_metric,
        fill=(134, 239, 172),
        anchor="mm",
    )
    draw.text(
        (lx + m_w // 2, 239 * S),
        "82.1% 2-Step Latent Re-Feed",
        font=f_metric,
        fill=(253, 224, 71),
        anchor="mm",
    )

    # ------------------------------------------------------------------
    # Right Column: Visual Isomorphism (Google Water Bottle <-> 16x16 AST Grid)
    # ------------------------------------------------------------------
    p_raw = os.path.expanduser(
        "~/workspaces/arc-agi-3/rendered/dgg2/t2i/docci/images/qual_dev_000031.webp"
    )
    p_dec = "outputs/qual_dev_000031_rot90_decoded.png"
    im_photo = Image.open(p_raw if os.path.exists(p_raw) else p_dec).convert("RGB")

    z = np.load("data/gridA_test360_httpx.npz")
    src_grid = z["src_grid"][0]
    im_grid_raw = Image.fromarray(render(src_grid, cell_px=32))

    box_sz = 120 * S
    y_top = 44 * S
    x_photo = 262 * S
    x_grid = 420 * S

    # Photo card
    draw.rounded_rectangle(
        [x_photo - 5 * S, y_top - 5 * S, x_photo + box_sz + 5 * S, y_top + box_sz + 28 * S],
        radius=10 * S,
        fill=(15, 23, 42),
        outline=(148, 163, 184),
        width=2 * S,
    )
    thumb_photo = im_photo.resize((box_sz, box_sz), Image.Resampling.LANCZOS)
    p_draw = ImageDraw.Draw(thumb_photo)
    step_p = box_sz / 8.0
    for k in range(1, 8):
        pos = int(round(k * step_p))
        p_draw.line([(pos, 0), (pos, box_sz)], fill=(255, 255, 255), width=2)
        p_draw.line([(0, pos), (box_sz, pos)], fill=(255, 255, 255), width=2)
    canvas.paste(thumb_photo, (x_photo, y_top))
    draw.text(
        (x_photo + box_sz // 2, y_top + box_sz + 13 * S),
        "768² RGB Patches",
        font=f_cap,
        fill=(226, 232, 240),
        anchor="mm",
    )

    # AST Code-Graph card
    draw.rounded_rectangle(
        [x_grid - 5 * S, y_top - 5 * S, x_grid + box_sz + 5 * S, y_top + box_sz + 28 * S],
        radius=10 * S,
        fill=(15, 23, 42),
        outline=(56, 189, 248),
        width=3 * S,
    )
    thumb_grid = im_grid_raw.resize((box_sz, box_sz), Image.Resampling.NEAREST)
    g_draw = ImageDraw.Draw(thumb_grid)
    step_g = box_sz / 16.0
    for k in range(1, 16):
        pos = int(round(k * step_g))
        g_draw.line([(pos, 0), (pos, box_sz)], fill=(203, 213, 225), width=2)
        g_draw.line([(0, pos), (box_sz, pos)], fill=(203, 213, 225), width=2)
    canvas.paste(thumb_grid, (x_grid, y_top))
    draw.text(
        (x_grid + box_sz // 2, y_top + box_sz + 13 * S),
        "16×16 AST Graph",
        font=f_cap,
        fill=(125, 211, 252),
        anchor="mm",
    )

    # Center 1:1 Isomorphism Bridge Badge
    cx = (x_photo + box_sz + x_grid) // 2
    cy = y_top + box_sz // 2
    bw, bh = 24 * S, 15 * S
    draw.rounded_rectangle(
        [cx - bw, cy - bh, cx + bw, cy + bh],
        radius=12 * S,
        fill=(30, 58, 138),
        outline=(147, 197, 253),
        width=2 * S,
    )
    draw.text((cx, cy), "1 : 1", font=f_pill, fill=(255, 255, 255), anchor="mm")

    # Bottom right shared canvas caption bar
    rx1, rx2 = x_photo - 5 * S, x_grid + box_sz + 5 * S
    rcx = (rx1 + rx2) // 2
    draw.rounded_rectangle(
        [rx1, 206 * S, rx2, 258 * S],
        radius=10 * S,
        fill=(23, 37, 84),
        outline=(96, 165, 250),
        width=2 * S,
    )
    draw.text(
        (rcx, 222 * S),
        "Shared 256-Token SigLIP h_v Canvas",
        font=f_metric,
        fill=(255, 255, 255),
        anchor="mm",
    )
    draw.text(
        (rcx, 243 * S),
        "Bidirectional Denoising Canvas",
        font=f_badge,
        fill=(186, 230, 253),
        anchor="mm",
    )

    final_img = canvas.resize((560, 280), Image.Resampling.LANCZOS)
    assert final_img.size == (560, 280), f"Expected (560, 280), got {final_img.size}"
    final_img.save(out_path, format="PNG", optimize=True)
    print(f"Saved Kaggle Card & Thumbnail Banner: {out_path} ({final_img.size[0]}x{final_img.size[1]})")


if __name__ == "__main__":
    create_kaggle_card_banner()
