#!/usr/bin/env python3
"""generate_fig1_contact_sheet.py -- Publication Figure 1: Code-Graph Contact Sheet.

Generates docs/figures/fig1_code_graph_contact_sheet.png:
- 8 rows x 5 columns:
  Source Grid | Instruction Text | Target Grid | LoRA DG Prediction (s42) | Non-DG Prediction (EncLoRA s42)
- Deterministic Row Rule (strictly first occurrence in test manifest):
  1-4: Seen ops from seen_bal360 (reverse_edges, reverse_order, keep_callees, mark_reachable)
  5:   Syntactic paraphrase from unseen_para_bal360 (reverse_order)
  6:   Indirect-intent success from unseen_infer_360 (keep_callees)
  7:   Indirect-intent failure from unseen_infer_360 (delete_tests)
  8:   Held-out failure from heldout_mark_ancestors_360 (mark_ancestors)
- Red error outlines around every wrong cell (pred[r, c] != tgt[r, c])
- Two badges per prediction: 22-way HIT/MISS and Exact HIT/MISS (k err)
- 8-class palette legend swatch bar at top
- Dedicated column header row to completely eliminate typography collisions
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "cg"))

from cg import cg_sec27_spec as S
from cg import cg_ops as O

PALETTE_HEX = [
    "#F5F5F5",  # 0: BG
    "#141E5A",  # 1: EDGE
    "#3CAA5A",  # 2: FUNC
    "#28AABE",  # 3: METHOD
    "#8C50BE",  # 4: CLASS
    "#F09628",  # 5: TEST
    "#DC2832",  # 6: SELECTED
    "#FAE132",  # 7: MARKED
]
PALETTE_NAMES = [
    "0 Background", "1 Call Edge", "2 Function", "3 Method",
    "4 Class", "5 Test", "6 Selected", "7 Marked"
]


def render_grid_pil(grid: np.ndarray, cell_px: int = 14, error_mask: np.ndarray = None) -> Image.Image:
    """Render 16x16 grid with cell boundaries and crisp red error outlines."""
    h, w = grid.shape
    img_arr = np.zeros((h * cell_px, w * cell_px, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            color = O.PALETTE_RGB[grid[r, c]]
            img_arr[r * cell_px:(r + 1) * cell_px, c * cell_px:(c + 1) * cell_px] = color

    im = Image.fromarray(img_arr)
    d = ImageDraw.Draw(im)

    # Subtle inner grid lines
    grid_line_color = (210, 210, 215)
    for i in range(1, h):
        d.line([(0, i * cell_px), (w * cell_px, i * cell_px)], fill=grid_line_color, width=1)
    for j in range(1, w):
        d.line([(j * cell_px, 0), (j * cell_px, h * cell_px)], fill=grid_line_color, width=1)

    # Crisp red error outlines
    if error_mask is not None:
        err_color = (220, 20, 40)
        for r in range(h):
            for c in range(w):
                if error_mask[r, c]:
                    x0, y0 = c * cell_px, r * cell_px
                    x1, y1 = (c + 1) * cell_px - 1, (r + 1) * cell_px - 1
                    d.rectangle([x0, y0, x1, y1], outline=err_color, width=2)

    return im


def extract_figure1_data():
    pack = np.load(REPO_ROOT / "data" / "gridA_test360_httpx.npz")
    mans = S.build_cg_eval_manifests(pack)
    z_dg = np.load(REPO_ROOT / "outputs" / "m2" / "decoded_grids_all_arms.npz")

    rows_spec = [
        ("seen_bal360", "reverse_edges",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_seen_bal360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_seen_bal360.npy",
         "Seen Operation", "reverse_edges"),
        ("seen_bal360", "reverse_order",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_seen_bal360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_seen_bal360.npy",
         "Seen Operation", "reverse_order"),
        ("seen_bal360", "keep_callees",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_seen_bal360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_seen_bal360.npy",
         "Seen Operation", "keep_callees"),
        ("seen_bal360", "mark_reachable",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_seen_bal360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_seen_bal360.npy",
         "Seen Operation", "mark_reachable"),
        ("unseen_para_bal360", "reverse_order",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_unseen_para_bal360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_unseen_para_bal360.npy",
         "Syntactic Paraphrase", "reverse_order"),
        ("unseen_infer_360", "keep_callees",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_unseen_infer_360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_unseen_infer_360.npy",
         "Indirect Intent (Success)", "keep_callees"),
        ("unseen_infer_360", "delete_tests",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_unseen_infer_360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_unseen_infer_360.npy",
         "Indirect Intent (Failure)", "delete_tests"),
        ("heldout_mark_ancestors_360", "mark_ancestors",
         "arms__cg_lora_dg_seed42_ep3__preds_cg_lora_dg_seed42_ep3_heldout_mark_ancestors_360",
         "outputs/m2/preds/pred_grids_nondg_enc_lora_s42_heldout_mark_ancestors_360.npy",
         "Held-Out Operation (Failure)", "mark_ancestors")
    ]

    items = []
    for m_name, op, dg_k, ndg_f, category, op_display in rows_spec:
        m = mans[m_name]
        idx = m["op_names"].index(op)
        img_idx = int(m["img_indices"][idx])
        inst = str(m["instructions"][idx])
        tgt_state_idx = int(m["tgt_state24_indices"][idx])
        tgt_name = S.CG_ALL_STATES[tgt_state_idx]

        src_grid = pack["src_grid"][img_idx]
        tgt_grid = pack["tgt_grid"][img_idx, tgt_state_idx]
        pred_dg = z_dg[dg_k][idx]
        pred_ndg = np.load(REPO_ROOT / ndg_f)[idx]
        cands = pack["tgt_grid"][img_idx]

        # DG candidate scoring
        d_dg = (pred_dg[None] != cands).reshape(22, -1).sum(-1)
        cand_dg = int(d_dg.argmin())
        hit_22_dg = bool((cand_dg == tgt_state_idx) and ((d_dg == d_dg.min()).sum() == 1))
        exact_dg = bool(np.array_equal(pred_dg, tgt_grid))
        err_dg = int((pred_dg != tgt_grid).sum())

        # Non-DG candidate scoring
        d_ndg = (pred_ndg[None] != cands).reshape(22, -1).sum(-1)
        cand_ndg = int(d_ndg.argmin())
        hit_22_ndg = bool((cand_ndg == tgt_state_idx) and ((d_ndg == d_ndg.min()).sum() == 1))
        exact_ndg = bool(np.array_equal(pred_ndg, tgt_grid))
        err_ndg = int((pred_ndg != tgt_grid).sum())

        items.append({
            "category": category,
            "op": op_display,
            "manifest": m_name,
            "manifest_idx": idx,
            "img_idx": img_idx,
            "instruction": inst,
            "tgt_name": tgt_name,
            "src_grid": src_grid,
            "tgt_grid": tgt_grid,
            "pred_dg": pred_dg,
            "err_dg_mask": pred_dg != tgt_grid,
            "hit_22_dg": hit_22_dg,
            "exact_dg": exact_dg,
            "err_dg": err_dg,
            "pred_ndg": pred_ndg,
            "err_ndg_mask": pred_ndg != tgt_grid,
            "hit_22_ndg": hit_22_ndg,
            "exact_ndg": exact_ndg,
            "err_ndg": err_ndg,
        })

    return items


def generate_figure1():
    out_dir = REPO_ROOT / "docs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "fig1_code_graph_contact_sheet.png"

    items = extract_figure1_data()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(14.0, 18.5), dpi=300)

    # 11 rows: Row 0 = Title & Legend, Row 1 = Column Titles, Rows 2-9 = Data rows 1-8, Row 10 = Footer
    gs = GridSpec(
        nrows=11, ncols=5,
        figure=fig,
        height_ratios=[0.55, 0.18, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.35],
        width_ratios=[1.0, 1.7, 1.0, 1.0, 1.0],
        left=0.04, right=0.96, top=0.965, bottom=0.025,
        wspace=0.15, hspace=0.22
    )

    # Row 0: Header & Legend
    header_ax = fig.add_subplot(gs[0, :])
    header_ax.axis("off")

    header_ax.text(
        0.5, 0.90,
        "Code-Graph Visual Transitions: DiffusionGemma 26B-A4B vs. Scratch Non-DG Baseline",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#111827"
    )
    header_ax.text(
        0.5, 0.62,
        "Comparison of 16x16 adjacency graph transformations under 8 distinct semantic, syntactic, and held-out regimes (Seed 42).",
        ha="center", va="center", fontsize=9.5, color="#4b5563"
    )

    # Legend Bar: 8 color swatches centered
    swatch_w, swatch_h = 0.088, 0.22
    start_x = 0.5 - (8 * swatch_w + 7 * 0.015) / 2.0
    for idx, (c_hex, c_name) in enumerate(zip(PALETTE_HEX, PALETTE_NAMES)):
        sx = start_x + idx * (swatch_w + 0.015)
        sy = 0.10
        rect = patches.Rectangle((sx, sy), swatch_w, swatch_h, facecolor=c_hex, edgecolor="#9ca3af", linewidth=0.8, transform=header_ax.transAxes)
        header_ax.add_patch(rect)
        txt_color = "#111827" if idx not in [1, 4] else "#ffffff"
        header_ax.text(sx + swatch_w / 2.0, sy + swatch_h / 2.0, c_name.split()[0], ha="center", va="center",
                       fontsize=7.5, fontweight="bold", color=txt_color, transform=header_ax.transAxes)
        header_ax.text(sx + swatch_w / 2.0, sy - 0.15, " ".join(c_name.split()[1:]), ha="center", va="top",
                       fontsize=7.0, color="#374151", transform=header_ax.transAxes)

    # Row 1: Dedicated Column Titles Row (Zero collision guarantee!)
    col_titles = [
        "Source Grid (Input)",
        "Operation & Instruction Text",
        "Target Grid (Ground Truth)",
        "LoRA DG Prediction (s42)",
        "Non-DG Prediction (EncLoRA s42)"
    ]
    col_colors = ["#1f2937", "#1f2937", "#1f2937", "#047857", "#b91c1c"]

    for c_idx in range(5):
        ax_title = fig.add_subplot(gs[1, c_idx])
        ax_title.axis("off")
        ax_title.text(0.5, 0.5, col_titles[c_idx], ha="center", va="center",
                      fontsize=9.5, fontweight="bold", color=col_colors[c_idx])

    # Rows 2-9: The 8 Items
    for row_idx, item in enumerate(items):
        grid_row = row_idx + 2  # 2 to 9

        # Col 0: Source Grid
        ax_src = fig.add_subplot(gs[grid_row, 0])
        ax_src.set_xticks([]); ax_src.set_yticks([])
        img_src = render_grid_pil(item["src_grid"], cell_px=14)
        ax_src.imshow(img_src)
        for spine in ax_src.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(1.0)
        ax_src.set_ylabel(f"Row {row_idx+1}\n[Item {item['img_idx']}]", fontsize=8.0, color="#4b5563", rotation=0, labelpad=28, va="center")

        # Col 1: Instruction Text
        ax_txt = fig.add_subplot(gs[grid_row, 1])
        ax_txt.axis("off")

        cat_color = "#2563eb" if "Seen" in item["category"] else ("#7c3aed" if "Syntactic" in item["category"] else ("#059669" if "Success" in item["category"] else "#dc2626"))
        ax_txt.text(0.04, 0.88, f"{item['category']}", fontsize=8.5, fontweight="bold", color=cat_color)
        ax_txt.text(0.04, 0.70, f"Op: {item['op']}", fontsize=8.5, fontweight="semibold", color="#111827")

        # Wrapped instruction in styled block
        inst_wrapped = item["instruction"]
        words = inst_wrapped.split()
        lines, curr = [], []
        for w in words:
            if sum(len(x) for x in curr) + len(curr) + len(w) > 34:
                lines.append(" ".join(curr))
                curr = [w]
            else:
                curr.append(w)
        if curr:
            lines.append(" ".join(curr))
        wrapped_txt = "\n".join(lines)

        ax_txt.text(0.04, 0.50, f'"{wrapped_txt}"', fontsize=8.0, style="italic", color="#374151", va="top")
        ax_txt.text(0.04, 0.08, f"Manifest: {item['manifest']} (idx {item['manifest_idx']})", fontsize=7.0, color="#6b7280")

        # Col 2: Target Grid
        ax_tgt = fig.add_subplot(gs[grid_row, 2])
        ax_tgt.set_xticks([]); ax_tgt.set_yticks([])
        img_tgt = render_grid_pil(item["tgt_grid"], cell_px=14)
        ax_tgt.imshow(img_tgt)
        for spine in ax_tgt.spines.values():
            spine.set_color("#d1d5db")
            spine.set_linewidth(1.0)
        ax_tgt.set_xlabel(f"target: {item['tgt_name']}", fontsize=7.5, color="#4b5563", labelpad=3)

        # Col 3: LoRA DG Prediction
        ax_dg = fig.add_subplot(gs[grid_row, 3])
        ax_dg.set_xticks([]); ax_dg.set_yticks([])
        img_dg = render_grid_pil(item["pred_dg"], cell_px=14, error_mask=item["err_dg_mask"])
        ax_dg.imshow(img_dg)
        for spine in ax_dg.spines.values():
            spine.set_color("#10b981" if item["hit_22_dg"] else "#ef4444")
            spine.set_linewidth(1.2)

        b_22 = "22w: HIT" if item["hit_22_dg"] else "22w: MISS"
        b_ex = "Exact: HIT" if item["exact_dg"] else f"Exact: MISS ({item['err_dg']}e)"
        c_22 = "#059669" if item["hit_22_dg"] else "#dc2626"
        c_ex = "#059669" if item["exact_dg"] else "#dc2626"
        ax_dg.set_xlabel(f"{b_22} | {b_ex}", fontsize=7.5, fontweight="bold", color=c_22, labelpad=3)

        # Col 4: Non-DG Prediction
        ax_ndg = fig.add_subplot(gs[grid_row, 4])
        ax_ndg.set_xticks([]); ax_ndg.set_yticks([])
        img_ndg = render_grid_pil(item["pred_ndg"], cell_px=14, error_mask=item["err_ndg_mask"])
        ax_ndg.imshow(img_ndg)
        for spine in ax_ndg.spines.values():
            spine.set_color("#10b981" if item["hit_22_ndg"] else "#ef4444")
            spine.set_linewidth(1.2)

        nb_22 = "22w: HIT" if item["hit_22_ndg"] else "22w: MISS"
        nb_ex = "Exact: HIT" if item["exact_ndg"] else f"Exact: MISS ({item['err_ndg']}e)"
        nc_22 = "#059669" if item["hit_22_ndg"] else "#dc2626"
        nc_ex = "#059669" if item["exact_ndg"] else "#dc2626"
        ax_ndg.set_xlabel(f"{nb_22} | {nb_ex}", fontsize=7.5, fontweight="bold", color=nc_22, labelpad=3)

    # Row 10: Footer Caption
    caption_ax = fig.add_subplot(gs[10, :])
    caption_ax.axis("off")
    caption_text = (
        "Deterministic Row Rule (NO cherry-picking): Each row displays the first occurrence (smallest manifest index) of the operation in its test manifest (Seed 42).\n"
        "Red outlines on predictions indicate cell-level errors (pred[r, c] != tgt[r, c]). Badges indicate 22-way nearest candidate hit/miss and exact 256-cell match.\n"
        "Rows 1–4: Seen training operations. Row 5: Syntactic paraphrase unseen during training. Row 6–7: Indirect intent inference queries. Row 8: Held-out operation."
    )
    caption_ax.text(0.5, 0.5, caption_text, ha="center", va="center", fontsize=8.0, color="#4b5563", style="italic")

    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"[Figure 1 Generated] Successfully saved: {out_png} ({out_png.stat().st_size / 1024:.1f} KB)")
    return out_png


if __name__ == "__main__":
    generate_figure1()
