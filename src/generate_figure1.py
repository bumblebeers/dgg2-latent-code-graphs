#!/usr/bin/env python3
"""Generate Figure 1: Shared 256-Token Canvas Hero Visual.

Renders a balanced, zero-overclaim 1920x960 publication graphic comparing
natural-image patch transformations, AST call-graph adjacency transitions
(22-way & exact-grid metrics), and the Exp A vs. Exp B architectural boundary.
"""
import os
import sys

sys.path.insert(0, os.path.abspath("."))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cg.cg_ops import render


def create_figure1():
    os.makedirs("docs/figures", exist_ok=True)

    W, H = 1920, 960
    canvas = Image.new("RGB", (W, H), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)

    font_path_bold = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf"
    font_path_reg = "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf"

    font_title = ImageFont.truetype(font_path_bold, 22)
    font_header_right = ImageFont.truetype(font_path_bold, 15)
    font_sec = ImageFont.truetype(font_path_bold, 16)
    font_sub_bold = ImageFont.truetype(font_path_bold, 13)
    font_sub_reg = ImageFont.truetype(font_path_reg, 12)
    font_card_head = ImageFont.truetype(font_path_bold, 13)
    font_card_body = ImageFont.truetype(font_path_reg, 12)
    font_card_accent = ImageFont.truetype(font_path_bold, 12)
    font_pill = ImageFont.truetype(font_path_bold, 12)

    # 1. Top Header Bar
    draw.rectangle([0, 0, W, 72], fill=(15, 23, 42))
    draw.text(
        (40, 24),
        "Figure 1: Shared 256-Token Canvas — From Natural Images to AST Adjacency Grids",
        font=font_title,
        fill=(255, 255, 255),
    )
    draw.text(
        (1430, 28),
        "DiffusionGemma 26B-A4B  |  256 Soft Tokens",
        font=font_header_right,
        fill=(148, 163, 184),
    )

    # 2. Panel A: Natural Image Domain
    card_a_x1, card_a_y1, card_a_x2, card_a_y2 = 40, 90, 940, 774
    draw.rounded_rectangle(
        [card_a_x1, card_a_y1, card_a_x2, card_a_y2],
        radius=10,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=2,
    )
    draw.rounded_rectangle(
        [card_a_x1, card_a_y1, card_a_x2, card_a_y1 + 42],
        radius=10,
        fill=(241, 245, 249),
        outline=(203, 213, 225),
        width=1,
    )
    draw.text(
        (card_a_x1 + 20, card_a_y1 + 11),
        "(A) Natural Images: 768x768 RGB Photo -> 16x16 SigLIP h_v Patches",
        font=font_sec,
        fill=(30, 41, 59),
    )

    p_bottle = os.path.expanduser(
        "~/workspaces/arc-agi-3/rendered/dgg2/t2i/docci/images/qual_dev_000031.webp"
    )
    p_decoded_rot = "outputs/qual_dev_000031_rot90_decoded.png"
    im_bottle = Image.open(p_bottle if os.path.exists(p_bottle) else p_decoded_rot).convert("RGB")
    if os.path.exists(p_decoded_rot):
        im_bottle_trans = Image.open(p_decoded_rot).convert("RGB")
        trans_label_main = "Decoded Output: rot90 Geometric Transition"
        trans_label_sub = "34.15M ViT-Conv Decoder from predicted 256 soft tokens"
    else:
        im_bottle_trans = im_bottle.transpose(Image.Transpose.ROTATE_90)
        trans_label_main = "Target Output: rot90 Geometric Transition"
        trans_label_sub = "Solved on 256-token canvas (100.0% seen, 89.5% unseen)"

    img_size = 390
    thumb_a1 = im_bottle.resize((img_size, img_size), Image.Resampling.LANCZOS)
    thumb_a2 = im_bottle_trans.resize((img_size, img_size), Image.Resampling.LANCZOS)
    canvas.paste(thumb_a1, (65, 148))
    canvas.paste(thumb_a2, (525, 148))

    draw.text((65, 548), "Input: DOCCI Photo (qual_dev_000031, 768x768 RGB)", font=font_sub_bold, fill=(30, 41, 59))
    draw.text((65, 567), "16x16 patches (48x48 px) -> frozen SigLIP -> embed_vision", font=font_sub_reg, fill=(100, 116, 139))
    draw.text((525, 548), trans_label_main, font=font_sub_bold, fill=(30, 41, 59))
    draw.text((525, 567), trans_label_sub, font=font_sub_reg, fill=(100, 116, 139))

    box_y1, box_y2 = 598, 758
    draw.rounded_rectangle([65, box_y1, 915, box_y2], radius=8, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    draw.text(
        (82, box_y1 + 12),
        "Natural-Image Findings (N=300 Held-Out DOCCI, 24-Way Candidate Pool):",
        font=font_card_head,
        fill=(30, 41, 59),
    )
    draw.text(
        (82, box_y1 + 36),
        "• Macroscopic patch routing: 91.3% seen & 90.5% unseen paraphrase accuracy (vs. 74.8% / 61.3% Non-DG).",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (82, box_y1 + 60),
        "• Sub-patch limit: interior vs. patch-permuted probes tie (p = 0.68), motivating 1-cell-per-patch grids.",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (82, box_y1 + 84),
        "• Single-pass geo x geo chaining fails (0.5%), while 2-step soft-token re-feeding reaches 88.3 ± 1.7%.",
        font=font_card_accent,
        fill=(185, 28, 28),
    )
    draw.text(
        (82, box_y1 + 108),
        "• Shuffled-instruction control drops to 0.0% because the model executes the mismatched edit faithfully.",
        font=font_card_body,
        fill=(51, 65, 85),
    )

    # 3. Panel B: Software Code Graph Domain
    card_b_x1, card_b_y1, card_b_x2, card_b_y2 = 980, 90, 1880, 774
    draw.rounded_rectangle(
        [card_b_x1, card_b_y1, card_b_x2, card_b_y2],
        radius=10,
        fill=(255, 255, 255),
        outline=(203, 213, 225),
        width=2,
    )
    draw.rounded_rectangle(
        [card_b_x1, card_b_y1, card_b_x2, card_b_y1 + 42],
        radius=10,
        fill=(241, 245, 249),
        outline=(203, 213, 225),
        width=1,
    )
    draw.text(
        (card_b_x1 + 20, card_b_y1 + 11),
        "(B) AST Call Graphs (Exp A): 16x16 Palette Grid -> 16x16 SigLIP h_v Cells",
        font=font_sec,
        fill=(30, 41, 59),
    )

    z = np.load("data/gridA_test360_httpx.npz")
    src_grid = z["src_grid"][0]
    tgt_del = z["tgt_grid"][0, 3]  # delete_selected
    thumb_b1 = Image.fromarray(render(src_grid, cell_px=48)).resize((img_size, img_size), Image.Resampling.NEAREST)
    thumb_b2 = Image.fromarray(render(tgt_del, cell_px=48)).resize((img_size, img_size), Image.Resampling.NEAREST)
    canvas.paste(thumb_b1, (1005, 148))
    canvas.paste(thumb_b2, (1465, 148))

    draw.text((1005, 548), "Source State: 16x16 Adjacency Grid (httpx, N=360)", font=font_sub_bold, fill=(30, 41, 59))
    draw.text((1005, 567), "Diagonal: symbol kind (func, method, test, class, selected)", font=font_sub_reg, fill=(100, 116, 139))
    draw.text((1465, 548), "Target State: delete_selected Transition", font=font_sub_bold, fill=(30, 41, 59))
    draw.text((1465, 567), "Selected red node & all incident call edges removed", font=font_sub_reg, fill=(100, 116, 139))

    draw.rounded_rectangle([1005, box_y1, 1855, box_y2], radius=8, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
    draw.text(
        (1022, box_y1 + 12),
        "Code-Graph Transition Results (httpx N=360, 22-Way Nearest / Exact-Grid Match):",
        font=font_card_head,
        fill=(30, 41, 59),
    )
    draw.text(
        (1022, box_y1 + 36),
        "• Gate A1 passes at 100.0% cell readability across all 8 classes (32x32/48x48 crushes 'selected' to 13.2%).",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (1022, box_y1 + 60),
        "• 2-Seed LoRA DG: 91.9% / 62.2% exact seen; 78.5% / 47.1% exact unseen (Non-DG exact: 7.2% = source copy).",
        font=font_card_accent,
        fill=(21, 128, 61),
    )
    draw.text(
        (1022, box_y1 + 84),
        "• Caller-decomposed 2-step re-feed: 82.1% 22-way / 47.8% exact (98.0% edge F1; Non-DG: 34.6% / 0.0% exact).",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (1022, box_y1 + 108),
        "• Open-ended edits commit to one valid branch (56.1% unseen vs. 11.4% Non-DG; ~1.01 distinct across seeds).",
        font=font_card_body,
        fill=(51, 65, 85),
    )

    # 4. Bottom Split Strip: (C) Transition Pipeline & (D) Exp A vs. Exp B Boundary
    bot_y1, bot_y2 = 792, 940
    # Left box (C)
    draw.rounded_rectangle([40, bot_y1, 940, bot_y2], radius=10, fill=(239, 246, 255), outline=(147, 197, 253), width=2)
    draw.text(
        (60, bot_y1 + 12),
        "(C) Stock DiffusionGemma Transition & Caller-Supplied Re-Feed Loop",
        font=font_card_head,
        fill=(30, 58, 138),
    )
    steps = [
        ("768² Grid / Photo", 60, 175),
        ("SigLIP + embed_vision", 250, 195),
        ("48-Step Bidir Denoise", 460, 195),
        ("Pointwise h_v Head", 670, 175),
    ]
    for label, sx, sw in steps:
        draw.rounded_rectangle(
            [sx, bot_y1 + 42, sx + sw, bot_y1 + 82],
            radius=6,
            fill=(255, 255, 255),
            outline=(96, 165, 250),
            width=2,
        )
        draw.text((sx + sw // 2, bot_y1 + 62), label, font=font_pill, fill=(30, 58, 138), anchor="mm")
    for ax in [238, 448, 658]:
        draw.text((ax + 4, bot_y1 + 62), "->", font=font_pill, fill=(37, 99, 235), anchor="mm")
    draw.text(
        (60, bot_y1 + 98),
        "Iterative Re-Feed: predicted h_v^(t) re-enters frozen embed_vision for step t+1; Ridge reader decodes 16x16 graph.",
        font=font_card_body,
        fill=(30, 41, 59),
    )
    draw.text(
        (60, bot_y1 + 120),
        "Zero-shot cross-repo transfer to click (N=360, s42): 88.3% seen (61.4% exact), 72.5% unseen (vs. 46.1% Non-DG).",
        font=font_card_accent,
        fill=(30, 58, 138),
    )

    # Right box (D)
    draw.rounded_rectangle([980, bot_y1, 1880, bot_y2], radius=10, fill=(254, 242, 242), outline=(252, 165, 165), width=2)
    draw.text(
        (1000, bot_y1 + 12),
        "(D) Architectural Boundary: Perceptual Grid (Exp A) vs. Cold-Start Symbol Latents (Exp B)",
        font=font_card_head,
        fill=(153, 27, 27),
    )
    draw.text(
        (1000, bot_y1 + 40),
        "• Exp A (Frozen SigLIP + Rich Codebook): Co-pretrained vision grounding preserves unedited cells (99.0% unchanged).",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (1000, bot_y1 + 66),
        "• Exp B (Trained Input Adapter, 256 x 384-d MiniLM): Reconstructing all 256 vectors across a 3-symbol vocab (_/+/-)",
        font=font_card_body,
        fill=(51, 65, 85),
    )
    draw.text(
        (1000, bot_y1 + 88),
        "  without a slot-wise residual skip incurs 18.0x unchanged-slot drift (100% delete recall vs. 2.2% non-delete edits).",
        font=font_card_accent,
        fill=(153, 27, 27),
    )
    draw.text(
        (1000, bot_y1 + 114),
        "• Contrast: Scratch 50M Non-DG transformer with slot residual (x + SA + CA + FFN) reaches 47.4% seen on Exp B.",
        font=font_card_body,
        fill=(51, 65, 85),
    )

    out_path = "docs/figures/figure1_hero_multimodal_roadmap.png"
    canvas.save(out_path, quality=95)
    print(f"Figure 1 successfully generated at {out_path} ({W}x{H})")


if __name__ == "__main__":
    create_figure1()
