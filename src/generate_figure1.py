#!/usr/bin/env python3
"""
Generate Figure 4 & Hero Flowchart: DiffusionGemma Shared 256-Token Latent Pipeline.

Produces a high-DPI vector-rendered diagram illustrating:
  - Shared spine: 768x768 RGB (AST grid or photo) -> SigLIP -> embed_vision ->
    256-token canvas + instruction -> DiffusionGemma 26B-A4B (frozen base + trained LoRA) ->
    trained pointwise readout head -> predicted h_v -> Ridge reader (grids) / ImageDecoder (photos)
  - Tensor shapes cleanly positioned on every arrow without colliding with boxes
  - Explicit frozen (slate) vs. trained (coral amber) color coding and status badges
  - Training lane (multinomial corruption, self-conditioning, CE + latent regression loss)
  - Inference lane (up to 48 discrete denoising steps, final self-conditioned readout, re-feed loop)
  - Experiment B inset (MiniLM-L6-v2 + trained LayerNorm+Linear adapter failure mode)

Outputs:
  - docs/figures/fig4_pipeline_flowchart.png
  - docs/figures/figure1_hero_multimodal_roadmap.png

Author: bumblebeers
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_pipeline_flowchart() -> None:
    # Exact 1:1 coordinate aspect (180 x 105 units on 18.0 x 10.5 in @ 300 DPI = 5400 x 3150 px)
    fig, ax = plt.subplots(figsize=(18.0, 10.5), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 180)
    ax.set_ylim(0, 105)
    ax.axis("off")

    # Palette
    C_BG_FROZEN = "#F1F5F9"
    C_BD_FROZEN = "#334155"
    C_TX_FROZEN = "#0F172A"

    C_BG_TRAIN = "#FFEDD5"
    C_BD_TRAIN = "#EA580C"
    C_TX_TRAIN = "#9A3412"

    C_ARROW = "#1E293B"
    C_REFEED = "#6D28D9"
    C_TLANE = "#15803D"

    def rbox(x: float, y: float, w: float, h: float, fc: str, ec: str, lw: float = 1.4, ls: str = "-", r: float = 0.9) -> None:
        p = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0.0,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls,
        )
        ax.add_patch(p)

    def badge(cx: float, cy: float, text: str, bg: str, fg: str = "#FFFFFF", w: float = 8.4, h: float = 2.1, fs: float = 6.2) -> None:
        rbox(cx - w / 2, cy - h / 2, w, h, fc=bg, ec="none", lw=0.0, r=0.5)
        ax.text(cx, cy, text, fontsize=fs, fontweight="bold", ha="center", va="center", color=fg)

    def spine_module(
        x: float, y: float, w: float, h: float,
        title: str, sub1: str, sub2: str,
        trained: bool = False, tag: str | None = None,
    ) -> None:
        fc = C_BG_TRAIN if trained else C_BG_FROZEN
        ec = C_BD_TRAIN if trained else C_BD_FROZEN
        tc = C_TX_TRAIN if trained else C_TX_FROZEN
        rbox(x, y, w, h, fc=fc, ec=ec, lw=1.5)
        b_txt = tag if tag else ("TRAINED" if trained else "FROZEN")
        b_col = "#EA580C" if trained else "#475569"
        badge(x + w / 2, y + h - 1.6, b_txt, bg=b_col, w=min(w - 2.0, 9.6), h=2.0, fs=6.0)
        ax.text(x + w / 2, y + h * 0.60, title, fontsize=8.8, fontweight="bold", ha="center", va="center", color=tc)
        ax.text(x + w / 2, y + h * 0.38, sub1, fontsize=7.0, ha="center", va="center", color="#334155")
        ax.text(x + w / 2, y + h * 0.17, sub2, fontsize=7.0, fontweight="bold", ha="center", va="center", color="#0369A1")

    def h_arrow(x1: float, x2: float, y: float, top_label: str = "", bot_label: str = "", color: str = C_ARROW) -> None:
        arr = patches.FancyArrowPatch(
            (x1 + 0.2, y), (x2 - 0.2, y),
            arrowstyle="-|>", mutation_scale=12, color=color, linewidth=1.7,
        )
        ax.add_patch(arr)
        mx = 0.5 * (x1 + x2)
        if top_label:
            ax.text(
                mx, y + 2.2, top_label,
                fontsize=6.6, fontweight="bold", ha="center", va="center", color="#0369A1",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=0.9),
            )
        if bot_label:
            ax.text(mx, y - 2.1, bot_label, fontsize=6.2, ha="center", va="center", color="#475569")

    # ------------------------------------------------------------------
    # 1. Title & Legend Strip
    # ------------------------------------------------------------------
    ax.text(
        90, 102.6,
        "DiffusionGemma 26B-A4B Shared 256-Token Latent Pipeline",
        fontsize=16.0, fontweight="bold", ha="center", va="center", color="#0F172A",
    )
    ax.text(
        90, 99.6,
        "Training & Inference Lanes Across 16x16 AST Code Graphs and 768x768 Natural Photographs",
        fontsize=10.2, ha="center", va="center", color="#475569",
    )

    leg_y = 95.4
    rbox(5.5, leg_y - 1.4, 41.5, 2.8, fc=C_BG_FROZEN, ec=C_BD_FROZEN, lw=1.2, r=0.6)
    ax.text(26.25, leg_y, "FROZEN: SigLIP, embed_vision, 26B Base, Readers", fontsize=6.9, fontweight="bold", ha="center", va="center", color=C_TX_FROZEN)

    rbox(49.5, leg_y - 1.4, 43.0, 2.8, fc=C_BG_TRAIN, ec=C_BD_TRAIN, lw=1.2, r=0.6)
    ax.text(71.0, leg_y, "TRAINED ONLY: LoRA Adapter (r=64) + Readout Head", fontsize=6.9, fontweight="bold", ha="center", va="center", color=C_TX_TRAIN)

    arr_leg_t = patches.FancyArrowPatch((95.0, leg_y), (100.5, leg_y), arrowstyle="-|>", mutation_scale=11, color=C_TLANE, linewidth=2.0)
    ax.add_patch(arr_leg_t)
    ax.text(101.6, leg_y, "Training Corruption & Loss", fontsize=7.2, fontweight="bold", ha="left", va="center", color=C_TLANE)

    arr_leg_r = patches.FancyArrowPatch((133.5, leg_y), (139.0, leg_y), arrowstyle="-|>", mutation_scale=11, color=C_REFEED, linewidth=2.2)
    ax.add_patch(arr_leg_r)
    ax.text(140.1, leg_y, "Inference Soft-Token Re-Feed", fontsize=7.2, fontweight="bold", ha="left", va="center", color=C_REFEED)

    # ------------------------------------------------------------------
    # 2. Top Lane: TRAINING LANE (y = 78.5 to 91.5)
    # ------------------------------------------------------------------
    rbox(26.5, 78.5, 129.0, 13.0, fc="#F0FDF4", ec="#4ADE80", lw=1.3, r=1.0)
    ax.text(
        29.0, 89.6,
        "TRAINING LANE — Discrete Multinomial Corruption, Self-Conditioning & Target Latent Supervision",
        fontsize=8.6, fontweight="bold", ha="left", va="center", color="#14532D",
    )

    t_cards = [
        (29.0, 39.0, "1. Multinomial Canvas Corruption", "Corrupt 256-token target canvas under stock\ndiscrete schedule; source image stays clean."),
        (71.0, 39.0, "2. Self-Conditioning (p = 0.5)", "Feed prior-pass logits on 50% of steps\n(when off, pass None/False — never zeros)."),
        (113.0, 40.0, "3. Token CE + Latent Regression", "Full-vocab token CE on stock LM head +\ncontinuous loss against clean target h_v."),
    ]
    for tx, tw, t_head, t_body in t_cards:
        rbox(tx, 79.8, tw, 7.8, fc="#FFFFFF", ec="#86EFAC", lw=1.0, r=0.7)
        ax.text(tx + 1.5, 85.9, t_head, fontsize=7.5, fontweight="bold", ha="left", va="center", color="#166534")
        ax.text(tx + 1.5, 84.1, t_body, fontsize=6.7, ha="left", va="top", color="#1E293B", linespacing=1.25)

    # Downward training arrow into Canvas (x = 89.0)
    arr_t_in = patches.FancyArrowPatch((89.0, 78.5), (89.0, 70.2), arrowstyle="-|>", mutation_scale=11, color=C_TLANE, linewidth=1.7)
    ax.add_patch(arr_t_in)
    ax.text(
        89.0, 74.4, "Corrupted 256 Tokens\n+ Self-Cond",
        fontsize=6.3, fontweight="bold", ha="center", va="center", color="#166534",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#86EFAC", lw=0.9),
    )

    # Upward readout loss arrow from Readout Head (x = 147.0)
    arr_t_loss = patches.FancyArrowPatch((147.0, 69.2), (147.0, 78.5), arrowstyle="-|>", mutation_scale=11, color=C_TLANE, linewidth=1.7)
    ax.add_patch(arr_t_loss)
    ax.text(
        147.0, 74.0, "Pred h_v vs.\nClean Target h_v",
        fontsize=6.3, fontweight="bold", ha="center", va="center", color="#166534",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#86EFAC", lw=0.9),
    )

    # ------------------------------------------------------------------
    # 3. Shared Horizontal Spine (Center Y = 61.5)
    # ------------------------------------------------------------------
    sy = 61.5

    # Dual Domain Inputs (x = 2.0 to 19.0)
    rbox(2.0, 62.5, 17.0, 8.2, fc="#FFFFFF", ec="#64748B", lw=1.3, r=0.8)
    ax.text(10.5, 68.7, "16x16 AST Grid", fontsize=8.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(10.5, 66.4, "8 Cell Classes (48 px/cell)", fontsize=6.6, ha="center", va="center", color="#475569")
    ax.text(10.5, 64.2, "768 x 768 x 3 RGB", fontsize=6.7, fontweight="bold", ha="center", va="center", color="#0369A1")

    rbox(2.0, 52.3, 17.0, 8.2, fc="#FFFFFF", ec="#64748B", lw=1.3, r=0.8)
    ax.text(10.5, 58.5, "Natural Photo", fontsize=8.2, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(10.5, 56.2, "DOCCI / Eval Image", fontsize=6.6, ha="center", va="center", color="#475569")
    ax.text(10.5, 54.0, "768 x 768 x 3 RGB", fontsize=6.7, fontweight="bold", ha="center", va="center", color="#0369A1")

    # Converging input arrows into SigLIP (x = 19.0 -> 26.5)
    ax.add_patch(patches.FancyArrowPatch((19.2, 66.5), (26.3, 63.2), arrowstyle="-|>", mutation_scale=11, color=C_ARROW, linewidth=1.5))
    ax.add_patch(patches.FancyArrowPatch((19.2, 56.5), (26.3, 59.8), arrowstyle="-|>", mutation_scale=11, color=C_ARROW, linewidth=1.5))
    ax.text(
        22.6, 61.5, "768²x3",
        fontsize=6.3, fontweight="bold", ha="center", va="center", color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=0.8),
    )

    # Box 1: SigLIP Vision Tower [26.5, 44.5]
    spine_module(
        26.5, 54.0, 18.0, 15.0,
        "SigLIP Vision", "AutoProcessor (16x16)", "h_v : 256 x 1152",
        trained=False, tag="FROZEN",
    )

    # Arrow 1 -> 2 [44.5 -> 53.5]
    h_arrow(44.5, 53.5, sy, top_label="256 x 1152", bot_label="patch h_v")

    # Box 2: embed_vision [53.5, 70.5]
    spine_module(
        53.5, 54.0, 17.0, 15.0,
        "embed_vision", "Vision Projector", "256 x 2816 softs",
        trained=False, tag="FROZEN",
    )

    # Arrow 2 -> 3 [70.5 -> 79.5]
    h_arrow(70.5, 79.5, sy, top_label="256 x 2816", bot_label="soft tokens")

    # Box 3: 256-Token Canvas + Instruction [79.5, 98.5]
    rbox(79.5, 53.0, 19.0, 17.0, fc="#F8FAFC", ec="#475569", lw=1.5)
    badge(89.0, 68.4, "SHARED CANVAS", bg="#334155", w=12.5, h=2.0, fs=5.9)
    ax.text(89.0, 65.1, "256-Token Canvas", fontsize=8.4, fontweight="bold", ha="center", va="center", color="#0F172A")
    ax.text(89.0, 62.5, "256 Vision Soft Tokens", fontsize=6.6, ha="center", va="center", color="#334155")
    ax.text(89.0, 60.1, "+ Edit Instruction", fontsize=6.9, fontweight="bold", ha="center", va="center", color="#6D28D9")
    ax.text(89.0, 57.6, "dg_attn_fix Mask", fontsize=6.5, ha="center", va="center", color="#475569")
    ax.text(89.0, 55.1, "Canvas : 256 x 2816", fontsize=6.7, fontweight="bold", ha="center", va="center", color="#0369A1")

    # Arrow 3 -> 4 [98.5 -> 107.5]
    h_arrow(98.5, 107.5, sy, top_label="256 x 2816", bot_label="+ prompt")

    # Box 4: DiffusionGemma 26B-A4B [107.5, 129.5]
    rbox(107.5, 51.8, 22.0, 19.4, fc=C_BG_FROZEN, ec=C_BD_FROZEN, lw=1.6)
    badge(118.5, 69.6, "FROZEN 26B BASE", bg="#475569", w=13.5, h=2.0, fs=5.9)
    ax.text(118.5, 66.7, "DiffusionGemma 26B", fontsize=8.2, fontweight="bold", ha="center", va="center", color=C_TX_FROZEN)
    ax.text(118.5, 64.4, "3.8B Active / 25.2B MoE", fontsize=6.5, ha="center", va="center", color="#475569")

    # Nested Trained LoRA Box inside DG
    rbox(109.0, 53.0, 19.0, 9.6, fc=C_BG_TRAIN, ec=C_BD_TRAIN, lw=1.4, r=0.7)
    badge(118.5, 61.1, "TRAINED LoRA", bg="#EA580C", w=11.0, h=1.8, fs=5.8)
    ax.text(118.5, 58.3, "LoRA (r=64, ~150M)", fontsize=7.2, fontweight="bold", ha="center", va="center", color=C_TX_TRAIN)
    ax.text(118.5, 56.0, "48 Multinomial Steps", fontsize=6.4, ha="center", va="center", color="#7C2D12")
    ax.text(118.5, 54.1, "Self-Cond Final Readout", fontsize=6.2, fontweight="bold", ha="center", va="center", color="#C2410C")

    # Arrow 4 -> 5 [129.5 -> 138.5]
    h_arrow(129.5, 138.5, sy, top_label="256 x 2816", bot_label="hidden state")

    # Box 5: Pointwise Readout Head [138.5, 155.5]
    spine_module(
        138.5, 54.0, 17.0, 15.0,
        "Readout Head", "Pointwise MLP Projector", "Pred h_v : 256 x 1152",
        trained=True, tag="TRAINED",
    )

    # Fork Arrows from Readout Head [155.5 -> 163.0]
    ax.add_patch(patches.FancyArrowPatch((155.7, 63.5), (162.8, 66.5), arrowstyle="-|>", mutation_scale=11, color=C_ARROW, linewidth=1.5))
    ax.add_patch(patches.FancyArrowPatch((155.7, 59.5), (162.8, 56.5), arrowstyle="-|>", mutation_scale=11, color=C_ARROW, linewidth=1.5))
    ax.text(
        159.2, 61.5, "256x1152",
        fontsize=6.2, fontweight="bold", ha="center", va="center", color="#0369A1",
        bbox=dict(boxstyle="round,pad=0.16", facecolor="#FFFFFF", edgecolor="#BAE6FD", lw=0.8),
    )

    # Dual Domain Decoders [163.0, 178.0]
    rbox(163.0, 62.5, 15.0, 8.6, fc=C_BG_FROZEN, ec=C_BD_FROZEN, lw=1.3, r=0.8)
    badge(170.5, 69.7, "FROZEN", bg="#475569", w=7.2, h=1.7, fs=5.5)
    ax.text(170.5, 67.3, "Ridge Reader", fontsize=7.8, fontweight="bold", ha="center", va="center", color=C_TX_FROZEN)
    ax.text(170.5, 65.3, "8-Class Linear Probe", fontsize=6.2, ha="center", va="center", color="#475569")
    ax.text(170.5, 63.5, "-> 16x16 AST Grid", fontsize=6.5, fontweight="bold", ha="center", va="center", color="#15803D")

    rbox(163.0, 51.9, 15.0, 8.6, fc=C_BG_FROZEN, ec=C_BD_FROZEN, lw=1.3, r=0.8)
    badge(170.5, 59.1, "FROZEN", bg="#475569", w=7.2, h=1.7, fs=5.5)
    ax.text(170.5, 56.7, "ImageDecoder", fontsize=7.8, fontweight="bold", ha="center", va="center", color=C_TX_FROZEN)
    ax.text(170.5, 54.7, "34.15M ViT+Conv", fontsize=6.2, ha="center", va="center", color="#475569")
    ax.text(170.5, 52.9, "-> 768x768 RGB", fontsize=6.5, fontweight="bold", ha="center", va="center", color="#0369A1")

    # ------------------------------------------------------------------
    # 4. Compositional Re-Feed Loop (Cleanly Routed at Y = 46.8)
    # ------------------------------------------------------------------
    rf_x1, rf_x2, rf_y = 147.0, 62.0, 46.8
    ax.plot([rf_x1, rf_x1], [54.0, rf_y], color=C_REFEED, lw=2.3, zorder=2)
    ax.plot([rf_x1, rf_x2], [rf_y, rf_y], color=C_REFEED, lw=2.3, zorder=2)
    ax.add_patch(patches.FancyArrowPatch((rf_x2, rf_y), (rf_x2, 54.0), arrowstyle="-|>", mutation_scale=14, color=C_REFEED, linewidth=2.3, zorder=3))

    ax.text(
        104.5, rf_y,
        "Compositional Soft-Token Re-Feed: Pred h_v (256 x 1152) -> Frozen embed_vision (Step k -> Step k+1)",
        fontsize=6.9, fontweight="bold", ha="center", va="center", color=C_REFEED, zorder=5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5F3FF", edgecolor="#C4B5FD", lw=1.2),
    )

    # ------------------------------------------------------------------
    # 5. Bottom-Left Inset: Experiment B Diagnostic [2.0, 86.5] x [3.5, 42.0]
    # ------------------------------------------------------------------
    rbox(2.0, 3.5, 84.5, 38.5, fc="#FEF2F2", ec="#F87171", lw=1.4, r=1.0)
    ax.text(
        4.5, 39.8,
        "INSET — Experiment B: Non-Visual Symbol Latents via Trained Adapter (Failed Variant)",
        fontsize=8.1, fontweight="bold", ha="left", va="center", color="#991B1B",
    )
    ax.text(
        4.5, 37.5,
        "Replaces the frozen SigLIP + embed_vision vision spine with frozen MiniLM-L6-v2 and a trained input projector",
        fontsize=7.0, ha="left", va="center", color="#7F1D1D",
    )

    # Mini flow inside Exp B (y = 24.5 to 35.2)
    spine_module(4.5, 24.5, 21.5, 10.7, "all-MiniLM-L6-v2", "256 AST Symbol Stubs", "z : 256 x 384", trained=False, tag="FROZEN")
    h_arrow(26.0, 34.0, 29.85, top_label="256 x 384")
    spine_module(34.0, 24.5, 22.0, 10.7, "LayerNorm + Linear", "Modified Input Adapter", "256 x 2816", trained=True, tag="TRAINED")
    h_arrow(56.0, 64.0, 29.85, top_label="256 x 2816")
    spine_module(64.0, 24.5, 20.0, 10.7, "DG + 384-d Head", "3-Symbol Canvas {_,+,-}", "Pred z : 256 x 384", trained=True, tag="TRAINED")

    # Exp B outcome card
    rbox(4.5, 5.5, 79.5, 17.0, fc="#FFFFFF", ec="#EF4444", lw=1.2, ls="--", r=0.8)
    ax.text(6.5, 20.4, "Diagnostic Outcome Across Two Seeds (Seeds 42 & 1337):", fontsize=7.8, fontweight="bold", ha="left", va="center", color="#B91C1C")
    expb_lines = [
        "• 8-Way Centered Cosine: 26.4% ± 0.3% seen / 27.6% ± 1.0% unseen — at constant mean-vector floor (25.0%).",
        "• Delete vs. Non-Delete Split: 100% accuracy on delete operations vs. 1.5%–4.8% on non-delete operations.",
        "• Unchanged-Slot Drift: 17.9x ± 0.1x higher drift than target (despite 98.5% ± 0.3% 3-class canvas token accuracy).",
        "• Takeaway: Suggests a slot-wise residual skip or pretraining-aligned input projector is needed beyond frozen vision.",
    ]
    for idx, line in enumerate(expb_lines):
        ax.text(6.5, 17.6 - idx * 3.3, line, fontsize=6.9, ha="left", va="center", color="#7F1D1D")

    # ------------------------------------------------------------------
    # 6. Bottom-Right Panel: Inference & Re-Feed Lane [89.5, 178.0] x [3.5, 42.0]
    # ------------------------------------------------------------------
    rbox(89.5, 3.5, 88.5, 38.5, fc="#F8FAFC", ec="#94A3B8", lw=1.4, r=1.0)
    ax.text(
        92.0, 39.8,
        "INFERENCE LANE — Discrete Denoising Readout & Multi-Pass Soft-Token Re-Feeding",
        fontsize=8.8, fontweight="bold", ha="left", va="center", color="#0F172A",
    )
    ax.text(
        92.0, 37.5,
        "Identical stock inference contract across AST adjacency grids (§4–§5) and natural images (§3)",
        fontsize=7.0, ha="left", va="center", color="#475569",
    )

    inf_cards = [
        (
            "1. Bidirectional Multimodal Prefill (dg_attn_fix / Upstream PR #50684)",
            "Restores dropped prefill_mm_mask in DiffusionGemmaEncoderModel so 256 image soft tokens attend bidirectionally.",
        ),
        (
            "2. Up to 48 Discrete Multinomial Denoising Steps + Self-Conditioned Readout",
            "Runs stock discrete token denoising with self-conditioning active; reads final-step hidden state via stock_generate_with_readout.",
        ),
        (
            "3. Continuous Soft-Token Re-Feeding (Predicted h_v -> Frozen embed_vision)",
            "Re-embeds predicted 256x1152 patch latents directly into soft tokens with the next edit prompt, avoiding intermediate quantization.",
        ),
        (
            "4. Within-Model Recovery: 1-Pass vs. Multi-Step Soft-Token Re-Feed",
            "Grids: 48.0% -> 82.1% (2-step compounds), 0.0% -> 90.8% (3-step keep_callers). Photos: 0.5% -> 88.3% (flip_h+rot90).",
        ),
    ]
    for idx, (c_title, c_body) in enumerate(inf_cards):
        cy_top = 35.2 - idx * 7.6
        rbox(92.0, cy_top - 6.7, 83.5, 6.7, fc="#FFFFFF", ec="#CBD5E1", lw=1.0, r=0.7)
        ax.text(93.8, cy_top - 1.8, c_title, fontsize=7.3, fontweight="bold", ha="left", va="center", color="#0F172A")
        ax.text(93.8, cy_top - 4.4, c_body, fontsize=6.5, ha="left", va="center", color="#334155")

    out_paths = [
        REPO_ROOT / "docs" / "figures" / "fig4_pipeline_flowchart.png",
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.12)
        print(f"[OK] Saved {p} ({p.stat().st_size / 1024:.1f} KiB)")

    plt.close(fig)


if __name__ == "__main__":
    generate_pipeline_flowchart()
