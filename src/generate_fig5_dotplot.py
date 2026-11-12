#!/usr/bin/env python3
"""
Generate Figure 5: Table 5 Cleveland Dot Plot (22-Way vs. Exact 256-Cell Match).

Loads every metric dynamically from committed JSON evaluation summaries (zero hardcoded numbers):
  - outputs/m2/offline_verification_report.json (2-seed LoRA DG)
  - outputs/m2/nondg_baseline_manifest_summary.json (Best 2-seed Non-DG & Source Copy)
  - outputs/m2/arms/gemma4_ar_lora_r64_s42/*_summary.json (Gemma 4 AR LoRA r=64)
  - outputs/m2/arms/gemma4_ar_prompted/*_summary.json (Gemma 4 AR Prompted)
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_table5_data() -> list[dict]:
    dg = json.load(open(REPO_ROOT / "outputs" / "m2" / "offline_verification_report.json"))
    nd = json.load(open(REPO_ROOT / "outputs" / "m2" / "nondg_baseline_manifest_summary.json"))

    splits = [
        ("seen_bal360", "Seen Operations\n(seen_bal360)"),
        ("unseen_para_bal360", "Unseen Paraphrases\n(unseen_para_bal360)"),
        ("unseen_infer_360", "Indirect Intent\n(unseen_infer_360)"),
        ("refeed_heldout_comp_bal360", "2-Step Re-Feed\n(refeed_heldout_comp)"),
        ("refeed_chain_keep_callers_180", "3-Step Mirror Chain\n(keep_callers, n=180)"),
    ]

    rows = []
    for key, label in splits:
        # 2-seed LoRA DG
        s42 = dg["cg_lora_dg_seed42_ep3"][key]
        s1337 = dg["cg_lora_dg_seed1337_ep3"][key]
        dg_22 = (s42["acc_kway"] + s1337["acc_kway"]) * 50.0
        dg_ex = (s42["exact"] + s1337["exact"]) * 50.0

        # Best 2-seed Non-DG
        nd_key = "refeed_chain_heldout_ops_360" if key == "refeed_chain_keep_callers_180" else key
        enc_22 = (nd["nondg_enc_lora_s42"][nd_key]["acc_kway"] + nd["nondg_enc_lora_s1337"][nd_key]["acc_kway"]) * 50.0
        frz_22 = (nd["nondg_frozen_inst_s42"][nd_key]["acc_kway"] + nd["nondg_frozen_inst_s1337"][nd_key]["acc_kway"]) * 50.0
        if enc_22 >= frz_22:
            best_nd_22 = enc_22
            best_nd_ex = (nd["nondg_enc_lora_s42"][nd_key]["exact"] + nd["nondg_enc_lora_s1337"][nd_key]["exact"]) * 50.0
        else:
            best_nd_22 = frz_22
            best_nd_ex = (nd["nondg_frozen_inst_s42"][nd_key]["exact"] + nd["nondg_frozen_inst_s1337"][nd_key]["exact"]) * 50.0

        # Gemma 4 AR LoRA & Prompted (None when not run)
        ar_lora_p = REPO_ROOT / "outputs" / "m2" / "arms" / "gemma4_ar_lora_r64_s42" / f"{key}_summary.json"
        ar_prom_p = REPO_ROOT / "outputs" / "m2" / "arms" / "gemma4_ar_prompted" / f"{key}_summary.json"
        ar_lora_22 = ar_lora_ex = None
        ar_prom_22 = ar_prom_ex = None
        if ar_lora_p.exists():
            d = json.load(open(ar_lora_p))
            ar_lora_22 = d["acc_kway"] * 100.0
            ar_lora_ex = d["exact"] * 100.0
        if ar_prom_p.exists():
            d = json.load(open(ar_prom_p))
            ar_prom_22 = d["acc_kway"] * 100.0
            ar_prom_ex = d["exact"] * 100.0

        rows.append(
            {
                "key": key,
                "label": label,
                "dg": (dg_22, dg_ex),
                "ar_lora": (ar_lora_22, ar_lora_ex),
                "nondg": (best_nd_22, best_nd_ex),
                "ar_prom": (ar_prom_22, ar_prom_ex),
            }
        )
    return rows


def generate_fig5() -> Path:
    out_png = REPO_ROOT / "docs" / "figures" / "fig5_accuracy_dotplot.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rows = load_table5_data()

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
        }
    )

    fig, (ax_k, ax_e) = plt.subplots(1, 2, figsize=(15.0, 6.8), dpi=300, sharey=True)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.75, bottom=0.11, wspace=0.10)

    fig.suptitle(
        "Code-Graph Evaluation Summary: 22-Way Selection vs. 256-Cell Exact Match (httpx, N=360)",
        fontsize=13.0,
        fontweight="bold",
        color="#0f172a",
        y=0.96,
    )

    y_pos = np.arange(len(rows))[::-1]
    offsets = {
        "dg": +0.26,
        "ar_lora": +0.09,
        "nondg": -0.09,
        "ar_prom": -0.26,
    }
    styles = [
        ("dg", "LoRA DG (2-Seed Mean)", "#059669", "o", 80),
        ("ar_lora", "Gemma 4 AR LoRA (r=64, 354 steps)", "#2563eb", "D", 64),
        ("nondg", "Best Non-DG (~50M, 2-Seed)", "#d97706", "s", 64),
        ("ar_prom", "Gemma 4 AR Prompted (2-Shot)", "#64748b", "^", 60),
    ]

    for panel_idx, (ax, title) in enumerate(
        [
            (ax_k, "A. 22-Way Nearest-Candidate Accuracy (%)"),
            (ax_e, "B. Exact 256-Cell Grid Match (%)"),
        ]
    ):
        ax.set_title(title, fontsize=11.0, fontweight="bold", color="#1e293b", pad=12)
        ax.set_xlim(-4, 109)
        ax.set_ylim(-0.55, len(rows) - 0.45)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="x", linestyle="--", alpha=0.45, color="#cbd5e1")
        for y in range(len(rows) - 1):
            ax.axhline(y + 0.5, color="#e2e8f0", linewidth=0.9)
        if panel_idx == 0:
            ax.axvline(100.0 / 22.0, color="#94a3b8", linestyle=":", linewidth=1.2, label="Chance (4.5%)")

        for arm_key, arm_label, color, marker, sz in styles:
            xs, ys = [], []
            for y, r in zip(y_pos, rows):
                val = r[arm_key][panel_idx]
                if val is not None:
                    xs.append(val)
                    ys.append(y + offsets[arm_key])
                    ax.text(
                        val + 1.9,
                        y + offsets[arm_key],
                        f"{val:.1f}%",
                        va="center",
                        ha="left",
                        fontsize=7.3,
                        fontweight="bold" if arm_key in ("dg", "ar_lora") else "normal",
                        color=color,
                    )
            ax.scatter(
                xs,
                ys,
                s=sz,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
                label=arm_label if panel_idx == 0 else None,
            )

        ax.set_xlabel("Accuracy (%)", fontsize=9.5, fontweight="bold", color="#334155")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#cbd5e1")

    ax_k.set_yticks(y_pos)
    ax_k.set_yticklabels([r["label"] for r in rows], fontsize=9.2, fontweight="bold", color="#1e293b")

    handles, labels = ax_k.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.90),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#cbd5e1",
        fontsize=8.4,
    )

    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[Done] Saved verified Table 5 dotplot: {out_png}")
    return out_png


if __name__ == "__main__":
    generate_fig5()
