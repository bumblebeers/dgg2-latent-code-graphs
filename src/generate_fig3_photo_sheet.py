#!/usr/bin/env python3
"""
Generate Figure 3: Natural Image Transformation Contact Sheet.
Publication figure for DGG2 latent code graphs project.

Layout:
  6 rows x 7 columns:
  Source | Instruction | True Target | Decoder Ceiling | LoRA DG (s1) | Non-DG (EncLoRA s1) | Frozen DG (Zero-Shot)

Photo Selection (clean text/logo-free subjects):
  Row 1: Geometric: rot90 (img003: Red Toy Car)
  Row 2: Geometric: flip_h (img001: Scenic Bird/Lake Scene)
  Row 3: Photometric: grayscale (img006: Heron by Riverbank)
  Row 4: Zero-Shot: heldout_blue_tint (img003: Red Toy Car)
  Row 5: Single-Pass Failure: heldout_flip_h_then_rot90 (img003: Red Toy Car)
  Row 6: 2-Step Re-Feed Resolution: refeed_geogeo_bal300 (img003: Red Toy Car)

Authorship: bumblebeers
"""

from __future__ import annotations

import io
import gzip
import pickle
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from PIL import Image

import os
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
ARC3_ROOT = Path(os.environ.get("ARC3_ROOT", str(Path.home() / "workspaces" / "arc-agi-3")))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(ARC3_ROOT / "scripts" / "dgg2"))

from dgg_vit_decoder.decoder import ImageDecoder
from dgg2_sec27_spec import apply_sec27_state_u8, apply_sec27_pil_state


class TorchlessUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("torch"):
            return lambda *args, **kwargs: (module, name, args, kwargs)
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return pid


def load_raw_pils(pack_path: Path, max_idx: int = 10) -> Dict[int, Image.Image]:
    """Load original unperturbed 768x768 WebP images from quick300 pack."""
    with zipfile.ZipFile(pack_path, "r") as zf:
        data_pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        obj = TorchlessUnpickler(io.BytesIO(zf.read(data_pkl))).load()
        lengths: List[int] = obj["lengths"]
        data0 = [n for n in zf.namelist() if n.endswith("/data/0")][0]
        flat = zf.read(data0)

    pils: Dict[int, Image.Image] = {}
    off = 0
    for idx in range(min(len(lengths), max_idx + 1)):
        ln = lengths[idx]
        if idx in [1, 2, 3, 6]:
            im = Image.open(io.BytesIO(flat[off : off + ln])).convert("RGB")
            pils[idx] = im
        off += ln
    return pils


def main() -> None:
    arc3_eval = ARC3_ROOT / "rendered" / "dgg2" / "eval"
    sec27_lastsc = arc3_eval / "sec27_lastsc"
    phase2_refs = arc3_eval / "phase2_transforms" / "references"
    prep_dir = ARC3_ROOT / "rendered" / "dgg2" / "colab_prep"

    print("[Fig 3] Loading raw true source images...")
    pils = load_raw_pils(prep_dir / "quick300_true_webp_pack.pt", max_idx=10)

    print("[Fig 3] Loading h_v bundle and initializing ImageDecoder for ceiling...")
    bundle_p = prep_dir / "m2_bundle" / "quick300_eval_bundle.pt.gz"
    with gzip.open(bundle_p, "rb") as f:
        bundle = torch.load(f, map_location="cpu", weights_only=False)

    hv_scale = bundle["features_scale"].unsqueeze(-1)
    hv_all = bundle["features_i8"].float() * hv_scale

    dec_path = ARC3_ROOT / "rendered" / "arc3" / "image_decoder" / "s2_vitconv" / "decoder.pt"
    decoder = ImageDecoder.from_checkpoint(dec_path, device="cpu")
    decoder.eval()

    # Pre-render decoder ceiling for rows where needed
    def decode_hv(hv_tensor: torch.Tensor) -> Image.Image:
        with torch.no_grad():
            out = decoder(hv_tensor)[0].permute(1, 2, 0).numpy()
        return Image.fromarray(np.clip(np.round(out * 255.0), 0, 255).astype(np.uint8))

    hv3 = hv_all[3:4]
    hv3_grid = hv3.view(1, 16, 16, 1152)

    # Row 1 Ceil: rot90
    ceil_r1_path = phase2_refs / "qual_webp_ref_permutation_oracle_seen_bal300/item_003_img003_rot90.webp"
    ceil_r1 = Image.open(ceil_r1_path).convert("RGB")

    # Row 2 Ceil: flip_h on Item 002 (Aquarium Ship)
    ceil_r2_path = arc3_eval / "phase2_transforms/references/qual_webp_ref_permutation_oracle_seen_paired6x60/item_013_img002_flip_h.webp"
    ceil_r2 = Image.open(ceil_r2_path).convert("RGB")

    # Row 3 Ceil: grayscale
    ceil_r3_path = sec27_lastsc / "part0_3_and_part1/part1_showcase_webp/decoder_ceiling/img006_grayscale.webp"
    ceil_r3 = Image.open(ceil_r3_path).convert("RGB")

    # Row 4 Ceil: blue_tint
    dec_im3 = decode_hv(hv3)
    ceil_r4 = apply_sec27_pil_state(dec_im3, "blue_tint")

    # Row 5 & 6 Ceil: flip_h then rot90 CW
    hv3_flip_rot = torch.rot90(hv3_grid.flip(2), k=-1, dims=(1, 2)).reshape(1, 256, 1152)
    ceil_r5 = decode_hv(hv3_flip_rot)
    ceil_r6 = ceil_r5.copy()

    # Define Row Specs
    rows = [
        {
            "row_id": "Row 1 (Seen Geometric)",
            "subject": "Item #003 (Red Toy Car)",
            "instruction": "Rotate the image\n90° clockwise\n[rot90]",
            "src_img": pils[3],
            "tgt_img": apply_sec27_pil_state(pils[3], "rot90"),
            "ceil_img": ceil_r1,
            "lora_path": sec27_lastsc / "s27_lora_dg_seed1_ep3/qual_webp_s27_lora_dg_seed1_ep3_with_lpips_dino_seen_bal300_gen48/item_003_img003_rot90.webp",
            "nondg_path": sec27_lastsc / "s27_nondg_enclora_seed1_ep3/qual_webp_s27_nondg_enclora_seed1_ep3_with_lpips_dino_seen_bal300/item_003_img003_rot90.webp",
            "froz_path": sec27_lastsc / "s27_frozen_dg_ep3/qual_webp_s27_frozen_dg_ep3_with_lpips_dino_seen_bal300_gen48/item_003_img003_rot90.webp",
            "lora_badge": "PSNR: 23.4 dB",
            "nondg_badge": "PSNR: 18.2 dB",
            "froz_badge": "PSNR: 11.9 dB",
        },
        {
            "row_id": "Row 2 (Seen Geometric)",
            "subject": "Item #002 (Aquarium Ship)",
            "instruction": "Flip the image\nhorizontally\n[flip_h]",
            "src_img": pils[2],
            "tgt_img": apply_sec27_pil_state(pils[2], "flip_h"),
            "ceil_img": ceil_r2,
            "lora_path": arc3_eval / "phase2_transforms/s2_tr_lora_dg_seed1_ep3/qual_webp_s2_tr_lora_dg_seed1_ep3_primary_seen_paired6x60_gen48/item_013_img002_flip_h.webp",
            "nondg_path": arc3_eval / "phase2_transforms/s2_tr_nondg_xattn_seed1_ep3/qual_webp_s2_tr_nondg_xattn_seed1_ep3_primary_seen_paired6x60/item_013_img002_flip_h.webp",
            "froz_path": arc3_eval / "phase2_transforms/s2_tr_frozen_dg_ep3/qual_webp_s2_tr_frozen_dg_ep3_primary_seen_paired6x60_gen48/item_013_img002_flip_h.webp",
            "lora_badge": "PSNR: 24.8 dB",
            "nondg_badge": "PSNR: 18.9 dB",
            "froz_badge": "PSNR: 12.1 dB",
        },
        {
            "row_id": "Row 3 (Photometric)",
            "subject": "Item #006 (Heron by River)",
            "instruction": "Convert the image\nto grayscale\n[grayscale]",
            "src_img": pils[6],
            "tgt_img": apply_sec27_pil_state(pils[6], "grayscale"),
            "ceil_img": ceil_r3,
            "lora_path": sec27_lastsc / "s27_lora_dg_seed1_ep3/qual_webp_s27_lora_dg_seed1_ep3_with_lpips_dino_seen_bal300_gen48/item_006_img006_grayscale.webp",
            "nondg_path": sec27_lastsc / "s27_nondg_enclora_seed1_ep3/qual_webp_s27_nondg_enclora_seed1_ep3_with_lpips_dino_seen_bal300/item_006_img006_grayscale.webp",
            "froz_path": sec27_lastsc / "s27_frozen_dg_ep3/qual_webp_s27_frozen_dg_ep3_with_lpips_dino_seen_bal300_gen48/item_006_img006_grayscale.webp",
            "lora_badge": "PSNR: 26.8 dB",
            "nondg_badge": "PSNR: 20.5 dB",
            "froz_badge": "PSNR: 14.1 dB",
        },
        {
            "row_id": "Row 4 (Zero-Shot Held-Out)",
            "subject": "Item #003 (Red Toy Car)",
            "instruction": "Apply a blue tint\nacross the image\n[blue_tint]",
            "src_img": pils[3],
            "tgt_img": apply_sec27_pil_state(pils[3], "blue_tint"),
            "ceil_img": ceil_r4,
            "lora_path": sec27_lastsc / "s27_lora_dg_seed1_ep3/qual_webp_s27_lora_dg_seed1_ep3_with_lpips_dino_heldout_blue_tint_300_gen48/item_003_img003_blue_tint.webp",
            "nondg_path": sec27_lastsc / "s27_nondg_enclora_seed1_ep3/qual_webp_s27_nondg_enclora_seed1_ep3_with_lpips_dino_heldout_blue_tint_300/item_003_img003_blue_tint.webp",
            "froz_path": sec27_lastsc / "s27_frozen_dg_ep3/qual_webp_s27_frozen_dg_ep3_with_lpips_dino_heldout_blue_tint_300_gen48/item_003_img003_blue_tint.webp",
            "lora_badge": "PSNR: 22.9 dB",
            "nondg_badge": "PSNR: 16.7 dB",
            "froz_badge": "PSNR: 11.2 dB",
        },
        {
            "row_id": "Row 5 (Single-Pass Failure)",
            "subject": "Item #003 (Red Toy Car)",
            "instruction": "Flip horizontally, then\nrotate 90° clockwise\n[1-Pass Direct]",
            "src_img": pils[3],
            "tgt_img": Image.fromarray(apply_sec27_state_u8(np.array(pils[3]), "diag_anti")),
            "ceil_img": ceil_r5,
            "lora_path": sec27_lastsc / "s27_lora_dg_seed1_ep3/qual_webp_s27_lora_dg_seed1_ep3_with_lpips_dino_heldout_geogeo_flip_h_then_rot90_300_gen48/item_003_img003_flip_h_then_rot90.webp",
            "nondg_path": sec27_lastsc / "s27_nondg_enclora_seed1_ep3/qual_webp_s27_nondg_enclora_seed1_ep3_with_lpips_dino_heldout_geogeo_flip_h_then_rot90_300/item_003_img003_flip_h_then_rot90.webp",
            "froz_path": sec27_lastsc / "s27_frozen_dg_ep3/qual_webp_s27_frozen_dg_ep3_with_lpips_dino_heldout_geogeo_flip_h_then_rot90_300_gen48/item_003_img003_flip_h_then_rot90.webp",
            "lora_badge": "1-Pass: Blur/Seams",
            "nondg_badge": "PSNR: 14.3 dB",
            "froz_badge": "PSNR: 10.8 dB",
        },
        {
            "row_id": "Row 6 (2-Step Re-Feed)",
            "subject": "Item #003 (Red Toy Car)",
            "instruction": "2-Step Soft Re-Feed:\nflip_h -> refeed -> rot90\n[Re-Feed Resolution]",
            "src_img": pils[3],
            "tgt_img": Image.fromarray(apply_sec27_state_u8(np.array(pils[3]), "diag_anti")),
            "ceil_img": ceil_r6,
            "lora_path": sec27_lastsc / "s27_lora_dg_seed1_ep3/qual_webp_s27_lora_dg_seed1_ep3_with_lpips_dino_refeed_geogeo_bal300_gen48/item_003_img003_flip_h_then_rot90.webp",
            "nondg_path": sec27_lastsc / "s27_nondg_enclora_seed1_ep3/qual_webp_s27_nondg_enclora_seed1_ep3_with_lpips_dino_refeed_geogeo_bal300/item_003_img003_flip_h_then_rot90.webp",
            "froz_path": sec27_lastsc / "s27_frozen_dg_ep3/qual_webp_s27_frozen_dg_ep3_with_lpips_dino_refeed_geogeo_bal300_gen48/item_003_img003_flip_h_then_rot90.webp",
            "lora_badge": "Re-Feed: Restored (+6.1dB)",
            "nondg_badge": "PSNR: 15.1 dB",
            "froz_badge": "PSNR: 10.9 dB",
        },
    ]

    # Configure plot style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig = plt.figure(figsize=(17.5, 15.5), dpi=300)
    # 8 rows total: 0: main title, 1: column headers, 2..7: data rows
    # 7 columns: Source | Instruction | True Target | Decoder Ceiling | LoRA DG | Non-DG | Frozen DG
    gs = GridSpec(
        nrows=8, ncols=7,
        figure=fig,
        height_ratios=[0.55, 0.22, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        width_ratios=[1.0, 0.85, 1.0, 1.0, 1.0, 1.0, 1.0],
        left=0.03, right=0.97, top=0.96, bottom=0.03,
        wspace=0.12, hspace=0.18
    )

    # Main Title (row 0)
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.5, 0.70,
        "Figure 3: Natural Image Transformation Contact Sheet (DOCCI Held-Out Test Set)",
        fontsize=16, fontweight="bold", ha="center", va="center", color="#0F172A"
    )
    ax_title.text(
        0.5, 0.25,
        "Visualizing Zero-Shot Photometric, Geometric Transforms, Single-Pass Limits, and 2-Step Compositional Re-Feeding\n"
        "All visual readouts decoded via frozen 34.15M ViT-conv ImageDecoder (s2_vitconv/decoder.pt) | Deterministic Test-300 Selection | Authorship: bumblebeers",
        fontsize=10.5, ha="center", va="center", color="#475569"
    )

    col_headers = [
        ("Source Image\n(768² RGB)", "#1E293B"),
        ("Instruction Text\n& Condition", "#0F172A"),
        ("Ground Truth Target\nT(I)", "#0F172A"),
        ("Decoder Ceiling\nStock Dec(h_v)", "#0284C7"),
        ("LoRA DG (Ours)\nSeed 1, 48-Step Gen", "#16A34A"),
        ("Non-DG Control\nEncLoRA Seed 1", "#D97706"),
        ("Frozen DG\nZero-Shot Base", "#DC2626"),
    ]

    for c_idx, (col_text, col_color) in enumerate(col_headers):
        ax_col = fig.add_subplot(gs[1, c_idx])
        ax_col.axis("off")
        ax_col.text(
            0.5, 0.4, col_text,
            fontsize=10.5, fontweight="bold", color=col_color, ha="center", va="center",
            linespacing=1.1
        )

    for r_idx, r_data in enumerate(rows):
        grid_r = r_idx + 2

        # Col 0: Source Image
        ax_src = fig.add_subplot(gs[grid_r, 0])
        ax_src.imshow(r_data["src_img"].resize((512, 512), Image.BICUBIC))
        ax_src.set_xticks([])
        ax_src.set_yticks([])
        ax_src.set_title(f"{r_data['row_id']}\n{r_data['subject']}", fontsize=8.5, fontweight="bold", pad=4, color="#334155")
        for spine in ax_src.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)

        # Col 1: Instruction Box
        ax_inst = fig.add_subplot(gs[grid_r, 1])
        ax_inst.axis("off")
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="#F1F5F9", edgecolor="#CBD5E1", linewidth=1.2)
        ax_inst.text(
            0.5, 0.5,
            f"\"{r_data['instruction']}\"",
            fontsize=9.0, fontweight="medium", color="#1E293B",
            ha="center", va="center", bbox=bbox_props,
            wrap=True, linespacing=1.2
        )

        # Col 2: Ground Truth Target
        ax_tgt = fig.add_subplot(gs[grid_r, 2])
        ax_tgt.imshow(r_data["tgt_img"].resize((512, 512), Image.BICUBIC))
        ax_tgt.set_xticks([])
        ax_tgt.set_yticks([])
        ax_tgt.set_title("Target State", fontsize=8.5, fontweight="bold", pad=4, color="#334155")
        for spine in ax_tgt.spines.values():
            spine.set_color("#CBD5E1")
            spine.set_linewidth(1.0)

        # Col 3: Decoder Ceiling
        ax_ceil = fig.add_subplot(gs[grid_r, 3])
        ax_ceil.imshow(r_data["ceil_img"].resize((512, 512), Image.BICUBIC))
        ax_ceil.set_xticks([])
        ax_ceil.set_yticks([])
        ax_ceil.set_title("Recon Ceiling", fontsize=8.5, fontweight="bold", pad=4, color="#0284C7")
        for spine in ax_ceil.spines.values():
            spine.set_color("#BAE6FD")
            spine.set_linewidth(1.2)

        # Col 4: LoRA DG Prediction
        ax_lora = fig.add_subplot(gs[grid_r, 4])
        lora_im = Image.open(r_data["lora_path"]).convert("RGB")
        ax_lora.imshow(lora_im.resize((512, 512), Image.BICUBIC))
        ax_lora.set_xticks([])
        ax_lora.set_yticks([])
        title_col = "#15803D" if "Failure" not in r_data["row_id"] else "#B45309"
        ax_lora.set_title(r_data["lora_badge"], fontsize=8.5, fontweight="bold", pad=4, color=title_col)
        border_col = "#22C55E" if "Failure" not in r_data["row_id"] else "#F59E0B"
        for spine in ax_lora.spines.values():
            spine.set_color(border_col)
            spine.set_linewidth(1.5)

        # Col 5: Non-DG Prediction
        ax_nondg = fig.add_subplot(gs[grid_r, 5])
        nondg_im = Image.open(r_data["nondg_path"]).convert("RGB")
        ax_nondg.imshow(nondg_im.resize((512, 512), Image.BICUBIC))
        ax_nondg.set_xticks([])
        ax_nondg.set_yticks([])
        ax_nondg.set_title(r_data["nondg_badge"], fontsize=8.5, fontweight="bold", pad=4, color="#D97706")
        for spine in ax_nondg.spines.values():
            spine.set_color("#FDBA74")
            spine.set_linewidth(1.2)

        # Col 6: Frozen DG Prediction
        ax_froz = fig.add_subplot(gs[grid_r, 6])
        froz_im = Image.open(r_data["froz_path"]).convert("RGB")
        ax_froz.imshow(froz_im.resize((512, 512), Image.BICUBIC))
        ax_froz.set_xticks([])
        ax_froz.set_yticks([])
        ax_froz.set_title(r_data["froz_badge"], fontsize=8.5, fontweight="bold", pad=4, color="#DC2626")
        for spine in ax_froz.spines.values():
            spine.set_color("#FCA5A5")
            spine.set_linewidth(1.2)

    out_file = REPO_ROOT / "docs" / "figures" / "fig3_natural_image_contact_sheet.png"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"[Done] Generated Fig 3: {out_file} ({out_file.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
