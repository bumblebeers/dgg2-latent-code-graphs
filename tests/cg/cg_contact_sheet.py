#!/usr/bin/env python3
"""cg_contact_sheet.py -- render a labelled sheet of one source and all of its candidate targets (for eyeballing)."""
import argparse, os, sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cg_ops import ALL_TARGETS, OPEN_TARGETS, TARGET_SPLIT, render  # noqa: E402


def sheet(grids: np.ndarray, labels=None, sublabels=None, cols: int = 5, title: str = "", side: int = 224) -> Image.Image:
    """Labelled sheet of palette grids (any size that divides `side`: 16, 32 ...; 48 uses 4 px cells)."""
    k = len(grids); n = grids[0].shape[0]; cell_px = max(1, side // n); side = cell_px * n
    labels = labels or ALL_TARGETS[:k]
    sublabels = sublabels or [f"[{TARGET_SPLIT.get(x, '')}]" for x in labels]
    pad, lab = 10, 28
    rows = (k + cols - 1) // cols
    im = Image.new("RGB", (cols * (side + pad) + pad, rows * (side + lab + pad) + pad + 18), "white")
    d = ImageDraw.Draw(im)
    d.text((pad, 4), title, fill="black")
    for i, g in enumerate(grids):
        x, y = pad + (i % cols) * (side + pad), 22 + pad + (i // cols) * (side + lab + pad)
        d.text((x, y), labels[i], fill="black")
        d.text((x, y + 12), sublabels[i], fill=(120, 120, 120))
        im.paste(Image.fromarray(render(g, cell_px)), (x, y + lab))
        d.rectangle([x - 1, y + lab - 1, x + side, y + lab + side], outline=(160, 160, 160))
    return im


def open_sheet(z, i: int, title: str = "") -> Image.Image:
    """Source, every valid answer and the did-it-to-all result for the three open-ended operations."""
    grids, labels, subs = [z["src_grid"][i]], ["source"], [""]
    for k, t in enumerate(OPEN_TARGETS):
        c = int(z["open_count"][i, k])
        for j in range(c):
            grids.append(z["open_valid"][i, k, j]); labels.append(t); subs.append(f"valid answer {j + 1} of {c}")
        grids.append(z["open_all"][i, k]); labels.append(t); subs.append("NOT valid: done to all")
    return sheet(np.stack(grids), labels, subs, cols=6, title=title)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pack"); ap.add_argument("--idx", type=int, nargs="+", default=[0]); ap.add_argument("--out", required=True)
    ap.add_argument("--open", action="store_true", help="draw the open-ended answers instead of the 20 fixed targets")
    a = ap.parse_args()
    z = np.load(a.pack); os.makedirs(a.out, exist_ok=True)
    base = os.path.basename(a.pack).split(".")[0]
    for i in a.idx:
        title = f"{os.path.basename(a.pack)}  source {i}  selected={int(z['selected'][i])}"
        p = os.path.join(a.out, f"{'open' if a.open else 'sheet'}_{base}_{i:04d}.png")
        (open_sheet(z, i, title) if a.open else sheet(z["tgt_grid"][i], title=title)).save(p)
        print(p)
