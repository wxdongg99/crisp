#!/usr/bin/env python3
"""Visualize all 18x4 ZJU anomalies with LFP v1 and LFP v3 masks (Qwen-Image-2.1 VAE)."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from highres.textual_inversion_qwen21 import lfp_v1_mask_qwen, lfp_v3_mask_qwen

PATTERNS = ["pattern16", "pattern17", "pattern18", "pattern19"]


def contour(mask_np):
    m = mask_np.astype(bool)
    inner = binary_erosion(m, iterations=2)
    return m & ~inner


def tint_source(src, mask_np, color):
    a = np.array(src).copy()
    edge = contour(mask_np)
    a[edge] = (0.15 * a[edge] + 0.85 * np.array(color)).astype(np.uint8)
    return Image.fromarray(a)


def cell(src, gt, v1, v3, label, mini=150):
    """2x2 mini panel: source, GT, v1 contour, v3 contour."""
    out = Image.new("RGB", (mini * 2, mini * 2 + 18), (255, 255, 255))
    d = ImageDraw.Draw(out)
    out.paste(src.resize((mini, mini), Image.Resampling.LANCZOS), (0, 18))
    out.paste(Image.fromarray(((np.array(gt) > 127) * 255).astype(np.uint8)).convert("RGB").resize((mini, mini), Image.Resampling.NEAREST), (mini, 18))
    out.paste(tint_source(src, np.array(v1) > 127, [40, 90, 255]).resize((mini, mini), Image.Resampling.LANCZOS), (0, mini + 18))
    out.paste(tint_source(src, np.array(v3) > 127, [20, 180, 70]).resize((mini, mini), Image.Resampling.LANCZOS), (mini, mini + 18))
    d.text((2, 2), label, fill=(0, 0, 0))
    d.text((2, mini + 20), "LFP v1", fill=(0, 0, 0))
    d.text((mini + 2, mini + 20), "LFP v3", fill=(0, 0, 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    ap.add_argument("--data", default="/data/solo/wxd/data/zju_leaper_native512/fabric")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out_dir", default="explore/zju_lfp_all")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(dev).eval()

    stats = []
    pattern_sheets = []
    for pat in PATTERNS:
        files = sorted((Path(args.data) / "test" / pat).glob("*.jpg"))
        if not files:
            files = sorted((Path(args.data) / "test" / pat).glob("*.png"))
        cells = []
        for f in files:
            stem = f.stem
            mp = Path(args.data) / "ground_truth" / pat / f"{stem}.png"
            src = Image.open(f).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
            gt = Image.open(mp).convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)
            v1 = lfp_v1_mask_qwen(gt, vae, vae_scale=16)
            a_gt = np.array(gt) > 127; a1 = np.array(v1) > 127
            ratio = int(a1.sum()) / max(1, int(a_gt.sum()))
            v3 = lfp_v3_mask_qwen(gt, vae, vae_scale=16, cache_key=str(mp), ratio=ratio)
            a3 = np.array(v3) > 127
            stats.append({"pattern": pat, "image": f.name, "raw_area": int(a_gt.sum()),
                          "v1_area": int(a1.sum()), "v3_area": int(a3.sum()),
                          "v1_ratio": int(a1.sum()) / max(1, int(a_gt.sum())),
                          "v3_ratio": int(a3.sum()) / max(1, int(a_gt.sum()))})
            cells.append(cell(src, gt, v1, v3, f"{pat}/{stem}"))
            print(pat, f.name, "raw", int(a_gt.sum()), "v1", int(a1.sum()), "v3", int(a3.sum()), flush=True)
        cols = 6; rows = (len(cells) + cols - 1) // cols
        cw, ch = 306, 342
        sheet = Image.new("RGB", (cols * cw, rows * ch + 36), (255, 255, 255))
        d = ImageDraw.Draw(sheet); d.text((6, 8), f"{pat}: 18 anomalies | source / GT / LFP v1 (blue) / LFP v3 (green)", fill=(0, 0, 0))
        for i, c in enumerate(cells):
            x = (i % cols) * cw; y = (i // cols) * ch + 36
            sheet.paste(c, (x + 2, y + 2))
        p = out / f"{pat}_lfp_v1_v3.png"; sheet.save(p); pattern_sheets.append(p)
    # combined
    ims = [Image.open(p) for p in pattern_sheets]
    W = max(i.width for i in ims); H = sum(i.height for i in ims)
    comb = Image.new("RGB", (W, H), (255, 255, 255)); yy = 0
    for im in ims:
        comb.paste(im, (0, yy)); yy += im.height
    comb.save(out / "zju_18x4_lfp_v1_v3_all.jpg", quality=92)
    with open(out / "lfp_area_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pattern", "image", "raw_area", "v1_area", "v3_area", "v1_ratio", "v3_ratio"])
        w.writeheader(); w.writerows(stats)
    print("saved", out / "zju_18x4_lfp_v1_v3_all.jpg", flush=True)


if __name__ == "__main__":
    main()
