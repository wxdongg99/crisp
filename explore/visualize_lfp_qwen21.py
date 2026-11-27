#!/usr/bin/env python3
"""Visualize LFP v1 -> LFP v3 construction with the Qwen-Image-2.1 VAE.

Run on the B300 host:
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$PWD /data/solo/wxd/envs/qwen21_py310/bin/python \
    explore/visualize_lfp_qwen21.py --model /data/solo/wxd/models/Qwen-Image-2.1 \
    --image /data/solo/wxd/data/zju_eval512/fabric/train/good/075428.jpg \
    --mask /data/solo/wxd/data/generated_mask_fabric/pattern16/136.jpg \
    --size 2048 --out_dir explore/lfp_qwen21_vis
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from highres.lfp_mask import _otsu, _nearest_dilate_to_area


def qwen_lfp_v1(mask: torch.Tensor, vae, target_h: int, target_w: int) -> Image.Image:
    """Same logic as Qwen21 _qwen_lfp_latent; returns pixel-upsampled v1 mask."""
    m = (mask[None, None] >= 0.5).float()
    bil = F.interpolate(m, size=(target_h, target_w), mode="bilinear", align_corners=False)
    rgb = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
    alpha = torch.ones_like(rgb[:, :1])
    probe = torch.cat([rgb, alpha], dim=1).to(next(vae.parameters()).device, next(vae.parameters()).dtype)
    with torch.no_grad():
        z = vae.encode(probe.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    resp = z[0].detach().float().cpu().numpy().__abs__().mean(axis=0)
    footprint = F.max_pool2d(m, kernel_size=mask.shape[0] // target_h, stride=mask.shape[0] // target_h)[0, 0].cpu().numpy() > 0.5
    lo, hi = float(resp.min()), float(resp.max())
    if hi - lo <= 1e-6:
        support = footprint.copy()
    else:
        thr = _otsu((resp - lo) / (hi - lo))
        support = resp > lo + thr * (hi - lo)
        if (support & footprint).sum() < 0.5 * support.sum():
            support = ~support
        support = support | footprint
    final = support.astype(np.float64)
    final[0, :] = 0; final[-1, :] = 0; final[:, 0] = 0; final[:, -1] = 0
    final = np.maximum(final, bil[0, 0].cpu().numpy())
    while True:
        p = np.pad(final, 1, constant_values=0)
        neigh = ((p[:-2, 1:-1] > .5).astype(int) + (p[2:, 1:-1] > .5).astype(int)
                 + (p[1:-1, :-2] > .5).astype(int) + (p[1:-1, 2:] > .5).astype(int))
        fill = (final < .5) & (neigh >= 3)
        if not fill.any():
            break
        final[fill] = 1.0
    pix = F.interpolate(torch.from_numpy((final >= .5).astype(np.float32))[None, None],
                        size=(mask.shape[0], mask.shape[1]), mode="nearest")[0, 0].numpy()
    return Image.fromarray((pix * 255).astype(np.uint8), mode="L"), resp, bil[0, 0].cpu().numpy()


def panel(label, pil, size=420):
    im = pil.convert("RGB").resize((size, size), Image.Resampling.NEAREST if pil.mode == "L" else Image.Resampling.LANCZOS)
    return label, im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    ap.add_argument("--image", default="/data/solo/wxd/data/zju_eval512/fabric/train/good/075428.jpg")
    ap.add_argument("--mask", default="/data/solo/wxd/data/generated_mask_fabric/pattern16/136.jpg")
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--out_dir", default=str(ROOT / "explore" / "lfp_qwen21_vis"))
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKLQwenImage21
    dtype = torch.float32
    vae = AutoencoderKLQwenImage21.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(dev).eval()

    src = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
    raw = Image.open(args.mask).convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)
    m = torch.from_numpy(np.asarray(raw)).float()[None, None] / 255.0
    m = (m >= 0.5).float().to(dev)
    scale = 16
    th = tw = args.size // scale
    v1, resp, bil = qwen_lfp_v1(m[0, 0], vae, th, tw)
    raw_np = np.asarray(raw) > 127
    v1_np = np.asarray(v1) > 127
    raw_area = int(raw_np.sum()); v1_area = int(v1_np.sum())
    ratio = v1_area / max(raw_area, 1)
    target = int(round(raw_area * ratio))
    base = F.interpolate(F.max_pool2d(m, kernel_size=scale, stride=scale),
                         size=(args.size, args.size), mode="nearest")[0, 0].cpu().numpy() > 0.5
    dist = distance_transform_edt(~base)
    nearest = np.argpartition(dist.ravel(), min(target, dist.size) - 1)[:min(target, dist.size)]
    v3 = np.zeros(dist.shape, dtype=bool); v3.ravel()[nearest] = True
    v3_img = Image.fromarray((v3.astype(np.uint8) * 255), mode="L")

    def heat(a):
        a = a.astype(np.float32); a = (a - a.min()) / max(a.max() - a.min(), 1e-8)
        u = (a * 255).astype(np.uint8)
        return Image.fromarray(u).convert("RGB").resize((args.size, args.size), Image.Resampling.BILINEAR)

    ov = src.copy(); ov = np.asarray(ov).copy()
    ov[raw_np] = (0.55 * ov[raw_np] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    ov[v3] = (0.55 * ov[v3] + 0.45 * np.array([0, 80, 255])).astype(np.uint8)
    ov = Image.fromarray(ov)

    panels = [
        panel("1 source 2048", src),
        panel("2 raw mask", raw),
        panel("3 VAE latent response", heat(resp)),
        panel("4 LFP v1 support", v1),
        panel("5 v3 maxpool base", Image.fromarray((base * 255).astype(np.uint8), mode="L")),
        panel("6 distance to base", heat(dist)),
        panel("7 LFP v3 final", v3_img),
        panel("8 overlay raw/v3", ov),
    ]
    W, H = 420, 460
    cols = 4; rows = 2
    sheet = Image.new("RGB", (W * cols, H * rows), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(panels):
        x = (i % cols) * W; y = (i // cols) * H
        sheet.paste(im, (x + 6, y + 28)); d.text((x + 8, y + 8), lab, fill=(0, 0, 0))
    sheet.save(out / "lfp_qwen21_pipeline.png")
    # individual artifacts
    for name, im in [("source", src), ("mask_raw", raw), ("lfp_v1", v1), ("lfp_v3", v3_img)]:
        im.save(out / f"{name}.png")
    (out / "stats.txt").write_text(
        f"size={args.size}\nraw_area_px={raw_area}\nv1_area_px={v1_area}\n"
        f"v3_target_area_px={target}\nratio={ratio:.4f}\n"
        f"raw_fraction={raw_area/(args.size*args.size):.4f}\n"
        f"v3_fraction={v3.sum()/(args.size*args.size):.4f}\n"
        f"v3_vs_raw={v3.sum()/max(raw_area,1):.2f}\n")
    print(open(out / "stats.txt").read())
    print("saved", out / "lfp_qwen21_pipeline.png")


if __name__ == "__main__":
    main()
