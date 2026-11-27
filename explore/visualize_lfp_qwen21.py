#!/usr/bin/env python3
"""Detailed LFP v1/v3 construction visualization on Qwen-Image-2.1.

Shows:
  * raw mask
  * ternary RGB+alpha VAE probe
  * all 64 VAE latent channels (heatmaps)
  * per-channel Otsu diagnostic
  * aggregate response, Otsu threshold, polarity correction, footprint union
  * trim / bilinear merge / hole fill -> LFP v1
  * maxpool base, distance map, nearest-point dilation -> LFP v3
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
from highres.lfp_mask import _otsu


def heat_rgb(a: np.ndarray, size: int) -> Image.Image:
    a = a.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    n = (a - lo) / max(hi - lo, 1e-8)
    # blue -> white -> red (readable on white sheet)
    r = np.clip(n * 2, 0, 1); b = np.clip((1 - n) * 2, 0, 1); g = np.clip(1 - np.abs(n * 2 - 1), 0, 1)
    out = (np.stack([r, g, b], -1) * 255).astype(np.uint8)
    return Image.fromarray(out).resize((size, size), Image.Resampling.BILINEAR)


def bin_img(a: np.ndarray, size: int) -> Image.Image:
    return Image.fromarray((a.astype(np.uint8) * 255), mode="L").resize((size, size), Image.Resampling.NEAREST)


def collect_v1(mask: torch.Tensor, vae):
    """Replicate Qwen21 _qwen_lfp_latent and expose every intermediate."""
    hw = mask.shape[-1]
    scale = 16
    th = tw = hw // scale
    m = (mask[None, None] >= 0.5).float()
    bil = F.interpolate(m, size=(th, tw), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    rgb = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
    alpha = torch.ones_like(rgb[:, :1])
    probe = torch.cat([rgb, alpha], 1).to(next(vae.parameters()).device, next(vae.parameters()).dtype)
    with torch.no_grad():
        z = vae.encode(probe.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    z_np = z[0].detach().float().cpu().numpy()
    if z_np.ndim == 4:      # (C,1,h,w)
        z_np = z_np[:, 0]
    elif z_np.ndim == 2:    # (h,w)
        z_np = z_np[None]
    resp = np.abs(z_np).mean(axis=0)
    footprint = F.max_pool2d(m, kernel_size=hw // th, stride=hw // th)[0, 0].cpu().numpy() > 0.5
    lo, hi = float(resp.min()), float(resp.max())
    if hi - lo <= 1e-6:
        support = footprint.copy(); thr = 0.0
    else:
        thr = _otsu((resp - lo) / (hi - lo))
        support = resp > lo + thr * (hi - lo)
    support_before = support.copy()
    if (support & footprint).sum() < 0.5 * support.sum():
        support = ~support
    support_polarity = support.copy()
    support_union = support | footprint
    trim = support_union.copy(); trim[0, :] = 0; trim[-1, :] = 0; trim[:, 0] = 0; trim[:, -1] = 0
    merge = np.maximum(trim.astype(np.float64), bil)
    final = merge.copy()
    while True:
        p = np.pad(final, 1, constant_values=0)
        neigh = ((p[:-2, 1:-1] > .5).astype(int) + (p[2:, 1:-1] > .5).astype(int)
                 + (p[1:-1, :-2] > .5).astype(int) + (p[1:-1, 2:] > .5).astype(int))
        fill = (final < .5) & (neigh >= 3)
        if not fill.any():
            break
        final[fill] = 1.0
    return dict(z=z_np, resp=resp, bil=bil, footprint=footprint, thr=thr,
                support_before=support_before, support_polarity=support_polarity,
                support_union=support_union, trim=trim, merge=merge, final=final)


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
    vae = AutoencoderKLQwenImage21.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae.to(dev).eval()

    src = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
    raw = Image.open(args.mask).convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)
    m = (torch.from_numpy(np.array(raw, dtype=np.float32)).to(dev)[None, None] / 255.0 >= 0.5).float()
    d = collect_v1(m[0, 0], vae)
    z = d["z"]; resp = d["resp"]; final = d["final"]
    C = z.shape[0]
    raw_np = np.array(raw) > 127
    v1_np = final >= 0.5
    # per-channel Otsu diagnostic
    ch_otsu = []
    ch_thr = []
    for c in range(C):
        a = z[c]; lo, hi = float(a.min()), float(a.max())
        if hi - lo <= 1e-6:
            ch_otsu.append(np.zeros_like(a, bool)); ch_thr.append(0.0)
        else:
            t = _otsu((a - lo) / (hi - lo)); ch_otsu.append(a > lo + t * (hi - lo)); ch_thr.append(t)
    # diagnostic: literal 64-channel Otsu union (the normal 4-channel LFP rule)
    white_minority = float(m.mean().item()) < 0.5
    ch_pol = []
    for c in range(C):
        a = z[c]; lo, hi = float(a.min()), float(a.max())
        if hi - lo <= 1e-6:
            above = np.zeros_like(a, bool)
        else:
            t = _otsu((a - lo) / (hi - lo)); above = a > lo + t * (hi - lo)
        above_minority = bool(above.sum() < above.size / 2)
        if white_minority:
            pol = above if above_minority else ~above
        else:
            pol = above if not above_minority else ~above
        ch_pol.append(pol)
    union_all = np.any(np.stack(ch_pol, 0), axis=0)
    # user rule: each channel's smaller Otsu side is foreground; then union
    ch_small = []
    for c in range(C):
        a = z[c]; lo, hi = float(a.min()), float(a.max())
        if hi - lo <= 1e-6:
            above = np.zeros_like(a, bool)
        else:
            t = _otsu((a - lo) / (hi - lo)); above = a > lo + t * (hi - lo)
        below = ~above
        ch_small.append(above if above.sum() <= below.sum() else below)
    union_small = np.any(np.stack(ch_small, 0), axis=0)
    # channel grids
    cell = 128; cols = 8; rows = int(np.ceil(C / cols)); pad = 6
    grid = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 255, 255)); dd = ImageDraw.Draw(grid)
    ogrid = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 255, 255)); od = ImageDraw.Draw(ogrid)
    for c in range(C):
        x = (c % cols) * cell; y = (c // cols) * (cell + 18)
        grid.paste(heat_rgb(z[c], cell - 4).convert("RGB"), (x + 2, y + 16))
        dd.text((x + 4, y + 2), f"ch{c:02d}", fill=(0, 0, 0))
        ogrid.paste(bin_img(ch_otsu[c], cell - 4).convert("RGB"), (x + 2, y + 16))
        od.text((x + 4, y + 2), f"ch{c:02d} otsu", fill=(0, 0, 0))
    grid.save(out / "vae_all_channels.png")
    ogrid.save(out / "vae_all_channels_otsu.png")
    # smaller-side foreground grid
    sgrid = Image.new("RGB", (cols * cell, rows * (cell + 18)), (255, 255, 255)); sgd = ImageDraw.Draw(sgrid)
    for c in range(C):
        x = (c % cols) * cell; y = (c // cols) * (cell + 18)
        sgrid.paste(bin_img(ch_small[c], cell - 4).convert("RGB"), (x + 2, y + 16))
        sgd.text((x + 4, y + 2), f"ch{c:02d} small", fill=(0, 0, 0))
    sgrid.save(out / "vae_all_channels_smaller_side.png")
    Image.fromarray((union_all.astype(np.uint8) * 255), mode="L").save(out / "v1_union_all64_raw.png")
    Image.fromarray((union_small.astype(np.uint8) * 255), mode="L").save(out / "v1_union_smaller_side_raw.png")
    # main step sheet
    def h(a): return heat_rgb(a, 400)
    def b(a): return bin_img(a.astype(bool), 400)
    steps = [
        ("1 source 2048", src.convert("RGB")),
        ("2 raw mask", raw.convert("RGB")),
        ("3 latent resp mean|z|", h(resp)),
        ("4 Otsu resp > thr", b(d["support_before"])),
        ("5 polarity fixed", b(d["support_polarity"])),
        ("6 union maxpool footprint", b(d["support_union"])),
        ("7 trim border + bilinear", b(d["merge"])),
        ("8 + hole fill = LFP v1", b(final)),
        ("9 v1 pixel overlay", None),
        ("10 Otsu histogram", None),
    ]
    W = 420; H = 455
    sheet = Image.new("RGB", (W * 4, H * 3), (255, 255, 255)); sd = ImageDraw.Draw(sheet)
    v1_pix = np.array(Image.fromarray((final * 255).astype(np.uint8), mode="L").resize(
        (args.size, args.size), Image.Resampling.NEAREST)) > 127
    src_np = np.array(src).astype(np.uint8)
    ov = np.where(v1_pix[..., None],
                  (0.45 * src_np + 0.55 * np.array([255, 0, 0])).astype(np.uint8), src_np).astype(np.uint8)
    steps[8] = ("9 v1 pixel overlay", Image.fromarray(ov))
    for i, (lab, im) in enumerate(steps[:-1]):
        x = (i % 4) * W; y = (i // 4) * H
        if im is not None:
            sheet.paste(im.resize((W - 12, W - 12), Image.Resampling.NEAREST).convert("RGB"), (x + 6, y + 26))
        sd.text((x + 8, y + 6), lab, fill=(0, 0, 0))
    # histogram panel with Otsu threshold
    import math
    hist, edges = np.histogram(resp.ravel(), bins=128)
    hp = Image.new("RGB", (W - 12, W - 12), (255, 255, 255)); hd = ImageDraw.Draw(hp)
    hm = hist.max()
    for i, v in enumerate(hist):
        x0 = int(i * (W - 16) / 128); x1 = int((i + 1) * (W - 16) / 128)
        hh = int(v / max(hm, 1) * (W - 40))
        hd.rectangle([x0 + 6, (W - 20) - hh, max(x0 + 6, x1 + 4), W - 20], fill=(70, 70, 70))
    tx = int(6 + (d['thr']) * (W - 16))
    hd.line([tx, 10, tx, W - 20], fill=(220, 0, 0), width=3)
    hd.text((tx + 4, 12), f"Otsu={d['thr']:.3f}", fill=(220, 0, 0))
    sheet.paste(hp.resize((W - 12, W - 12)), ((9 % 4) * W + 6, (9 // 4) * H + 26))
    sd.text(((9 % 4) * W + 8, (9 // 4) * H + 6), "10 Otsu threshold on resp", fill=(0, 0, 0))
    sheet.save(out / "lfp_qwen21_v1_steps.png")
    # stats
    v1_latent_area = int(final.sum())
    v1_pixel_area = int(v1_pix.sum())
    stats = {
        "size": args.size, "latent_shape": list(z.shape), "channels": C,
        "raw_area_px": int(raw_np.sum()), "raw_fraction": float(raw_np.mean()),
        "v1_area_latent": v1_latent_area, "v1_area_px": v1_pixel_area,
        "v1_fraction_px": v1_pixel_area / (args.size * args.size),
        "v1_vs_raw_px": v1_pixel_area / max(1, int(raw_np.sum())),
        "otsu_threshold_normalized": float(d["thr"]),
        "resp_min": float(resp.min()), "resp_max": float(resp.max()),
        "footprint_area": int(d["footprint"].sum()),
        "union_all64_area_latent": int(union_all.sum()),
        "union_all64_fraction_latent": float(union_all.mean()),
        "union_smaller_side_area_latent": int(union_small.sum()),
        "union_smaller_side_fraction_latent": float(union_small.mean()),
        "support_before_area": int(d["support_before"].sum()),
        "support_polarity_area": int(d["support_polarity"].sum()),
        "support_union_area": int(d["support_union"].sum()),
    }
    (out / "stats_v1.txt").write_text("\n".join(f"{k}={v}" for k, v in stats.items()) + "\n")
    (out / "per_channel_otsu.txt").write_text("\n".join(f"ch{c}: thr={ch_thr[c]:.4f} area={int(ch_otsu[c].sum())}" for c in range(C)) + "\n")
    print(open(out / "stats_v1.txt").read())
    print("saved", out / "lfp_qwen21_v1_steps.png")
    print("saved", out / "vae_all_channels.png")
    print("saved", out / "vae_all_channels_otsu.png")


if __name__ == "__main__":
    main()
