#!/usr/bin/env python3
"""Compare mask-in anomaly latent reconstruction using LFP v1 vs LFP v3 masks.

z_recon = z_fill.clone()
z_recon[mask inside]  = z_orig[mask inside]   # keep anomaly latent
z_recon[mask outside] = z_fill[mask outside]  # clean fill latent

The same z_orig / z_fill are used for both masks; only the replacement region differs.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from highres.textual_inversion_qwen21 import vae_latents, lfp_v1_mask_qwen, lfp_v3_mask_qwen


def to_tensor(im, size):
    a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
    rgb = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
    return torch.cat([rgb, torch.ones_like(rgb[:, :1])], 1)


def decode(vae, z):
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    with torch.no_grad():
        im = vae.decode(z.unsqueeze(2) * std + mean, return_dict=False)[0][:, :, 0]
    im = ((im.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((im * 255).round().astype(np.uint8))


def dino(model, im1, im2, size, device):
    def prep(im):
        a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
        a = (a - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)[None]).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model.forward_features(torch.cat([prep(im1), prep(im2)], 0))
    cls = out["x_norm_clstoken"]; patch = out["x_norm_patchtokens"]
    cls_cos = float(F.cosine_similarity(cls[:1], cls[1:], dim=-1).item())
    pc = F.cosine_similarity(patch[0], patch[1], dim=-1)
    gh = int(round(patch.shape[1] ** 0.5)); grid = pc.reshape(gh, gh).cpu().numpy()
    return cls_cos, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--dinov2-size", type=int, default=518)
    ap.add_argument("--dinov2-repo", default="/data/solo/wxd/torch_hub/hub/facebookresearch_dinov2_main")
    ap.add_argument("--dinov2-weights", default="/data/solo/wxd/torch_hub/hub/checkpoints/dinov2_vitb14_pretrain.pth")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from diffusers import AutoencoderKLQwenImage21
    vae = AutoencoderKLQwenImage21.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(dev).eval()

    ori = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC)
    raw = Image.open(args.mask).convert("L").resize((args.size, args.size), Image.Resampling.NEAREST)
    v1 = lfp_v1_mask_qwen(raw, vae, vae_scale=16)
    v3 = lfp_v3_mask_qwen(raw, vae, vae_scale=16)
    arr = np.array(ori); m_raw = np.array(raw) > 127
    color = np.round(arr[~m_raw].reshape(-1, 3).mean(0)).astype(np.uint8)
    fill = arr.copy(); fill[m_raw] = color
    I_fill = Image.fromarray(fill)
    z_orig = vae_latents(vae, to_tensor(ori, args.size).to(dev))
    z_fill = vae_latents(vae, to_tensor(I_fill, args.size).to(dev))
    masks = {"v1": v1, "v3": v3}
    results = {}
    recons = {}
    for name, mpil in masks.items():
        m = np.array(mpil) > 127
        ml = F.interpolate(torch.from_numpy(m.astype(np.float32))[None, None].to(dev),
                           size=z_orig.shape[-2:], mode="nearest")
        z_recon = z_fill.clone()
        z_recon = torch.where(ml > 0.5, z_orig, z_fill)
        recon = decode(vae, z_recon)
        recons[name] = recon
        # DINO against ori and fill
        model = None
        results[name] = {"mask": m, "ml": ml, "recon": recon}
    # Load DINO once
    repo = Path(args.dinov2_repo); weights = Path(args.dinov2_weights)
    sys.path.insert(0, str(repo))
    from dinov2.hub.backbones import dinov2_vitb14
    model = dinov2_vitb14(pretrained=False)
    model.load_state_dict(torch.load(str(weights), map_location="cpu"))
    model = model.to(dev).eval()
    for name, r in results.items():
        r["cls_ori"], r["grid_ori"] = dino(model, r["recon"], ori, args.dinov2_size, dev)
        r["cls_fill"], r["grid_fill"] = dino(model, r["recon"], I_fill, args.dinov2_size, dev)
        gh = r["grid_ori"].shape[0]
        mp = np.array(masks[name].resize((gh, gh), Image.Resampling.NEAREST)) > 127
        r["mask_cos_ori"] = float(r["grid_ori"][mp].mean()) if mp.any() else float("nan")
        r["out_cos_ori"] = float(r["grid_ori"][~mp].mean()) if (~mp).any() else float("nan")
        r["mask_cos_fill"] = float(r["grid_fill"][mp].mean()) if mp.any() else float("nan")
        r["out_cos_fill"] = float(r["grid_fill"][~mp].mean()) if (~mp).any() else float("nan")
        z_recon = z_fill.clone(); z_recon = torch.where(r["ml"] > 0.5, z_orig, z_fill)
        r["lat_in"] = float(((z_recon - z_orig).abs() * r["ml"]).sum() / r["ml"].sum())
        r["lat_out"] = float(((z_recon - z_orig).abs() * (1 - r["ml"])).sum() / (1 - r["ml"]).sum())
        print(f"[{name}] area={float(r['mask'].mean()):.4f} cls_ori={r['cls_ori']:.4f} mask_cos_ori={r['mask_cos_ori']:.4f} "
              f"out_cos_ori={r['out_cos_ori']:.4f} mask_cos_fill={r['mask_cos_fill']:.4f} lat_in={r['lat_in']:.4f} lat_out={r['lat_out']:.4f}", flush=True)
    # save artifacts
    for name in ["v1", "v3"]:
        recons[name].save(out / f"I_recon_{name}.png")
        masks[name].save(out / f"mask_{name}.png")
    ori.save(out / "I_ori.png"); I_fill.save(out / "I_fill.png")
    # overlays + sheet
    ov = np.array(ori).copy()
    ov[results["v1"]["mask"]] = (0.5 * ov[results["v1"]["mask"]] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
    ov[results["v3"]["mask"]] = (0.7 * ov[results["v3"]["mask"]] + 0.3 * np.array([0, 80, 255])).astype(np.uint8)
    Image.fromarray(ov).save(out / "mask_v1_v3_overlay.png")
    panels = [
        ("I_ori", ori), ("raw mask", raw.convert("RGB")), ("I_fill", I_fill),
        ("LFP v1", masks["v1"].convert("RGB")), ("I_recon_v1", recons["v1"]),
        ("LFP v3", masks["v3"].convert("RGB")), ("I_recon_v3", recons["v3"]),
        ("mask v1/v3 overlay", Image.fromarray(ov)),
    ]
    W, H, cols = 330, 365, 4
    sheet = Image.new("RGB", (W * cols, H * 2), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(panels):
        x = (i % cols) * W; y = (i // cols) * H
        sheet.paste(im.resize((W - 8, W - 8), Image.Resampling.LANCZOS), (x + 4, y + 28)); d.text((x + 6, y + 6), lab, fill=(0, 0, 0))
    sheet.save(out / "v1_v3_recon_compare.png")
    # stats
    lines = [f"image={args.image}", f"mask={args.mask}", f"size={args.size}", f"fill_color={color.tolist()}"]
    for name in ["v1", "v3"]:
        r = results[name]
        lines += [
            f"[{name}]",
            f"area_fraction={float(r['mask'].mean()):.6f}",
            f"dinov2_cls_vs_ori={r['cls_ori']:.6f}",
            f"dinov2_mask_patch_vs_ori={r['mask_cos_ori']:.6f}",
            f"dinov2_outside_patch_vs_ori={r['out_cos_ori']:.6f}",
            f"dinov2_cls_vs_fill={r['cls_fill']:.6f}",
            f"dinov2_mask_patch_vs_fill={r['mask_cos_fill']:.6f}",
            f"latent_absdiff_inside_vs_orig={r['lat_in']:.6f}",
            f"latent_absdiff_outside_vs_orig={r['lat_out']:.6f}",
        ]
    (out / "stats_v1_v3.txt").write_text("\n".join(lines) + "\n")
    print(open(out / "stats_v1_v3.txt").read())
    print("saved", out / "v1_v3_recon_compare.png")


if __name__ == "__main__":
    main()
