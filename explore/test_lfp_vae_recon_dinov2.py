#!/usr/bin/env python3
"""Fill LFP mask with a solid color, VAE-recompose, decode, and score with DINOv2.

Pipeline:
  I_ori -> z_orig
  I_fill (LFp mask set to solid color) -> z_fill
  z_recon = (1-M_lfp)*z_orig + M_lfp*z_fill
  I_recon = VAE.decode(z_recon)
  DINOv2 cosine similarity / masked-patch change score between I_recon and I_ori

Run on B300:
  CUDA_VISIBLE_DEVICES=2 TORCH_HOME=/data/solo/wxd/torch_hub PYTHONPATH=$PWD \
  /data/solo/wxd/envs/qwen21_py310/bin/python explore/test_lfp_vae_recon_dinov2.py \
    --model /data/solo/wxd/models/Qwen-Image-2.1 \
    --image /data/solo/wxd/data/zju_leaper_native512/fabric/test/pattern16/001.jpg \
    --mask /data/solo/wxd/data/zju_leaper_native512/fabric/ground_truth/pattern16/001.png \
    --size 1024 --out_dir explore/lfp_recon_dinov2/pattern16_001
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
from highres.textual_inversion_qwen21 import vae_latents, lfp_v3_mask_qwen


def to_tensor(im: Image.Image, size: int):
    a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
    rgb = torch.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1
    alpha = torch.ones_like(rgb[:, :1])  # Qwen VAE encode expects RGB + alpha
    return torch.cat([rgb, alpha], dim=1)


def decode(vae, z):
    mean = torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    with torch.no_grad():
        im = vae.decode(z.unsqueeze(2) * std + mean, return_dict=False)[0][:, :, 0]
    im = ((im.clamp(-1, 1) + 1) / 2)[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((im * 255).round().astype(np.uint8))


def dino_patch_similarity(model, im1, im2, size=518, device="cuda"):
    # DINOv2 ViT-B/14: 518 = 37*14
    def prep(im):
        a = np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)).astype(np.float32) / 255.0
        a = (a - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)[None]).astype(np.float32)).to(device)
    with torch.no_grad():
        out = model.forward_features(torch.cat([prep(im1), prep(im2)], 0))
    cls = out["x_norm_clstoken"]
    patch = out["x_norm_patchtokens"]
    cls_cos = float(F.cosine_similarity(cls[0:1], cls[1:2], dim=-1).item())
    patch_cos = F.cosine_similarity(patch[0], patch[1], dim=-1)  # [N]
    gh = int(round(patch.shape[1] ** 0.5))
    grid = patch_cos.reshape(gh, gh).float().cpu().numpy()
    return cls_cos, grid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fill-mode", default="mean", choices=["mean", "gray", "white"])
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
    lfp = lfp_v3_mask_qwen(raw, vae, vae_scale=16)
    m = np.array(lfp) > 127
    arr = np.array(ori).copy()
    if args.fill_mode == "gray":
        color = np.array([128, 128, 128], dtype=np.uint8)
    elif args.fill_mode == "white":
        color = np.array([255, 255, 255], dtype=np.uint8)
    else:
        color = np.round(arr[~m].reshape(-1, 3).mean(0)).astype(np.uint8)
    fill = arr.copy(); fill[m] = color
    fill_im = Image.fromarray(fill)
    I_ori = ori
    I_fill = fill_im
    z_orig = vae_latents(vae, to_tensor(I_ori, args.size).to(dev))
    z_fill = vae_latents(vae, to_tensor(I_fill, args.size).to(dev))
    # latent-size LFP mask (nearest)
    ml = F.interpolate(torch.from_numpy((m.astype(np.float32)))[None, None].to(dev),
                       size=z_orig.shape[-2:], mode="nearest")
    # Goal: isolate how much anomaly remains inside the LFP latent mask.
    # Start from z_fill (normal fill), keep z_orig inside the mask, z_fill outside.
    z_recon = z_fill.clone()
    z_recon = torch.where(ml > 0.5, z_orig, z_fill)
    with torch.no_grad():
        I_recon = decode(vae, z_recon)
    I_recon.save(out / "I_recon.png")
    I_ori.save(out / "I_ori.png")
    I_fill.save(out / "I_fill.png")
    lfp.save(out / "mask_lfp_v3.png")
    # DINOv2: prefer the local cached repo + explicit checkpoint (offline-safe)
    repo = Path(args.dinov2_repo); weights = Path(args.dinov2_weights)
    if repo.exists() and weights.exists():
        import sys as _sys
        _sys.path.insert(0, str(repo))
        from dinov2.hub.backbones import dinov2_vitb14
        model = dinov2_vitb14(pretrained=False)
        model.load_state_dict(torch.load(str(weights), map_location="cpu"))
        print("[recon] loaded local DINOv2", repo, weights, flush=True)
    else:
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", source="github")
    model = model.to(dev).eval()
    cls_cos, grid = dino_patch_similarity(model, I_recon, I_ori, args.dinov2_size, dev)
    # map LFP mask to DINO patch grid and average patch cosine inside/outside
    mp = np.array(lfp.resize((grid.shape[1], grid.shape[0]), Image.Resampling.NEAREST)) > 127
    mask_cos = float(grid[mp].mean()) if mp.any() else float("nan")
    outside_cos = float(grid[~mp].mean()) if (~mp).any() else float("nan")
    cls_cos_fill, grid_fill = dino_patch_similarity(model, I_recon, I_fill, args.dinov2_size, dev)
    fill_mask_cos = float(grid_fill[mp].mean()) if mp.any() else float("nan")
    fill_outside_cos = float(grid_fill[~mp].mean()) if (~mp).any() else float("nan")
    # save patch cosine heatmap
    g = np.clip((grid + 1) / 2, 0, 1)
    heat = (g * 255).astype(np.uint8)
    Image.fromarray(heat, mode="L").resize((args.size, args.size), Image.Resampling.BILINEAR).save(out / "dinov2_patch_cosine.png")
    # latent-level visualization (mean over 64 channels)
    def latent_heat(t):
        a = t[0].float().mean(0).cpu().numpy()
        lo, hi = float(a.min()), float(a.max()); n = (a - lo) / max(hi - lo, 1e-8)
        r = np.clip(n * 2, 0, 1); b = np.clip((1 - n) * 2, 0, 1); g = np.clip(1 - np.abs(n * 2 - 1), 0, 1)
        return Image.fromarray((np.stack([r, g, b], -1) * 255).astype(np.uint8)).resize((args.size, args.size), Image.Resampling.BILINEAR)
    zmask = (ml[0, 0] > 0.5).float()
    lat_diff_out = (z_recon - z_orig).abs().mean(0) * (1 - zmask)
    lat_panels = [("z_orig mean", latent_heat(z_orig)), ("z_fill mean", latent_heat(z_fill)),
                  ("z_recon mean", latent_heat(z_recon)), ("|z_recon-z_orig| × outside", latent_heat(lat_diff_out[None]))]
    lW = 420
    lsheet = Image.new("RGB", (lW * 4, lW + 30), (255, 255, 255)); ld = ImageDraw.Draw(lsheet)
    for i, (lab, im) in enumerate(lat_panels):
        lsheet.paste(im.resize((lW - 8, lW - 8), Image.Resampling.LANCZOS), (i * lW + 4, 26)); ld.text((i * lW + 6, 6), lab, fill=(0, 0, 0))
    lsheet.save(out / "latent_recon_compare.png")
    # visualization sheet
    panels = [("I_ori", I_ori), ("LFP v3 mask", lfp.convert("RGB")), ("I_fill", I_fill), ("I_recon", I_recon)]
    W, H = 420, 460
    sheet = Image.new("RGB", (W * 4, H), (255, 255, 255)); d = ImageDraw.Draw(sheet)
    for i, (lab, im) in enumerate(panels):
        sheet.paste(im.resize((W - 8, W - 8), Image.Resampling.LANCZOS), (i * W + 4, 30)); d.text((i * W + 6, 8), lab, fill=(0, 0, 0))
    sheet.save(out / "recon_compare.png")
    (out / "stats.txt").write_text(
        f"size={args.size}\nfill_mode={args.fill_mode}\nfill_color={color.tolist()}\n"
        f"lfp_fraction={float(m.mean()):.6f}\n"
        f"dinov2_cls_cosine_recon_vs_ori={cls_cos:.6f}\n"
        f"dinov2_patch_cosine_mask_mean_vs_ori={mask_cos:.6f}\n"
        f"dinov2_patch_cosine_outside_mean_vs_ori={outside_cos:.6f}\n"
        f"dinov2_cls_cosine_recon_vs_fill={cls_cos_fill:.6f}\n"
        f"dinov2_patch_cosine_mask_mean_vs_fill={fill_mask_cos:.6f}\n"
        f"dinov2_patch_cosine_outside_mean_vs_fill={fill_outside_cos:.6f}\n"
        f"dinov2_mask_change_1_minus_cos={1.0-mask_cos:.6f}\n"
        f"dinov2_global_change_1_minus_cls={1.0-cls_cos:.6f}\n"
        f"dinov2_patch_grid={grid.shape[0]}x{grid.shape[1]}\n"
        f"latent_absdiff_inside_mask_vs_orig={float(((z_recon-z_orig).abs()*zmask).sum()/zmask.sum().clamp_min(1)):.6f}\n"
        f"latent_absdiff_outside_mask_vs_orig={float(((z_recon-z_orig).abs()*(1-zmask)).sum()/(1-zmask).sum().clamp_min(1)):.6f}\n")
    print(open(out / "stats.txt").read())
    print("saved", out / "recon_compare.png")


if __name__ == "__main__":
    main()
