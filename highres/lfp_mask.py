"""LFP mask projection adapted for a diffusers VAE (Trim + Bilinear + Fill)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _otsu(x: np.ndarray) -> float:
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom < 1e-12] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return float(edges[int(np.argmax(sigma_b))])


def _lfp_latent(mask: torch.Tensor, vae, scale_factor: float, target_h: int, target_w: int) -> torch.Tensor:
    """mask: [B,1,H,W] in {0,1}; returns [B,1,th,tw] binary latent mask."""
    out = []
    for b in range(mask.shape[0]):
        m = (mask[b:b+1] >= 0.5).float()
        bil = F.interpolate(m.float(), size=(target_h, target_w), mode="bilinear", align_corners=False)
        m3 = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
        m3 = m3.to(next(vae.parameters()).device, next(vae.parameters()).dtype)
        with torch.no_grad():
            z = vae.encode(m3).latent_dist.mode() * scale_factor
        abs_z = z[0].detach().float().cpu().numpy()
        white_minority = float(m[0, 0].mean().item()) < 0.5
        channels = np.zeros_like(abs_z)
        for c in range(4):
            data = abs_z[c].ravel()
            if data.max() <= 1e-6:
                continue
            thr = _otsu((data - data.min()) / max(data.max() - data.min(), 1e-8))
            above = abs_z[c] > data.min() + thr * max(data.max() - data.min(), 1e-8)
            above_minority = above.sum() < above.size / 2
            if white_minority:
                channels[c] = above if above_minority else 1.0 - above
            else:
                channels[c] = above if not above_minority else 1.0 - above
        union = (channels.sum(axis=0) > 0).astype(np.float64)
        # trim one-latent border, merge bilinear support
        union[0, :] = 0; union[-1, :] = 0; union[:, 0] = 0; union[:, -1] = 0
        bil_np = bil[0, 0].cpu().numpy()
        final = np.maximum(union, bil_np)
        # 4-neighbour hole filling
        while True:
            p = np.pad(final, 1, constant_values=0)
            neigh = (p[:-2, 1:-1] > .5).astype(int) + (p[2:, 1:-1] > .5).astype(int) + \
                    (p[1:-1, :-2] > .5).astype(int) + (p[1:-1, 2:] > .5).astype(int)
            fill = (final < .5) & (neigh >= 3)
            if not fill.any():
                break
            final[fill] = 1.0
        out.append(torch.from_numpy((final >= 0.5).astype(np.float32))[None, None])
    return torch.cat(out, 0)


def lfp_pixel_mask(mask_pil: Image.Image, vae, scale_factor: float, vae_scale: int = 8) -> Image.Image:
    """Apply LFP at latent resolution, then upsample support back to pixels."""
    w, h = mask_pil.size
    m = torch.from_numpy(np.asarray(mask_pil.convert("L"))).float()[None, None] / 255.0
    th, tw = h // vae_scale, w // vae_scale
    latent = _lfp_latent(m, vae, scale_factor, th, tw)
    pix = F.interpolate(latent, size=(h, w), mode="nearest")[0, 0].cpu().numpy()
    return Image.fromarray((pix * 255).astype(np.uint8), mode="L")
