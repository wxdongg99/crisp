"""CRISP LFP mask downsampling.

Extracted from ``ldm/models/diffusion/ddpm.py`` so the LFP implementation
lives in its own module next to ``ddpm.py``.

LFP = Trim + Bilinear + Fill:
    VAE-encode mask -> per-channel Otsu -> 4-channel union -> trim border
    -> max with bilinear support -> iterative hole filling -> binarize.
"""

import numpy as np
import torch
from skimage.filters import threshold_otsu


def tbf_resize_mask(mask, target_h, target_w, vae, scale_factor):
    """Project a pixel mask to latent resolution with CRISP LFP/TBF.

    Args:
        mask: (B, 1, H, W) binary/continuous mask on any device.
        target_h: target latent height.
        target_w: target latent width.
        vae: first-stage autoencoder exposing ``encode`` returning a
            distribution with ``.mode()``.
        scale_factor: LDM latent scale factor (e.g. 0.18215).

    Returns:
        (B, 1, target_h, target_w) binary float tensor on ``mask.device``.
    """
    mask = (mask >= 0.5).float()
    results = []
    for b in range(mask.shape[0]):
        m = mask[b:b + 1]

        m_bilinear = torch.nn.functional.interpolate(
            m.float(), size=(target_h, target_w),
            mode="bilinear", align_corners=False)
        m_bilinear_np = m_bilinear[0, 0].cpu().numpy()

        # VAE-encode the binary mask as a 3-channel image in [-1, 1].
        m_3ch = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
        with torch.no_grad():
            posterior = vae.encode(m_3ch)
            z = posterior.mode() * scale_factor
        mask_latent = z[0].detach().float().cpu().numpy()
        lh, lw = mask_latent.shape[1], mask_latent.shape[2]
        n_pixels = lh * lw

        m_np = m[0, 0].float().cpu().numpy()
        white_is_minority = m_np.sum() / m_np.size < 0.5

        abs_latent = np.abs(mask_latent)
        vae_mask_channels = np.zeros_like(mask_latent)
        for ch in range(4):
            ch_data = abs_latent[ch].ravel()
            if ch_data.max() > 1e-6:
                thr = threshold_otsu(ch_data)
                above = (abs_latent[ch] > thr).astype(np.float64)
                above_is_minority = above.sum() < n_pixels / 2
                if white_is_minority:
                    vae_mask_channels[ch] = (
                        above if above_is_minority else 1.0 - above)
                else:
                    vae_mask_channels[ch] = (
                        above if not above_is_minority else 1.0 - above)

        vae_mask_union = (vae_mask_channels.sum(axis=0) > 0).astype(np.float64)

        # Trim the border, then merge with the bilinear reference.
        vae_mask_trimmed = vae_mask_union.copy()
        vae_mask_trimmed[0, :] = 0
        vae_mask_trimmed[-1, :] = 0
        vae_mask_trimmed[:, 0] = 0
        vae_mask_trimmed[:, -1] = 0
        vae_mask_final = np.maximum(vae_mask_trimmed, m_bilinear_np)

        # Fill holes: >=3 of 4 neighbours white.
        while True:
            padded = np.pad(vae_mask_final, 1, mode="constant",
                            constant_values=0)
            up = padded[:-2, 1:-1]
            down = padded[2:, 1:-1]
            left = padded[1:-1, :-2]
            right = padded[1:-1, 2:]
            neighbours_white = (
                (up > 0.5).astype(int) + (down > 0.5).astype(int) +
                (left > 0.5).astype(int) + (right > 0.5).astype(int)) >= 3
            fill_candidates = (vae_mask_final < 0.5) & neighbours_white
            if fill_candidates.sum() == 0:
                break
            vae_mask_final[fill_candidates] = 1.0

        # LFP returns a binary support mask for the inpainting sampler.
        vae_mask_final = (vae_mask_final >= 0.5).astype(np.float64)
        results.append(
            torch.from_numpy(vae_mask_final).float()
            .unsqueeze(0).unsqueeze(0))

    return torch.cat(results, dim=0).to(mask.device)
