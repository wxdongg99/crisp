"""Background-Principal-Residual Conditioning (BPRC), core primitives.

This is the self-contained core of CRISP's BPRC, kept independent of the
Stage-0 / Stage-A / Stage-B training framework:

    eps = eps_A + alpha * eps_R

where:
  * eps_A is the frozen principal (Stage-A) prediction;
  * eps_R is the residual (Stage-B) prediction, trained on
    ``noise - stop_gradient(eps_A)`` inside the LFP/defect support;
  * alpha is the BPRC residual gain (CRISP: ``--cfg_b``).

The training-side residual target follows CRISP's objective:

    L_R = mean( mask * (eps_R - (noise - sg(eps_A)))^2 )

The inference-side composition is exactly the v10 formula.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BPRCConfig:
    """Small configuration object for BPRC composition/loss."""
    alpha: float = 1.0
    objective: str = "residual"  # "residual" (BPRC) or "full_noise" control

    def __post_init__(self):
        if self.objective not in {"residual", "full_noise"}:
            raise ValueError(
                f"BPRC objective must be 'residual' or 'full_noise', "
                f"got {self.objective!r}")


def compose_bprc_residual(eps_a, eps_r, alpha=1.0):
    """BPRC inference composition: ``eps_A + alpha * eps_R``.

    Args:
        eps_a: frozen Stage-A noise prediction.
        eps_r: Stage-B residual noise prediction.
        alpha: residual gain (CRISP ``--cfg_b``).
    """
    if eps_a.shape != eps_r.shape:
        raise ValueError(
            f"BPRC prediction shapes must match, got "
            f"{tuple(eps_a.shape)} vs {tuple(eps_r.shape)}")
    return eps_a + float(alpha) * eps_r


def bprc_residual_target(noise, eps_a):
    """Return ``noise - stop_gradient(eps_A)`` for Stage-B training."""
    if noise.shape != eps_a.shape:
        raise ValueError(
            f"BPRC noise/eps_A shapes must match, got "
            f"{tuple(noise.shape)} vs {tuple(eps_a.shape)}")
    return noise - eps_a.detach()


def bprc_residual_loss(eps_r, noise, eps_a, mask=None, eps=1e-8):
    """CRISP Stage-B residual loss.

    ``mask`` is the LFP/defect support in latent space. The masked squared
    error is reduced over ``(C, H, W)`` per sample and averaged over the batch,
    matching CRISP's ``get_loss(..., mean=False).mean([1, 2, 3])``.
    """
    target = bprc_residual_target(noise, eps_a)
    diff = (eps_r - target) ** 2
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[0] != diff.shape[0]:
            raise ValueError(
                f"BPRC mask batch {mask.shape[0]} != prediction batch "
                f"{diff.shape[0]}")
        if mask.shape[-2:] != diff.shape[-2:]:
            raise ValueError(
                f"BPRC mask spatial {tuple(mask.shape[-2:])} != prediction "
                f"{tuple(diff.shape[-2:])}")
        diff = diff * mask.float()
    per_sample = diff.mean(dim=(1, 2, 3))
    return per_sample.mean().clamp_min(eps) if eps > 0 else per_sample.mean()


class BPRCResidualComposer:
    """Tiny callable wrapper for the BPRC inference formula."""

    def __init__(self, alpha=1.0):
        self.config = BPRCConfig(alpha=float(alpha))

    def __call__(self, eps_a, eps_r):
        return compose_bprc_residual(
            eps_a, eps_r, alpha=self.config.alpha)

    def __repr__(self):
        return f"BPRCResidualComposer(alpha={self.config.alpha})"
