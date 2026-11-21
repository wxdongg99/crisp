#!/usr/bin/env python3
"""Basic textual inversion for the official AnomalyDiffusion LDM.

What this does
--------------
Optimizes the embedding of a placeholder token (default ``*``) so that a
frozen LDM denoiser reconstructs a target image under a prompt such as
``a photo of *``.  Only the token embedding is trainable; VAE, UNet and the
BERT text encoder stay frozen.

This is a smoke-testable baseline, not the CRISP Stage-A/Stage-B training.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model(config_path: str, ckpt_path: str, device: str):
    config = OmegaConf.load(config_path)
    # Basic TI here deliberately removes the PSP/personalization manager;
    # the official manager imports custom CUDA ops that are unnecessary for
    # a standalone soft-prompt inversion smoke test.
    config.model.params.personalization_config = None
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd.get("state_dict", pl_sd)
    config.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[TI] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def load_image(path: str, size: int, device: str) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml")
    p.add_argument("--ckpt", required=True, help="Base LDM checkpoint")
    p.add_argument("--image", required=True, help="Target image to invert")
    p.add_argument("--prompt", default="a photo of the object")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="textual_inversion_runs/basic")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_samples", type=int, default=1, help="Images to generate after inversion")
    p.add_argument("--sample_steps", type=int, default=50)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _ = load_model(args.config, args.ckpt, args.device)
    image = load_image(args.image, args.size, args.device)
    with torch.no_grad():
        z = model.get_first_stage_encoding(model.encode_first_stage(image))
        c0, _ = model.get_learned_conditioning([args.prompt])
    delta = torch.zeros_like(c0, requires_grad=True)
    print(f"[TI] latent={tuple(z.shape)} cond={tuple(c0.shape)} prompt={args.prompt!r}")

    opt = torch.optim.AdamW([delta], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    for step in range(1, args.steps + 1):
        t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=args.device).long()
        noise = torch.randn_like(z)
        z_t = model.q_sample(z, t, noise)
        c = c0 + delta
        eps = model.apply_model(z_t, t, c)
        loss = F.mse_loss(eps, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI] step={step:4d} loss={loss.item():.6f}")

    emb_path = out / "inverted_prompt_embedding.pt"
    torch.save({"delta": delta.detach().cpu(), "prompt": args.prompt}, emb_path)
    print(f"[TI] saved {emb_path}")

    # Sample from the learned token.
    sampler = DDIMSampler(model)
    with torch.no_grad():
        samples, _ = sampler.sample(
            S=args.sample_steps,
            batch_size=args.num_samples,
            shape=(4, args.size // 8, args.size // 8),
            conditioning=(c0 + delta).repeat(args.num_samples, 1, 1),
            eta=0.0,
            verbose=False,
        )
        decoded = model.decode_first_stage(samples)
        save_image((decoded + 1) / 2, out / "inverted_sample.png", nrow=1)
    print(f"[TI] wrote {out / 'inverted_sample.png'}")


if __name__ == "__main__":
    main()
