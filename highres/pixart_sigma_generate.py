#!/usr/bin/env python3
"""PixArt-Sigma native high-resolution baseline.

This is intentionally a standalone harness: it does NOT modify the CRISP
LDM/Stage-A/B training path. It is used to validate native 512/1024/2048
generation before deciding whether a model migration is justified.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
                   help="HF model id or local diffusers directory")
    p.add_argument("--prompt", default="a high-resolution photograph of woven fabric texture")
    p.add_argument("--prompt_file", default="", help="One prompt per line; overrides --prompt")
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts")
    p.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--height", type=int, default=0, help="Overrides --resolution")
    p.add_argument("--width", type=int, default=0, help="Overrides --resolution")
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1,
                   help="Images per forward pass; use 1 for 2048.")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    p.add_argument("--use_resolution_binning", action="store_true",
                   help="Enable PixArt aspect-ratio binning. Default false: denoise at the requested native resolution.")
    p.add_argument("--output_dir", default="highres_runs/pixart_sigma")
    p.add_argument("--run_name", default="")
    return p.parse_args()


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file:
        path = Path(args.prompt_file)
        if not path.is_file():
            raise FileNotFoundError(path)
        prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.lstrip().startswith("#")]
        if not prompts:
            raise ValueError(f"No prompts in {path}")
        return prompts
    return [args.prompt]


def main() -> None:
    args = parse_args()
    from diffusers import PixArtSigmaPipeline

    height = args.height or args.resolution
    width = args.width or args.resolution
    if height % 64 or width % 64:
        raise ValueError("PixArt-Sigma height/width must be multiples of 64")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.device == "cpu":
        dtype = torch.float32

    run_name = args.run_name or time.strftime("pixart_%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / run_name
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PixArt-Sigma] model={args.model}")
    print(f"[PixArt-Sigma] resolution={width}x{height} dtype={args.dtype} device={args.device}")
    pipe = PixArtSigmaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()

    vae_scale_factor = int(pipe.vae_scale_factor)
    transformer_sample_size = int(pipe.transformer.config.sample_size)
    latent_height = height // vae_scale_factor
    latent_width = width // vae_scale_factor
    if height % vae_scale_factor or width % vae_scale_factor:
        raise ValueError(f"Resolution must be divisible by VAE scale factor {vae_scale_factor}")
    print(f"[PixArt-Sigma] denoise={width}x{height} latent={latent_width}x{latent_height} "
          f"sample_size={transformer_sample_size} resolution_binning={args.use_resolution_binning}")

    prompts = load_prompts(args)
    records = []
    for prompt_idx, prompt in enumerate(prompts):
        done = 0
        while done < args.num_images:
            n = min(args.batch_size, args.num_images - done)
            seed = args.seed + prompt_idx * 100000 + done
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=n,
                generator=generator,
                use_resolution_binning=args.use_resolution_binning,
            )
            for j, image in enumerate(result.images):
                idx = done + j
                name = f"p{prompt_idx:02d}_{idx:04d}.png"
                image.save(img_dir / name)
                records.append({
                    "file": str(img_dir / name),
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "seed": seed,
                    "index": idx,
                    "requested_width": width,
                    "requested_height": height,
                    "denoise_width": width,
                    "denoise_height": height,
                    "latent_width": latent_width,
                    "latent_height": latent_height,
                    "vae_scale_factor": vae_scale_factor,
                    "transformer_sample_size": transformer_sample_size,
                    "use_resolution_binning": args.use_resolution_binning,
                    "output_width": image.width,
                    "output_height": image.height,
                })
            done += n
            print(f"[PixArt-Sigma] prompt {prompt_idx + 1}/{len(prompts)}: {done}/{args.num_images}")

    manifest = {
        "model": args.model,
        "height": height,
        "width": width,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "use_resolution_binning": args.use_resolution_binning,
        "vae_scale_factor": vae_scale_factor,
        "transformer_sample_size": transformer_sample_size,
        "prompts": prompts,
        "records": records,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[PixArt-Sigma] wrote {len(records)} images to {out}")


if __name__ == "__main__":
    main()
