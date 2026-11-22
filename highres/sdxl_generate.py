#!/usr/bin/env python3
"""Minimal SDXL base baseline for high-resolution exploration."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/sdxl-base-1.0")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts")
    p.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="highres_runs/sdxl")
    p.add_argument("--run_name", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from diffusers import StableDiffusionXLPipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.device == "cpu":
        dtype = torch.float32
    run_name = args.run_name or time.strftime("sdxl_%Y%m%d_%H%M%S")
    out = Path(args.output_dir) / run_name
    imgs = out / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    print(f"[SDXL] model={args.model} resolution={args.resolution} dtype={args.dtype}")
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=dtype, variant="fp16", use_safetensors=True)
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(args.device)
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loaded = time.time()
    records = []
    gen_start = time.time()
    for i in range(args.num_images):
        seed = args.seed + i
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        name = f"p00_{i:04d}.png"
        image.save(imgs / name)
        records.append({"file": str(imgs / name), "seed": seed, "prompt": args.prompt,
                        "resolution": args.resolution, "output_size": list(image.size)})
        print(f"[SDXL] {i+1}/{args.num_images}")
    manifest = {
        "model": args.model, "resolution": args.resolution, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "dtype": args.dtype, "seed": args.seed,
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "records": records,
        "pipeline_load_sec": round(loaded - start, 2),
        "generation_elapsed_sec": round(time.time() - gen_start, 2),
        "run_elapsed_sec": round(time.time() - start, 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated()/(1024**2), 1) if torch.cuda.is_available() else None,
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved()/(1024**2), 1) if torch.cuda.is_available() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[SDXL] wrote {args.num_images} images to {out}")


if __name__ == "__main__":
    main()
