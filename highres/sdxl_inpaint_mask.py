#!/usr/bin/env python3
"""SDXL mask-conditioned probe with optional learned placeholder token."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import sys
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from highres.lfp_mask import lfp_pixel_mask, lfp_pixel_mask_v3


def load_image(path, size):
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
    ])
    return tf(Image.open(path).convert("RGB"))


def make_mask(size, box):
    m = Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = [int(round(v * size)) for v in box]
    m.paste(255, (x0, y0, x1, y1))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/sdxl-base-1.0")
    p.add_argument("--image", required=True)
    p.add_argument("--mask-box", nargs=4, type=float, default=[0.35, 0.35, 0.65, 0.65])
    p.add_argument("--mask-image", default="", help="optional real defect mask image")
    p.add_argument("--mask-project", default="raw", choices=["raw", "lfp", "lfp_v3"])
    p.add_argument("--composite-mask", default="", help="optional raw defect mask for final compositing")
    p.add_argument("--prompt", default="a realistic macro photograph of natural beige linen fabric")
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts, text, watermark")
    p.add_argument("--embedding", default="", help="optional token embedding .pt from sdxl token TI")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="highres_runs/sdxl_mask")
    p.add_argument("--run_name", default="")
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    args = p.parse_args()

    from diffusers import StableDiffusionXLInpaintPipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = Path(args.output_dir) / (args.run_name or time.strftime("sdxl_mask_%Y%m%d_%H%M%S"))
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    source = load_image(args.image, args.resolution)
    if args.mask_image:
        mask = Image.open(args.mask_image).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
    else:
        mask = make_mask(args.resolution, args.mask_box)
    source.save(out / "source.png")
    mask.save(out / "mask_raw.png")

    start = time.time()
    print(f"[SDXL-mask] model={args.model} resolution={args.resolution}")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, torch_dtype=dtype)
    hooks = []
    if args.embedding:
        ckpt = torch.load(args.embedding, map_location="cpu")
        placeholder = ckpt["placeholder"]
        vec1 = ckpt["vec1"].to(dtype)
        vec2 = ckpt["vec2"].to(dtype)
        pipe.tokenizer.add_tokens([placeholder])
        pipe.tokenizer_2.add_tokens([placeholder])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
        tid1 = pipe.tokenizer.convert_tokens_to_ids(placeholder)
        tid2 = pipe.tokenizer_2.convert_tokens_to_ids(placeholder)
        def expand_mask(mask, ids, token_id, n_vec):
            out = mask.clone()
            for b in range(out.shape[0]):
                pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                p = int(pos[0])
                out[b] = torch.cat([out[b, :p], torch.ones(n_vec, device=out.device, dtype=out.dtype),
                                    out[b, p + 1:]], dim=0)[:out.shape[1]]
            return out
        def install(encoder, token_id, vec):
            def hook(module, a, kw, output):
                ids = (kw or {}).get("input_ids", a[0] if a else None)
                o = output.clone()
                for b in range(o.shape[0]):
                    pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
                    if pos.numel() == 0:
                        continue
                    p = int(pos[0])
                    v = vec.to(device=o.device, dtype=o.dtype)
                    o[b] = torch.cat([o[b, :p], v, o[b, p + 1:]], dim=0)[:o.shape[1]]
                return o
            emb = encoder.text_model.embeddings.register_forward_hook(hook, with_kwargs=True)
            def mask_pre(module, hargs, hkwargs):
                ids = hkwargs.get("input_ids")
                mask = hkwargs.get("attention_mask")
                if ids is not None and mask is not None:
                    hkwargs["attention_mask"] = expand_mask(mask, ids, token_id, vec.shape[0])
                return hargs, hkwargs
            mh = encoder.register_forward_pre_hook(mask_pre, with_kwargs=True)
            return emb, mh
        hooks = [install(pipe.text_encoder, tid1, vec1), install(pipe.text_encoder_2, tid2, vec2)]
        print(f"[SDXL-mask] loaded token embedding {args.embedding}")
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
    if args.mask_project in ("lfp", "lfp_v3"):
        vae = pipe.vae
        dev = next(vae.parameters()).device
        if dev.type != "cpu":
            vae.to("cpu")
        if args.mask_project == "lfp":
            mask = lfp_pixel_mask(mask, vae, pipe.vae.config.scaling_factor, pipe.vae_scale_factor)
        else:
            mask = lfp_pixel_mask_v3(mask, vae, pipe.vae.config.scaling_factor, pipe.vae_scale_factor)
        if dev.type != "cpu":
            vae.to(dev)
        print(f"[SDXL-mask] LFP projected mask saved")
    mask.save(out / "mask.png")
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loaded = time.time()
    records = []
    gen_start = time.time()
    mask_np = np.asarray(mask) > 127
    for i in range(args.num_images):
        seed = args.seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        image = pipe(prompt=args.prompt, negative_prompt=args.negative_prompt,
                     image=source, mask_image=mask, height=args.resolution, width=args.resolution,
                     num_inference_steps=args.steps, strength=args.strength,
                     guidance_scale=args.guidance_scale, generator=gen).images[0]
        name = f"p00_{i:04d}.png"
        image.save(img_dir / name)
        arr = np.asarray(image).astype(np.float32)
        src = np.asarray(source).astype(np.float32)
        bg = np.abs(arr[~mask_np] - src[~mask_np]).mean()
        composite_path = ""
        if args.composite_mask:
            cm = Image.open(args.composite_mask).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
            cropped = Image.composite(image, source, cm)
            composite_path = str(out / f"composite_{i:04d}.png")
            cropped.save(composite_path)
        records.append({"file": str(img_dir / name), "seed": seed, "output_size": list(image.size),
                        "background_mae_outside_mask": round(float(bg), 3),
                        "composite": composite_path})
        print(f"[SDXL-mask] {i+1}/{args.num_images} bg_mae={bg:.2f}")
    for h in hooks:
        if isinstance(h, (tuple, list)):
            for hh in h:
                hh.remove()
        else:
            h.remove()
    manifest = {
        "model": args.model, "image": args.image, "mask_box": args.mask_box,
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "mask_image": args.mask_image, "mask_project": args.mask_project,
        "composite_mask": args.composite_mask,
        "embedding": args.embedding, "resolution": args.resolution,
        "steps": args.steps, "strength": args.strength, "seed": args.seed,
        "records": records,
        "pipeline_load_sec": round(loaded - start, 2),
        "generation_elapsed_sec": round(time.time() - gen_start, 2),
        "run_elapsed_sec": round(time.time() - start, 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated()/(1024**2), 1) if torch.cuda.is_available() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[SDXL-mask] wrote {out}")


if __name__ == "__main__":
    main()
