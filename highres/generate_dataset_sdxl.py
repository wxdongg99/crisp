#!/usr/bin/env python3
"""Batch SDXL inpainting dataset generator for classification/localization eval."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]


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
    def emb_hook(module, a, kw, output):
        ids = (kw or {}).get("input_ids", a[0] if a else None)
        o = output.clone()
        for b in range(o.shape[0]):
            pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                continue
            p = int(pos[0]); v = vec.to(device=o.device, dtype=o.dtype)
            o[b] = torch.cat([o[b, :p], v, o[b, p + 1:]], dim=0)[:o.shape[1]]
        return o
    def pre_hook(module, hargs, hkwargs):
        ids = hkwargs.get("input_ids"); mask = hkwargs.get("attention_mask")
        if ids is not None and mask is not None:
            hkwargs["attention_mask"] = expand_mask(mask, ids, token_id, vec.shape[0])
        return hargs, hkwargs
    return (encoder.text_model.embeddings.register_forward_hook(emb_hook, with_kwargs=True),
            encoder.register_forward_pre_hook(pre_hook, with_kwargs=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--token-dir", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--prompt", default="", help="empty = canonical imagenet template")
    p.add_argument("--template-index", type=int, default=0)
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    from diffusers import StableDiffusionXLInpaintPipeline
    dtype = torch.bfloat16
    out = Path(args.output_dir) / "fabric" / args.pattern
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    srcs = sorted([x for x in Path(args.source_dir).glob("*") if x.is_file()])
    masks = sorted([x for x in Path(args.mask_dir).glob("*") if x.is_file()])
    if not srcs or not masks:
        raise RuntimeError("empty source or mask list")
    rng = random.Random(args.seed)

    ckpt = torch.load(Path(args.token_dir) / "sdxl_token_embedding_ema.pt", map_location="cpu")
    placeholder = ckpt["placeholder"]
    if not args.prompt:
        from highres.prompt_templates import canonical_prompt
        args.prompt = canonical_prompt(placeholder, args.template_index)
    vec1 = ckpt["vec1"].to(dtype); vec2 = ckpt["vec2"].to(dtype)
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.tokenizer.add_tokens([placeholder]); pipe.tokenizer_2.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
    tid1 = pipe.tokenizer.convert_tokens_to_ids(placeholder); tid2 = pipe.tokenizer_2.convert_tokens_to_ids(placeholder)
    hooks = [install(pipe.text_encoder, tid1, vec1), install(pipe.text_encoder_2, tid2, vec2)]
    pipe = pipe.to(args.device)
    pipe.enable_attention_slicing()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.count):
        src = Image.open(srcs[i % len(srcs)]).convert("RGB").resize((args.resolution, args.resolution), Image.Resampling.BICUBIC)
        mask = Image.open(masks[(i * 37 + args.seed) % len(masks)]).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
        gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
        with torch.no_grad():
            im = pipe(prompt=args.prompt, negative_prompt="blurry, low resolution, jpeg artifacts, text, watermark",
                      image=src, mask_image=mask, height=args.resolution, width=args.resolution,
                      num_inference_steps=args.steps, strength=args.strength,
                      guidance_scale=args.cfg, generator=gen).images[0]
        im.save(out / "image" / f"{i}.jpg", quality=95)
        mask.save(out / "mask" / f"{i}.jpg", quality=95)
        if (i + 1) % 50 == 0:
            print(f"[gen] {args.pattern} {i+1}/{args.count} elapsed={time.time()-t0:.1f}s", flush=True)
    for pair in hooks:
        for h in pair: h.remove()
    (out / "manifest.json").write_text(json.dumps({
        "pattern": args.pattern, "prompt": args.prompt, "count": args.count,
        "resolution": args.resolution, "steps": args.steps, "strength": args.strength,
        "cfg": args.cfg, "seed": args.seed,
        "elapsed_sec": round(time.time()-t0, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated()/1024**2, 1) if torch.cuda.is_available() else None,
    }, indent=2))
    print("[gen] done", out, flush=True)


if __name__ == "__main__":
    main()
