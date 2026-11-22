#!/usr/bin/env python3
"""Generate from a saved SDXL placeholder-token embedding (no training)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/sdxl-base-1.0")
    p.add_argument("--embedding", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="highres_runs/sdxl_token")
    p.add_argument("--run_name", default="")
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts")
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    args = p.parse_args()

    from diffusers import StableDiffusionXLPipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ckpt = torch.load(args.embedding, map_location="cpu")
    prompt = args.prompt or ckpt.get("prompt", "")
    placeholder = ckpt["placeholder"]
    vec1 = ckpt["vec1"].to(dtype)
    vec2 = ckpt["vec2"].to(dtype)
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = "sdxl_token_" + Path(args.embedding).parent.name
    out = Path(args.output_dir) / run_name
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    print(f"[SDXL-TI-gen] model={args.model} embedding={args.embedding}")
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.tokenizer.add_tokens([placeholder])
    pipe.tokenizer_2.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
    tid1 = pipe.tokenizer.convert_tokens_to_ids(placeholder)
    tid2 = pipe.tokenizer_2.convert_tokens_to_ids(placeholder)

    def install_hook(encoder, token_id, vec):
        def hook(module, args, kwargs, output):
            ids = kwargs.get("input_ids") if kwargs else None
            if ids is None:
                ids = args[0]
            o = output.clone()
            pos = (ids == token_id).nonzero(as_tuple=False)
            for _, idx in pos:
                o[0, idx] = vec.to(o.dtype)
            return o
        return encoder.text_model.embeddings.register_forward_hook(hook, with_kwargs=True)

    h1 = install_hook(pipe.text_encoder, tid1, vec1)
    h2 = install_hook(pipe.text_encoder_2, tid2, vec2)
    if args.enable_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")
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
        image = pipe(prompt=prompt, negative_prompt=args.negative_prompt,
                     height=args.resolution, width=args.resolution,
                     num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                     generator=generator).images[0]
        name = f"p00_{i:04d}.png"
        image.save(img_dir / name)
        records.append({"file": str(img_dir / name), "seed": seed,
                        "resolution": args.resolution, "output_size": list(image.size)})
        print(f"[SDXL-TI-gen] {i+1}/{args.num_images}")
    h1.remove(); h2.remove()
    manifest = {
        "model": args.model, "embedding": args.embedding, "prompt": prompt,
        "placeholder": placeholder, "resolution": args.resolution,
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "seed": args.seed, "records": records,
        "pipeline_load_sec": round(loaded - start, 2),
        "generation_elapsed_sec": round(time.time() - gen_start, 2),
        "run_elapsed_sec": round(time.time() - start, 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated()/(1024**2), 1) if torch.cuda.is_available() else None,
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved()/(1024**2), 1) if torch.cuda.is_available() else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[SDXL-TI-gen] wrote {args.num_images} images to {out}")


if __name__ == "__main__":
    main()
