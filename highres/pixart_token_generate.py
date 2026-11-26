#!/usr/bin/env python3
"""Generate from a saved PixArt-Sigma T5 token embedding."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/pixart_sigma_1024")
    p.add_argument("--embedding", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 2048])
    p.add_argument("--num_images", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="highres_runs/pixart_token")
    p.add_argument("--run_name", default="")
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts")
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    args = p.parse_args()

    from diffusers import PixArtSigmaPipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    ckpt = torch.load(args.embedding, map_location="cpu")
    placeholder = ckpt["placeholder"]
    prompt = args.prompt or ckpt.get("prompt", "")
    if not prompt:
        from highres.prompt_templates import canonical_prompt
        prompt = canonical_prompt(placeholder, int(ckpt.get("template_index", 0)))
    vec = ckpt["vec"].to(dtype)
    num_vectors = vec.shape[0]
    out = Path(args.output_dir) / (args.run_name or ("pixart_token_" + Path(args.embedding).parent.name))
    imgs = out / "images"
    imgs.mkdir(parents=True, exist_ok=True)
    start = time.time()
    pipe = PixArtSigmaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.tokenizer.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    tid = pipe.tokenizer.convert_tokens_to_ids(placeholder)

    def encode_text(prompt, with_vec):
        tok = pipe.tokenizer([prompt], padding="max_length", max_length=120,
                             truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(pipe.text_encoder.device)
        mask = tok.attention_mask.to(pipe.text_encoder.device)
        emb = pipe.text_encoder.get_input_embeddings()(ids).detach().clone()
        if with_vec:
            for b in range(emb.shape[0]):
                pos = (ids[b] == tid).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                p = int(pos[0])
                v = vec.to(device=emb.device, dtype=emb.dtype)
                emb[b] = torch.cat([emb[b, :p], v, emb[b, p + 1:]], dim=0)[:emb.shape[1]]
                mask[b] = torch.cat([mask[b, :p], torch.ones(v.shape[0], device=mask.device, dtype=mask.dtype), mask[b, p + 1:]], dim=0)[:mask.shape[1]]
        h = pipe.text_encoder(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        return h, mask

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
        gen = torch.Generator(device="cpu").manual_seed(seed)
        pos, pos_mask = encode_text(prompt, True)
        neg, neg_mask = encode_text(args.negative_prompt, False)
        image = pipe(prompt_embeds=pos, prompt_attention_mask=pos_mask,
                     negative_prompt=None, negative_prompt_embeds=neg, negative_prompt_attention_mask=neg_mask,
                     height=args.resolution, width=args.resolution,
                     num_inference_steps=args.steps, guidance_scale=args.guidance_scale,
                     generator=gen, use_resolution_binning=False).images[0]
        name = f"p00_{i:04d}.png"
        image.save(imgs / name)
        records.append({"file": str(imgs / name), "seed": seed, "output_size": list(image.size)})
        print(f"[PixArt-TI-gen] {i+1}/{args.num_images}")
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "embedding": args.embedding, "prompt": prompt,
        "placeholder": placeholder, "num_vectors": num_vectors, "resolution": args.resolution, "steps": args.steps,
        "seed": args.seed, "records": records,
        "pipeline_load_sec": round(loaded-start,2),
        "generation_elapsed_sec": round(time.time()-gen_start,2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated()/(1024**2),1) if torch.cuda.is_available() else None,
    }, indent=2))
    print(f"[PixArt-TI-gen] wrote {out}")


if __name__ == "__main__":
    main()
