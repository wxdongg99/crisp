#!/usr/bin/env python3
"""Batch PixArt-1024 inpainting dataset generator; saves 512-sized eval data."""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--embedding", required=True)
    p.add_argument("--source-dir", required=True)
    p.add_argument("--mask-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pattern", required=True)
    p.add_argument("--placeholder", default="")
    p.add_argument("--template-index", type=int, default=0)
    p.add_argument("--prompt", default="", help="fixed prompt; empty = canonical imagenet template")
    p.add_argument("--count", type=int, default=500)
    p.add_argument("--native-resolution", type=int, default=1024)
    p.add_argument("--output-resolution", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.9)
    p.add_argument("--cfg", type=float, default=4.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mask-project", default="lfp_v3", choices=["raw", "lfp_v3"])
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--t5-device", default="cuda")
    p.add_argument("--enable-attention-slicing", action="store_true")
    args = p.parse_args()

    from diffusers import PixArtSigmaPipeline
    from transformers import T5Tokenizer
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device("cuda")
    out = Path(args.output_dir) / "fabric" / args.pattern
    (out / "image").mkdir(parents=True, exist_ok=True)
    (out / "mask").mkdir(parents=True, exist_ok=True)
    srcs = sorted([x for x in Path(args.source_dir).glob("*") if x.is_file()])
    masks = sorted([x for x in Path(args.mask_dir).glob("*") if x.is_file()])
    if not srcs or not masks:
        raise RuntimeError("empty source/mask list")
    rng = random.Random(args.seed)

    ckpt = torch.load(args.embedding, map_location="cpu")
    placeholder = ckpt["placeholder"]; vec = ckpt["vec"].to(dtype)
    if args.placeholder:
        placeholder = args.placeholder
    if not args.prompt:
        from highres.prompt_templates import canonical_prompt
        args.prompt = canonical_prompt(placeholder, args.template_index)
    tok = T5Tokenizer.from_pretrained(str(Path(args.model) / "tokenizer"), use_fast=False, legacy=True)
    pipe = PixArtSigmaPipeline.from_pretrained(args.model, torch_dtype=dtype, tokenizer=tok)
    pipe.tokenizer.add_tokens([placeholder]); pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    tid = pipe.tokenizer.convert_tokens_to_ids(placeholder)
    pipe.text_encoder.to(device); pipe.transformer.to(device); pipe.vae.to(device)
    pipe.transformer.requires_grad_(False); pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False)
    if args.enable_attention_slicing: pipe.enable_attention_slicing()
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.transformer.set_attn_processor(AttnProcessor2_0())
    torch.set_grad_enabled(False)

    def encode_text(text, with_token):
        t = pipe.tokenizer([text], padding="max_length", max_length=min(120, pipe.tokenizer.model_max_length),
                           truncation=True, return_tensors="pt")
        ids = t.input_ids.to(pipe.text_encoder.device); attn = t.attention_mask.to(pipe.text_encoder.device)
        emb = pipe.text_encoder.get_input_embeddings()(ids).detach()
        if with_token:
            for b in range(emb.shape[0]):
                pos = (ids[b] == tid).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0: continue
                p0 = int(pos[0]); v = vec.to(device=emb.device, dtype=emb.dtype)
                emb[b] = torch.cat([emb[b, :p0], v, emb[b, p0 + 1:]], dim=0)[:emb.shape[1]]
                attn[b] = torch.cat([attn[b, :p0], torch.ones(v.shape[0], device=attn.device, dtype=attn.dtype), attn[b, p0 + 1:]], dim=0)[:attn.shape[1]]
        h = pipe.text_encoder(inputs_embeds=emb, attention_mask=attn).last_hidden_state
        return h, attn
    pos, pm = encode_text(args.prompt, True)
    neg, nm = encode_text("", False)
    prompt_embeds = torch.cat([neg, pos], 0).to(device, dtype); prompt_attn = torch.cat([nm, pm], 0).to(device)
    pipe.text_encoder = None
    import gc; gc.collect(); torch.cuda.empty_cache()

    scheduler = pipe.scheduler
    t0 = time.time()
    for i in range(args.count):
        src = Image.open(srcs[i % len(srcs)]).convert("RGB").resize((args.native_resolution, args.native_resolution), Image.Resampling.BICUBIC)
        mask_path = masks[(i * 37 + args.seed) % len(masks)]
        mask = Image.open(mask_path).convert("L").resize((args.native_resolution, args.native_resolution), Image.Resampling.NEAREST)
        if args.mask_project == "lfp_v3":
            from highres.lfp_mask import lfp_pixel_mask_v3_cached
            mask = lfp_pixel_mask_v3_cached(mask, pipe.vae, float(pipe.vae.config.scaling_factor), cache_key=str(mask_path))
        src_t = transforms.ToTensor()(src).unsqueeze(0).to(device, dtype) * 2 - 1
        with torch.no_grad():
            src_lat = pipe.vae.encode(src_t).latent_dist.sample() * pipe.vae.config.scaling_factor
        mask_lat = F.interpolate(transforms.ToTensor()(mask).unsqueeze(0).to(device, dtype),
                                 size=src_lat.shape[-2:], mode="nearest")
        mask_lat = (mask_lat > 0.5).to(dtype)
        scheduler.set_timesteps(args.steps)
        timesteps = scheduler.timesteps
        start = min(int(args.steps * (1.0 - args.strength)), len(timesteps) - 1)
        timesteps = timesteps[start:]
        gen = torch.Generator(device="cpu").manual_seed(args.seed + i)
        noise = torch.randn(src_lat.shape, generator=gen, device="cpu").to(device, dtype)
        latents = scheduler.add_noise(src_lat, noise, timesteps[0].reshape(1).to(device))
        for j, ts in enumerate(timesteps):
            inp = scheduler.scale_model_input(torch.cat([latents] * 2), ts)
            tb = ts.reshape(1).to(device).expand(inp.shape[0])
            pred = pipe.transformer(inp, encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_attn,
                                    timestep=tb, added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                                    return_dict=False)[0]
            pu, pt = pred.chunk(2); pred = pu + args.cfg * (pt - pu); pred = pred[:, :4]
            latents = scheduler.step(pred, ts, latents, return_dict=False)[0]
            bg = scheduler.add_noise(src_lat, noise, timesteps[j + 1].reshape(1).to(device)) if j < len(timesteps) - 1 else src_lat
            latents = (1 - mask_lat) * bg + mask_lat * latents
        image = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(device)).sample
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image512 = image.resize((args.output_resolution, args.output_resolution), Image.Resampling.LANCZOS)
        mask512 = mask.resize((args.output_resolution, args.output_resolution), Image.Resampling.NEAREST)
        image512.save(out / "image" / f"{i}.jpg", quality=95)
        mask512.save(out / "mask" / f"{i}.jpg", quality=95)
        if (i + 1) % 25 == 0:
            print(f"[gen-pixart] {args.pattern} {i+1}/{args.count} elapsed={time.time()-t0:.1f}s", flush=True)
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "embedding": args.embedding, "pattern": args.pattern,
        "prompt": args.prompt, "count": args.count, "native_resolution": args.native_resolution,
        "output_resolution": args.output_resolution, "steps": args.steps, "strength": args.strength,
        "cfg": args.cfg, "seed": args.seed, "mask_project": args.mask_project, "elapsed_sec": round(time.time()-t0, 2),
        "peak_vram_mb": round(torch.cuda.max_memory_allocated()/1024**2, 1),
    }, indent=2))
    print("[gen-pixart] done", out, flush=True)

if __name__ == "__main__":
    main()
