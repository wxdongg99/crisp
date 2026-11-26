#!/usr/bin/env python3
"""PixArt-Sigma mask probe with sampler-level background preservation."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_image(path, size):
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB"))


def make_mask(size, box):
    m = Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = [int(round(v * size)) for v in box]
    m.paste(255, (x0, y0, x1, y1))
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/pixart_sigma_1024")
    p.add_argument("--image", required=True)
    p.add_argument("--mask-box", nargs=4, type=float, default=[0.35, 0.35, 0.65, 0.65])
    p.add_argument("--mask-image", default="", help="optional real/derived mask image")
    p.add_argument("--mask-project", default="lfp_v3", choices=["raw", "lfp_v3"])
    p.add_argument("--composite-mask", default="", help="optional fine mask for final compositing")
    p.add_argument("--slow-tokenizer", action="store_true", help="use T5Tokenizer slow path")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative_prompt", default="blurry, low resolution, jpeg artifacts, text, watermark")
    p.add_argument("--embedding", default="")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--strength", type=float, default=0.9)
    p.add_argument("--guidance_scale", type=float, default=4.5)
    p.add_argument("--num_images", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="highres_runs/pixart_mask")
    p.add_argument("--run_name", default="")
    p.add_argument("--enable_cpu_offload", action="store_true")
    p.add_argument("--enable_attention_slicing", action="store_true")
    p.add_argument("--enable_sequential_offload", action="store_true")
    p.add_argument("--t5-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--attn-backend", default="sdpa", choices=["default", "sdpa", "xformers", "sliced"])
    p.add_argument("--fp8", action="store_true", help="FP8 transformer weights; requires torchao")
    p.add_argument("--tx-device", default="cuda", help="transformer device, e.g. cuda:1")
    p.add_argument("--vae-device", default="cuda", help="VAE device, e.g. cuda:0")
    args = p.parse_args()

    from diffusers import PixArtSigmaPipeline

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = Path(args.output_dir) / (args.run_name or time.strftime("pixart_mask_%Y%m%d_%H%M%S"))
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    source_pil = Image.open(args.image).convert("RGB")
    src_t = load_image(args.image, args.resolution).unsqueeze(0).to("cuda", dtype)
    if args.mask_image:
        mask_pil = Image.open(args.mask_image).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
    else:
        mask_pil = make_mask(args.resolution, args.mask_box)
    source_pil.resize((args.resolution, args.resolution)).save(out / "source.png")
    mask_pil.save(out / "mask_raw.png")

    start = time.time()
    tok_override = None
    if args.slow_tokenizer:
        from transformers import T5Tokenizer
        tok_override = T5Tokenizer.from_pretrained(str(Path(args.model) / "tokenizer"), use_fast=False)
    pipe = PixArtSigmaPipeline.from_pretrained(args.model, torch_dtype=dtype, tokenizer=tok_override)
    hooks = []
    has_token = bool(args.embedding)
    if args.embedding:
        ckpt = torch.load(args.embedding, map_location="cpu")
        placeholder = ckpt["placeholder"]
        vec = ckpt["vec"].to(dtype)
        pipe.tokenizer.add_tokens([placeholder])
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
        tid = pipe.tokenizer.convert_tokens_to_ids(placeholder)
        def hook(module, a, kw, output):
            ids = (kw or {}).get("input_ids", a[0] if a else None)
            o = output.clone()
            for b in range(o.shape[0]):
                pos = (ids[b] == tid).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                p = int(pos[0])
                v = vec.to(device=o.device, dtype=o.dtype)
                o[b] = torch.cat([o[b, :p], v, o[b, p + 1:]], dim=0)[:o.shape[1]]
            return o
        def mask_pre(module, hargs, hkwargs):
            ids = hkwargs.get("input_ids")
            mask = hkwargs.get("attention_mask")
            if ids is not None and mask is not None:
                out = mask.clone()
                for b in range(out.shape[0]):
                    pos = (ids[b] == tid).nonzero(as_tuple=False).flatten()
                    if pos.numel() == 0:
                        continue
                    p = int(pos[0])
                    out[b] = torch.cat([out[b, :p], torch.ones(vec.shape[0], device=out.device, dtype=out.dtype), out[b, p+1:]], dim=0)[:out.shape[1]]
                hkwargs["attention_mask"] = out
            return hargs, hkwargs
        hooks.append(pipe.text_encoder.encoder.embed_tokens.register_forward_hook(hook, with_kwargs=True))
        hooks.append(pipe.text_encoder.register_forward_pre_hook(mask_pre, with_kwargs=True))
    if args.enable_sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        t5_device = torch.device(args.t5_device)
        pipe.text_encoder.to(t5_device)
        if args.t5_device == "cpu":
            pipe.text_encoder.to(torch.float32)
        pipe.transformer.to(torch.device(args.tx_device))
        pipe.vae.to(torch.device(args.vae_device))
    if args.mask_project == "lfp_v3":
        from highres.lfp_mask import lfp_pixel_mask_v3_cached
        mask_pil = lfp_pixel_mask_v3_cached(mask_pil, pipe.vae, float(pipe.vae.config.scaling_factor), cache_key=args.mask_image or "box")
        mask_pil.save(out / "mask_lfp_v3.png")
    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()
    if args.fp8:
        from highres.fp8_utils import apply_fp8_inference
        apply_fp8_inference(pipe.transformer)
        print("[PixArt-mask] FP8 weight-only transformer enabled")
    if args.attn_backend != "default":
        from diffusers.models.attention_processor import AttnProcessor2_0, SlicedAttnProcessor, XFormersAttnProcessor
        if args.attn_backend == "sdpa":
            pipe.transformer.set_attn_processor(AttnProcessor2_0())
        elif args.attn_backend == "xformers":
            pipe.transformer.set_attn_processor(XFormersAttnProcessor())
        else:
            pipe.transformer.set_attn_processor(SlicedAttnProcessor(slice_size=1))
    pipe.transformer.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    torch.set_grad_enabled(False)
    device = torch.device(args.tx_device)
    vae_device = torch.device(args.vae_device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    loaded = time.time()

    do_cfg = args.guidance_scale > 1.0
    t5_device = torch.device(args.t5_device)

    def encode_text(text, with_token):
        tok = pipe.tokenizer([text], padding="max_length", max_length=min(120, pipe.tokenizer.model_max_length),
                             truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(t5_device)
        attn = tok.attention_mask.to(t5_device)
        emb = pipe.text_encoder.get_input_embeddings()(ids).detach()
        if with_token:
            for b in range(emb.shape[0]):
                pos = (ids[b] == tid).nonzero(as_tuple=False).flatten()
                if pos.numel() == 0:
                    continue
                p0 = int(pos[0])
                v = vec.to(device=emb.device, dtype=emb.dtype)
                emb[b] = torch.cat([emb[b, :p0], v, emb[b, p0 + 1:]], dim=0)[:emb.shape[1]]
                attn[b] = torch.cat([attn[b, :p0], torch.ones(v.shape[0], device=attn.device, dtype=attn.dtype), attn[b, p0 + 1:]], dim=0)[:attn.shape[1]]
        h = pipe.text_encoder(inputs_embeds=emb, attention_mask=attn).last_hidden_state
        return h, attn
    with torch.no_grad():
        prompt_embeds, prompt_attn = encode_text(args.prompt, has_token)
        if do_cfg:
            neg_embeds, neg_attn = encode_text(args.negative_prompt, False)
            prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            prompt_attn = torch.cat([neg_attn, prompt_attn], dim=0)
    prompt_embeds = prompt_embeds.to(device, dtype)
    prompt_attn = prompt_attn.to(device)
    import gc
    try:
        pipe.components.pop("text_encoder", None)
    except Exception:
        pass
    pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[PixArt-mask] post-T5 vram={torch.cuda.memory_allocated()/1024**2:.0f}MB")

    scheduler = pipe.scheduler
    scheduler.set_timesteps(args.steps)
    timesteps = scheduler.timesteps
    start_idx = min(int(args.steps * (1.0 - args.strength)), len(timesteps) - 1)
    timesteps = timesteps[start_idx:]
    with torch.no_grad():
        src_lat = pipe.vae.encode(src_t.to(vae_device)).latent_dist.sample() * pipe.vae.config.scaling_factor
    src_lat = src_lat.to(device)
    mask_lat = F.interpolate(
        transforms.ToTensor()(mask_pil).unsqueeze(0).to(device, dtype),
        size=(src_lat.shape[-2], src_lat.shape[-1]), mode="nearest")
    mask_lat = (mask_lat > 0.5).to(dtype)

    gen_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    records = []
    mask_np = np.asarray(mask_pil) > 127
    src_np = np.asarray(source_pil.resize((args.resolution, args.resolution))).astype(np.float32)
    for i in range(args.num_images):
        # Reset the scheduler state for each independent image.
        scheduler.set_timesteps(args.steps)
        timesteps = scheduler.timesteps
        start_idx = min(int(args.steps * (1.0 - args.strength)), len(timesteps) - 1)
        timesteps = timesteps[start_idx:]
        seed = args.seed + i
        gen = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(src_lat.shape, generator=gen, device="cpu").to(device, dtype)
        t0 = timesteps[0]
        latents = scheduler.add_noise(src_lat, noise, t0.reshape(1).to(device))
        for j, t in enumerate(timesteps):
            inp = (torch.cat([latents] * 2) if do_cfg else latents).to(device, dtype)
            inp = scheduler.scale_model_input(inp, t)
            t_batch = t.reshape(1).to(device).expand(inp.shape[0])
            pred = pipe.transformer(inp, encoder_hidden_states=prompt_embeds,
                                    encoder_attention_mask=prompt_attn, timestep=t_batch,
                                    added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                                    return_dict=False)[0]
            if do_cfg:
                pu, pt = pred.chunk(2)
                pred = pu + args.guidance_scale * (pt - pu)
            pred = pred[:, :4]
            latents = scheduler.step(pred, t, latents, return_dict=False)[0]
            if j < len(timesteps) - 1:
                bg = scheduler.add_noise(src_lat, noise, timesteps[j + 1].reshape(1).to(device))
            else:
                bg = src_lat
            latents = (1 - mask_lat) * bg + mask_lat * latents
        with torch.no_grad():
            image = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(vae_device)).sample
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        name = f"p00_{i:04d}.png"
        image.save(img_dir / name)
        composite_path = ""
        if args.composite_mask:
            cm = Image.open(args.composite_mask).convert("L").resize((args.resolution, args.resolution), Image.Resampling.NEAREST)
            comp = Image.composite(image.resize((args.resolution, args.resolution)), source_pil.resize((args.resolution, args.resolution)), cm)
            composite_path = str(out / f"composite_{i:04d}.png")
            comp.save(composite_path)
        arr = np.asarray(image.resize((args.resolution, args.resolution))).astype(np.float32)
        bg = np.abs(arr[~mask_np] - src_np[~mask_np]).mean()
        records.append({"file": str(img_dir / name), "seed": seed,
                        "background_mae_outside_mask": round(float(bg), 3), "output_size": list(image.size),
                        "composite": composite_path})
        print(f"[PixArt-mask] {i+1}/{args.num_images} bg_mae={bg:.2f}")
    for h in hooks:
        h.remove()
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "image": args.image, "mask_box": args.mask_box,
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
        "mask_image": args.mask_image, "composite_mask": args.composite_mask,
        "embedding": args.embedding, "resolution": args.resolution,
        "steps": args.steps, "strength": args.strength, "seed": args.seed,
        "records": records,
        "pipeline_load_sec": round(loaded-start,2),
        "generation_elapsed_sec": round(time.time()-gen_start,2),
        "run_elapsed_sec": round(time.time()-start,2),
        "attn_backend": args.attn_backend,
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated()/(1024**2),1) if torch.cuda.is_available() else None,
        "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved()/(1024**2),1) if torch.cuda.is_available() else None,
    }, indent=2))
    print(f"[PixArt-mask] wrote {out}")


if __name__ == "__main__":
    main()
