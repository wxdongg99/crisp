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
    mask_pil = make_mask(args.resolution, args.mask_box)
    source_pil.resize((args.resolution, args.resolution)).save(out / "source.png")
    mask_pil.save(out / "mask.png")

    start = time.time()
    pipe = PixArtSigmaPipeline.from_pretrained(args.model, torch_dtype=dtype)
    hooks = []
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
            for _, idx in (ids == tid).nonzero(as_tuple=False):
                o[0, idx] = vec.to(o.dtype)
            return o
        hooks.append(pipe.text_encoder.encoder.embed_tokens.register_forward_hook(hook, with_kwargs=True))
    if args.enable_sequential_offload:
        pipe.enable_sequential_cpu_offload()
    else:
        t5_device = torch.device(args.t5_device)
        pipe.text_encoder.to(t5_device)
        if args.t5_device == "cpu":
            pipe.text_encoder.to(torch.float32)
        pipe.transformer.to(torch.device(args.tx_device))
        pipe.vae.to(torch.device(args.vae_device))
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
    with torch.no_grad():
        prompt_embeds, prompt_attn, neg_embeds, neg_attn = pipe.encode_prompt(
            args.prompt, do_classifier_free_guidance=do_cfg, negative_prompt=args.negative_prompt,
            num_images_per_prompt=1, device=t5_device, clean_caption=False)
    if do_cfg:
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
        arr = np.asarray(image.resize((args.resolution, args.resolution))).astype(np.float32)
        bg = np.abs(arr[~mask_np] - src_np[~mask_np]).mean()
        records.append({"file": str(img_dir / name), "seed": seed,
                        "background_mae_outside_mask": round(float(bg), 3), "output_size": list(image.size)})
        print(f"[PixArt-mask] {i+1}/{args.num_images} bg_mae={bg:.2f}")
    for h in hooks:
        h.remove()
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "image": args.image, "mask_box": args.mask_box,
        "prompt": args.prompt, "negative_prompt": args.negative_prompt,
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
