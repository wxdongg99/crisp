#!/usr/bin/env python3
"""Generate with a FLUX.1-dev textual-inversion token, with 2K high-res fix.

Stage 1 (1024): mask-conditioned FLUX inpainting through diffusers
`FluxInpaintPipeline` (official 0.36 pipeline; FLUX's transformer has 64 input
channels, so inpainting is implemented as masked img2img: at every step the
known region is replaced by the noised source latent
`latents = (1-m)*noised_source + m*latents`).

Stage 2 (2K high-res fix) -- default `pixel_upscale_img2img`:
  1024 output -> bicubic upscale to 2048 -> `FluxInpaintPipeline` at 2048x2048
  with the same prompt/token vectors and the upscaled defect mask, denoising only
  `--hires-strength` of the schedule (default 0.35) -> composite the refined
  defect pixels back over the bicubic-upscaled source using the *raw* mask.
  Rationale: FLUX has no native 2K prior. Its 16-channel VAE latent at 1024 is
  128x128; bicubic latent upsampling adds no detail and the refine pass would
  first have to undo interpolation blur, so the pixel-upscale path is used.
  `--hires-mode latent_upscale_img2img` implements the latent path explicitly
  (unpack free: 1024 latent -> bicubic 2x -> repack at 2K -> partial sigma
  denoise) for comparison. For >2K, `--tile-size` refines the high-res pass in
  overlapping tiles (experimental).

The prompt is the same canonical `imagenet_templates_small` text used by the
training script (highres/prompt_templates.py). Token vectors are spliced into
*both* CLIP-L and T5-XXL by the shared `FluxTextEncoders` class; the embeddings
are precomputed and passed to the pipeline, so the training and generation paths
use identical attention-mask handling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# See highres/textual_inversion_flux.py: the B300 venv leaks another env's
# sentencepiece into sys.path, which breaks the slow->fast T5 tokenizer
# conversion unless protobuf uses its pure-python implementation.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

if torch.cuda.is_available():
    # Same B300 issue as in highres/textual_inversion_flux.py: cuDNN's fused MHA
    # backend raises "mha_graph->execute(...).is_good()" and illegal memory
    # access. Use flash / memory-efficient SDPA only.
    torch.backends.cuda.enable_cudnn_sdp(False)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_image(path, size, crop=True):
    tf = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)]
    if crop:
        tf.append(transforms.CenterCrop(size))
    return transforms.Compose(tf)(Image.open(path).convert("RGB"))


def make_box_mask(size, box):
    m = Image.new("L", (size, size), 0)
    x0, y0, x1, y1 = [int(round(v * size)) for v in box]
    m.paste(255, (x0, y0, x1, y1))
    return m


def resize_mask(mask, size):
    return mask.convert("L").resize((size, size), Image.Resampling.NEAREST)


def tile_windows(h, w, tile, overlap):
    step = max(tile - overlap, 1)
    ys = list(range(0, max(h - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(w - tile, 0) + 1, step)) or [0]
    if ys[-1] != max(h - tile, 0):
        ys.append(max(h - tile, 0))
    if xs[-1] != max(w - tile, 0):
        xs.append(max(w - tile, 0))
    return [(y, x) for y in ys for x in xs]


def bg_mae(generated, source, mask_np):
    """MAE outside the mask: how much of the non-defect background moved."""
    a = np.asarray(generated.convert("RGB"), dtype=np.float32)
    b = np.asarray(source.convert("RGB"), dtype=np.float32)
    keep = ~mask_np
    return round(float(np.abs(a[keep] - b[keep]).mean()) if keep.any() else 0.0, 3)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/FLUX.1-dev")
    p.add_argument("--embedding", required=True, help="flux_token_embedding.pt from TI")
    p.add_argument("--image", default="", help="source image; empty = generation on grey")
    p.add_argument("--mask-image", default="", help="raw defect mask (else --mask-box)")
    p.add_argument("--mask-box", nargs=4, type=float, default=[0.35, 0.35, 0.65, 0.65])
    p.add_argument("--prompt", default="", help="empty = canonical template prompt")
    p.add_argument("--template-index", type=int, default=-1, help="-1 = from checkpoint")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--guidance", type=float, default=-1.0, help="-1 = from checkpoint")
    p.add_argument("--mask-project", default="", help="raw|lfp_v3|lfp_v3_f16 (default: from ckpt)")
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--hires", type=int, default=0, help="0 = off, e.g. 2048")
    p.add_argument("--hires-mode", default="pixel_upscale_img2img",
                   choices=["pixel_upscale_img2img", "latent_upscale_img2img"])
    p.add_argument("--hires-steps", type=int, default=12)
    p.add_argument("--hires-strength", type=float, default=0.35)
    p.add_argument("--hires-guidance", type=float, default=-1.0)
    p.add_argument("--tile-size", type=int, default=0, help="refine high-res pass in tiles")
    p.add_argument("--tile-overlap", type=int, default=128)
    p.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false")
    p.add_argument("--output-dir", default="highres_runs/flux_token")
    p.add_argument("--run-name", default="")
    args = p.parse_args()

    from diffusers import FluxInpaintPipeline
    from highres.prompt_templates import canonical_prompt
    from highres.textual_inversion_flux import FluxTextEncoders, lfp_v3_f16
    from highres.lfp_mask import lfp_pixel_mask_v3

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ckpt = torch.load(args.embedding, map_location="cpu")
    placeholder = ckpt["placeholder"]
    n_vec = int(ckpt.get("num_vectors", ckpt["vec_clip"].shape[0]))
    vec_clip = ckpt["vec_clip"].to("cuda", dtype)
    vec_t5 = ckpt["vec_t5"].to("cuda", dtype)
    template_index = args.template_index if args.template_index >= 0 else int(ckpt.get("template_index", 0))
    guidance = args.guidance if args.guidance >= 0 else float(ckpt.get("guidance", 3.5))
    mask_project = args.mask_project or ckpt.get("mask_project", "raw")
    attn_mode = ckpt.get("attn_mode", "expanded")
    max_seq = int(ckpt.get("max_sequence_length", 512))

    out = Path(args.output_dir) / (args.run_name or ("flux_token_" + Path(args.embedding).parent.name))
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    print(f"[FLUX-gen] model={args.model} ckpt={args.embedding} n_vec={n_vec} "
          f"attn_mode={attn_mode} mask_project={mask_project}", flush=True)
    pipe = FluxInpaintPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe.tokenizer.add_tokens([placeholder])
    pipe.tokenizer_2.add_tokens([placeholder])
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    pipe.text_encoder_2.resize_token_embeddings(len(pipe.tokenizer_2))
    pipe = pipe.to("cuda")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    enc = FluxTextEncoders(pipe.text_encoder, pipe.text_encoder_2, pipe.tokenizer,
                           pipe.tokenizer_2, placeholder, attn_mode=attn_mode,
                           max_sequence_length=max_seq, device="cuda")

    prompt = args.prompt or canonical_prompt(placeholder, template_index)
    print(f"[FLUX-gen] prompt={prompt!r}", flush=True)
    with torch.no_grad():
        prompt_embeds, pooled, text_ids = enc(prompt, vec_clip, vec_t5)
    prompt_embeds = prompt_embeds.to(dtype)
    pooled = pooled.to(dtype)
    del enc, vec_clip, vec_t5
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # ---- source image + masks -------------------------------------------
    res = args.resolution
    source = load_image(args.image, res) if args.image else Image.new("RGB", (res, res), (128, 128, 128))
    raw_mask = resize_mask(Image.open(args.mask_image), res) if args.mask_image else make_box_mask(res, args.mask_box)
    source.save(out / "source_1024.png")
    raw_mask.save(out / "mask_raw_1024.png")

    if mask_project in ("lfp_v3", "lfp_v3_f16"):
        vae = pipe.vae
        vae_dev = next(vae.parameters()).device
        if vae_dev.type != "cpu":
            vae.to("cpu")
        if mask_project == "lfp_v3":
            mask = lfp_pixel_mask_v3(raw_mask, vae, vae.config.scaling_factor, pipe.vae_scale_factor)
        else:
            mask = lfp_v3_f16(raw_mask, vae, vae.config.scaling_factor, pipe.vae_scale_factor)
        vae.to(vae_dev)
        print(f"[FLUX-gen] mask_project={mask_project} "
              f"raw_area={np.mean(np.asarray(raw_mask) > 127):.4f} "
              f"proj_area={np.mean(np.asarray(mask) > 127):.4f}", flush=True)
    else:
        mask = raw_mask
    mask.save(out / "mask_projected_1024.png")
    mask_np = np.asarray(raw_mask) > 127

    loaded = time.time()
    gen_start = time.time()
    records = []
    for i in range(args.num_images):
        seed = args.seed + i
        t0 = time.time()
        image = pipe(prompt=None, prompt_2=None, prompt_embeds=prompt_embeds,
                     pooled_prompt_embeds=pooled, image=source, mask_image=mask,
                     height=res, width=res, num_inference_steps=args.steps,
                     strength=args.strength, guidance_scale=guidance,
                     generator=torch.Generator(device="cpu").manual_seed(seed),
                     output_type="pil").images[0]
        s1 = time.time() - t0
        path1 = img_dir / f"p00_{i:04d}_1024.png"
        image.save(path1)
        rec = {"seed": seed, "stage1_1024": str(path1), "stage1_sec": round(s1, 2),
               "bg_mae_outside_mask_1024": bg_mae(image, source, mask_np)}
        final = image

        if args.hires:
            hi = args.hires
            t1 = time.time()
            up = source.resize((hi, hi), Image.Resampling.BICUBIC)
            up_mask = resize_mask(mask, hi)
            up_mask_raw = resize_mask(raw_mask, hi)
            if args.hires_mode == "latent_upscale_img2img":
                refined = _latent_refine(pipe, image, hi, prompt_embeds, pooled, text_ids,
                                         guidance, args.hires_steps, args.hires_strength, seed)
            elif args.tile_size and 512 <= args.tile_size < hi:
                refined, n_tiles = _tiled_refine(pipe, up, up_mask, prompt_embeds, pooled,
                                                 guidance, args.hires_steps,
                                                 args.hires_strength, args.tile_size,
                                                 args.tile_overlap, seed)
                rec["hires_tiles"] = n_tiles
            else:
                refined = pipe(prompt=None, prompt_2=None, prompt_embeds=prompt_embeds,
                               pooled_prompt_embeds=pooled, image=up, mask_image=up_mask,
                               height=hi, width=hi, num_inference_steps=args.hires_steps,
                               strength=args.hires_strength,
                               guidance_scale=(args.hires_guidance if args.hires_guidance >= 0
                                               else guidance),
                               generator=torch.Generator(device="cpu").manual_seed(seed),
                               output_type="pil").images[0]
            # composite only the defect region over the bicubic-upscaled source
            comp = np.asarray(up).astype(np.float32).copy()
            ref = np.asarray(refined.convert("RGB")).astype(np.float32)
            m_hi = np.asarray(up_mask_raw) > 127
            comp[m_hi] = ref[m_hi]
            final = Image.fromarray(comp.astype(np.uint8))
            path2 = img_dir / f"p00_{i:04d}_{hi}_composite.png"
            final.save(path2)
            refined.save(img_dir / f"p00_{i:04d}_{hi}_refine_raw.png")
            rec.update({"stage2_hires": hi, "hires_mode": args.hires_mode,
                        "hires_strength": args.hires_strength,
                        "hires_sec": round(time.time() - t1, 2),
                        "final": str(path2),
                        "final_size": list(final.size)})
        records.append(rec)
        print(f"[FLUX-gen] {i+1}/{args.num_images} stage1={rec['stage1_sec']}s "
              f"stage2={rec.get('hires_sec', 0)}s bg_mae={rec['bg_mae_outside_mask_1024']}",
              flush=True)

    manifest = {
        "model": args.model, "embedding": args.embedding, "prompt": prompt,
        "placeholder": placeholder, "num_vectors": n_vec, "attn_mode": attn_mode,
        "mask_project": mask_project, "resolution": res, "steps": args.steps,
        "strength": args.strength, "guidance": guidance, "seed": args.seed,
        "hires": args.hires, "hires_mode": args.hires_mode if args.hires else None,
        "hires_steps": args.hires_steps if args.hires else None,
        "hires_strength": args.hires_strength if args.hires else None,
        "tile_size": args.tile_size,
        "source": args.image, "mask_image": args.mask_image, "mask_box": args.mask_box,
        "records": records,
        "load_sec": round(loaded - start, 2),
        "generation_sec": round(time.time() - gen_start, 2),
        "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[FLUX-gen] wrote {out}", flush=True)


def _latent_refine(pipe, image, size, prompt_embeds, pooled, text_ids,
                   guidance, steps, strength, seed):
    """Latent-upscale refine: encode 1024 output, bicubic 2x the latent, partial denoise."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from highres.textual_inversion_flux import flux_calculate_shift
    dtype = prompt_embeds.dtype
    lat_small = size // 16  # 2048 -> 128 latent tokens per side
    with torch.no_grad():
        img_t = transforms.ToTensor()(image).unsqueeze(0).to("cuda", pipe.vae.dtype) * 2 - 1
        z = pipe.vae.encode(img_t).latent_dist.mode()
    z = (z.float() - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    z_hi = F.interpolate(z.float(), size=(lat_small * 2, lat_small * 2), mode="bicubic")
    z_packed = pipe._pack_latents(z_hi.to(dtype), 1, 16, lat_small * 2, lat_small * 2)
    img_ids = pipe._prepare_latent_image_ids(1, lat_small, lat_small, "cuda", dtype)

    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # FLUX ships use_dynamic_shifting=True, so `mu` (resolution-dependent shift)
    # must be passed exactly as FluxPipeline.__call__ does.
    mu = (flux_calculate_shift(lat_small * lat_small, scheduler)
          if getattr(scheduler.config, "use_dynamic_shifting", False) else None)
    scheduler.set_timesteps(steps, device="cuda", mu=mu)
    ts = scheduler.timesteps
    ts = ts[max(int(len(ts) * (1.0 - strength)), 0):]
    sigma0 = float(ts[0]) / 1000.0
    noise = torch.randn_like(z_packed)
    lat = (1.0 - sigma0) * z_packed + sigma0 * noise
    guid = torch.full((1,), float(guidance), device="cuda", dtype=torch.float32)
    for t in ts:
        pred = pipe.transformer(hidden_states=lat, timestep=(t / 1000).expand(1),
                                guidance=guid, pooled_projections=pooled,
                                encoder_hidden_states=prompt_embeds, txt_ids=text_ids,
                                img_ids=img_ids, return_dict=False)[0]
        lat = scheduler.step(pred, t, lat, return_dict=False)[0]
    lat = pipe._unpack_latents(lat, size, size, pipe.vae_scale_factor)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    img = pipe.vae.decode(lat.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(img, output_type="pil")[0]


def _tiled_refine(pipe, up, up_mask, prompt_embeds, pooled, guidance,
                  steps, strength, tile, overlap, seed):
    """Refine the high-res image tile by tile; mask pixels of covered tiles are pasted."""
    w, h = up.size
    canvas = np.asarray(up).astype(np.float32).copy()
    mask_np = np.asarray(up_mask) > 127
    kept = 0
    for (y, x) in tile_windows(h, w, tile, overlap):
        tile_mask_np = mask_np[y:y + tile, x:x + tile]
        if tile_mask_np.sum() == 0:
            continue
        crop = up.crop((x, y, x + tile, y + tile))
        tm = up_mask.crop((x, y, x + tile, y + tile))
        refined = pipe(prompt=None, prompt_2=None, prompt_embeds=prompt_embeds,
                       pooled_prompt_embeds=pooled, image=crop, mask_image=tm,
                       height=tile, width=tile, num_inference_steps=steps,
                       strength=strength, guidance_scale=guidance,
                       generator=torch.Generator(device="cpu").manual_seed(seed),
                       output_type="pil").images[0]
        ref_np = np.asarray(refined.convert("RGB")).astype(np.float32)
        canvas[y:y + tile, x:x + tile][tile_mask_np] = ref_np[tile_mask_np]
        kept += 1
    return Image.fromarray(canvas.astype(np.uint8)), kept


if __name__ == "__main__":
    main()
