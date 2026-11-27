#!/usr/bin/env python3
"""Generate 2K data with a Qwen-Image-2.1 textual-inversion token.

Two modes, both driven by the *same* canonical ``imagenet_templates_small`` prompt that the
training script recorded in its manifest:

* ``--mode t2i``      - plain 2K text-to-image from the learned token (KV cache enabled, exactly
                        like ``QwenImage21Pipeline.__call__``); with ``--mask-image`` the sample
                        is composited through the mask onto ``--image`` to produce paired
                        image/mask training data.
* ``--mode inpaint``  - masked flow-matching sampling: at every step the known region is
                        re-noised with the clean latents (blended latent inpainting), so the
                        result is a 2K inpainting consistent with the real image outside the mask.

The prompt encoder is imported from ``highres/textual_inversion_qwen21.py`` so that the
conditioning is byte-identical to training (same template, same token splicing, same
``_drop_idx`` handling).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highres.prompt_templates import canonical_prompt  # noqa: E402
from highres.textual_inversion_qwen21 import (  # noqa: E402
    encode_prompt_embeds,
    lfp_v3_mask_qwen,
    pack_latents,
    transformer_velocity,
    unpack_latents,
    vae_latents,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def load_mask(path: str, size: int, vae, project: str) -> Image.Image:
    mask = Image.open(path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    mask = Image.fromarray(((np.asarray(mask) > 127).astype(np.uint8) * 255), mode="L")
    if project == "lfp_v3":
        mask = lfp_v3_mask_qwen(mask, vae, vae_scale=16, cache_key=path)
    return mask


def feather(mask: Image.Image, radius: int = 16) -> torch.Tensor:
    m = torch.from_numpy(np.asarray(mask).astype(np.float32) / 255.0)[None, None]
    if radius <= 0:
        return m
    kernel = 2 * radius + 1
    box = torch.ones(1, 1, kernel, kernel) / (kernel * kernel)
    for _ in range(2):  # two box passes ~ gaussian
        m = torch.nn.functional.conv2d(
            torch.nn.functional.pad(m, (radius,) * 4, mode="replicate"), box
        )
    return m.clamp(0, 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    p.add_argument("--embedding", required=True, help="qwen21_token_embedding.pt")
    p.add_argument("--mode", default="t2i", choices=["t2i", "inpaint"])
    p.add_argument("--image", default="", help="real 2K-ready image (inpaint / composite source)")
    p.add_argument("--mask-image", default="", help="defect mask for inpaint / composite")
    p.add_argument("--mask-project", default="lfp_v3", choices=["raw", "lfp_v3"])
    p.add_argument("--height", type=int, default=2048)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--template-index", type=int, default=None,
                   help="override the manifest's template index")
    p.add_argument("--output_dir", default="highres_runs/qwen21_generate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-vae-tiling", dest="vae_tiling", action="store_false", default=True)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = p.parse_args()

    from diffusers import QwenImage21Pipeline
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (
        calculate_shift,
        retrieve_timesteps,
    )

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    ckpt = torch.load(args.embedding, map_location="cpu")
    vec = ckpt["vec"].float().to(args.device)
    n_vec = int(ckpt["num_vectors"])
    placeholder = ckpt["placeholder"]
    template_index = args.template_index if args.template_index is not None else int(
        ckpt.get("template_index", 0)
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    pipe.set_progress_bar_config(disable=False)
    if args.vae_tiling:
        # a 2048x2048 decode does not fit untiled (measured OOM at ~53 GiB on a shared B300 GPU)
        pipe.vae.enable_tiling()
    tokenizer = pipe.processor.tokenizer
    tokenizer.add_tokens([placeholder])
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    prompt = canonical_prompt(placeholder, template_index)
    print(f"[gen-qwen21] prompt={prompt!r} n_vec={n_vec} template_index={template_index}",
          flush=True)

    with torch.no_grad():
        prompt_embeds, prompt_embeds_mask = encode_prompt_embeds(
            pipe.text_encoder, pipe.processor, pipe, prompt, args.device, placeholder,
            placeholder_id, vec, n_vec,
        )
    print(f"[gen-qwen21] prompt_embeds={tuple(prompt_embeds.shape)} "
          f"dtype={prompt_embeds.dtype}", flush=True)

    h_lat = args.height // pipe.vae_scale_factor
    w_lat = args.width // pipe.vae_scale_factor
    latent_hw = (h_lat, w_lat)

    real_img = None
    mask_pil = None
    if args.mask_image:
        mask_pil = load_mask(args.mask_image, args.height, pipe.vae, args.mask_project)
    if args.image:
        real_img = Image.open(args.image).convert("RGB").resize(
            (args.width, args.height), Image.Resampling.BICUBIC
        )

    if args.mode == "t2i":
        for i in range(args.num_images):
            g = torch.Generator(device=args.device).manual_seed(args.seed + i)
            image = pipe(
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                generator=g,
                output_type="pil",
                use_kv_cache=True,
            ).images[0]
            # the Qwen VAE is RGBA; ours is trained with an opaque alpha channel
            image = image.convert("RGB") if image.mode != "RGB" else image
            stem = out_dir / f"qwen21_t2i_{i:03d}"
            image.save(f"{stem}.png")
            print(f"[gen-qwen21] wrote {stem}.png {image.size}", flush=True)
            if mask_pil is not None and real_img is not None:
                a = feather(mask_pil, 16)
                gen = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
                real = torch.from_numpy(np.asarray(real_img).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
                comp = (gen * a + real * (1 - a))[0].clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray((comp * 255).round().astype(np.uint8)).save(f"{stem}_masked.png")
                mask_pil.save(f"{stem}_mask.png")
                print(f"[gen-qwen21] wrote {stem}_masked.png / _mask.png", flush=True)
        return

    # ---- masked (inpainting) sampling ----
    if real_img is None or mask_pil is None:
        raise SystemExit("--mode inpaint needs --image and --mask-image")
    x0 = vae_latents(pipe.vae, torch.cat([
        torch.from_numpy(np.asarray(real_img).astype(np.float32) / 255.0).permute(2, 0, 1)[None] * 2 - 1,
        torch.ones(1, 1, args.height, args.width),
    ], dim=1).to(args.device, dtype))  # (1, 64, h, w)
    x0_packed = pack_latents(x0)
    m_lat = torch.nn.functional.interpolate(
        torch.from_numpy(np.asarray(mask_pil).astype(np.float32) / 255.0)[None, None].to(x0.device),
        size=latent_hw, mode="nearest",
    )  # (1, 1, h, w)
    m_lat = (m_lat > 0.5).to(x0.dtype)
    m_flat = m_lat.reshape(1, -1, 1)  # (1, h*w, 1)

    num_latent = h_lat * w_lat
    mu = calculate_shift(
        num_latent,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 8192),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 0.9),
    )
    sigmas = np.linspace(1.0, 1 / args.steps, args.steps)
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler, args.steps, torch.device(args.device), sigmas=sigmas, mu=mu
    )
    pipe.scheduler.set_begin_index(0)
    for i in range(args.num_images):
        g = torch.Generator(device=args.device).manual_seed(args.seed + i)
        noise = torch.randn(x0_packed.shape, generator=g, device=args.device, dtype=dtype)
        x = noise.clone()
        # no autograd here: the transformer is frozen and the loop would otherwise keep 32 layers
        # of graph per step (measured 74 GiB -> OOM before this was wrapped)
        with torch.no_grad():
            for step, t in enumerate(timesteps):
                s = float(pipe.scheduler.sigmas[step])
                x_in = m_flat * x + (1 - m_flat) * ((1 - s) * x0_packed + s * noise)
                pred = transformer_velocity(
                    pipe.transformer, x_in, torch.tensor([s], device=args.device),
                    prompt_embeds, prompt_embeds_mask, *latent_hw,
                )
                x = pipe.scheduler.step(pred, t, x, return_dict=False)[0]
                s_next = float(pipe.scheduler.sigmas[step + 1])
                x = m_flat * x + (1 - m_flat) * ((1 - s_next) * x0_packed + s_next * noise)
                if step % 10 == 0 or step == len(timesteps) - 1:
                    print(f"[gen-qwen21] inpaint step {step + 1}/{len(timesteps)} "
                          f"sigma={s:.3f} peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GiB",
                          flush=True)
        with torch.no_grad():
            lat = unpack_latents(x, *latent_hw).unsqueeze(2).to(pipe.vae.dtype)
            stat = (1, pipe.vae.config.z_dim, 1, 1, 1)
            mean = torch.tensor(pipe.vae.config.latents_mean, device=lat.device,
                                dtype=lat.dtype).view(stat)
            std = torch.tensor(pipe.vae.config.latents_std, device=lat.device,
                               dtype=lat.dtype).view(stat)
            image = pipe.vae.decode(lat * std + mean, return_dict=False)[0][:, :, 0]
        image = pipe.image_processor.postprocess(image, output_type="pil")[0]
        image = image.convert("RGB") if image.mode != "RGB" else image
        stem = out_dir / f"qwen21_inpaint_{i:03d}"
        image.save(f"{stem}.png")
        # keep the real pixels outside the LFP mask so the pair is exactly aligned
        a = feather(mask_pil, 16)
        gen = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        real = torch.from_numpy(np.asarray(real_img).astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        comp = (gen * a + real * (1 - a))[0].clamp(0, 1).permute(1, 2, 0).numpy()
        Image.fromarray((comp * 255).round().astype(np.uint8)).save(f"{stem}_masked.png")
        mask_pil.save(f"{stem}_mask.png")
        print(f"[gen-qwen21] wrote {stem}.png (+ _masked/_mask) {image.size}", flush=True)

    (out_dir / "gen_manifest.json").write_text(json.dumps({
        "model": args.model,
        "embedding": args.embedding,
        "prompt": prompt,
        "template_index": template_index,
        "num_vectors": n_vec,
        "mode": args.mode,
        "height": args.height,
        "width": args.width,
        "steps": args.steps,
        "seed": args.seed,
        "image": args.image,
        "mask_image": args.mask_image,
        "mask_project": args.mask_project,
        "sigma_mu": mu,
    }, indent=2))
    print(f"[gen-qwen21] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
