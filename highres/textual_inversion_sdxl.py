#!/usr/bin/env python3
"""Minimal token-level textual inversion for SDXL (dual CLIP encoders).

Only the newly added placeholder token rows in both SDXL text encoders are
trained. VAE, UNet and both text encoders stay frozen.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import random_transform


def load_crisp_prompt_templates():
    """Read the existing new_crisp template list without importing imgaug."""
    path = ROOT / "ldm" / "data" / "personalized.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "imagenet_templates_small":
                    return ast.literal_eval(node.value)
    raise RuntimeError("imagenet_templates_small not found")

_CRISP_DATA_ENHANCE = random_transform()

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def load_targets(paths, mask_paths, size, crops_per_image, device):
    """Build deterministic image patches; optionally carry matching masks."""
    to_t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    pairs = []
    for idx, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        mask_img = None
        if mask_paths:
            mask_img = Image.open(mask_paths[idx]).convert("L")
        w, h = img.size
        for c in range(crops_per_image):
            if c == 0:
                crop = img
                mcrop = mask_img
            else:
                frac = 0.65 if c % 2 else 0.8
                cw, ch = int(w * frac), int(h * frac)
                if c % 4 == 1:
                    x0, y0 = 0, 0
                elif c % 4 == 2:
                    x0, y0 = w - cw, h - ch
                elif c % 4 == 3:
                    x0, y0 = w - cw, 0
                else:
                    x0, y0 = 0, h - ch
                crop = img.crop((x0, y0, x0 + cw, y0 + ch))
                mcrop = mask_img.crop((x0, y0, x0 + cw, y0 + ch)) if mask_img is not None else None
            m = None
            if mcrop is not None:
                m = (mask_tf(mcrop).unsqueeze(0) > 0.5).to(device)
            pairs.append((to_t(crop).unsqueeze(0).to(device), m))
    return pairs

def load_dynamic_pair(path, mask_path, size, device, rng):
    """Use new_crisp's existing utils.random_transform for dynamic augmentation."""
    img = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    img_t = transforms.ToTensor()(img).unsqueeze(0)
    mask_t = (transforms.ToTensor()(mask).unsqueeze(0) > 0.5).float()
    try:
        img_t2, mask_t2 = _CRISP_DATA_ENHANCE(img_t, mask_t)
    except RuntimeError:
        # Thin masks can rotate out of frame; fall back to the untransformed pair.
        img_t2, mask_t2 = img_t, mask_t
    if mask_t2.sum() == 0:
        img_t2, mask_t2 = img_t, mask_t
    img_t, mask_t = img_t2, mask_t2
    img_t = F.interpolate(img_t, size=(size, size), mode="bicubic", align_corners=False)
    mask_t = (F.interpolate(mask_t, size=(size, size), mode="nearest") > 0.5).float()
    img_t = (img_t * 2.0 - 1.0).to(device)
    mask_t = mask_t.to(device)
    assert img_t.shape[-2:] == mask_t.shape[-2:], "image/mask spatial mismatch"
    return img_t, mask_t

def _expand_mask(mask, ids, token_id, n_vec):
    out = mask.clone()
    for b in range(out.shape[0]):
        pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        p = int(pos[0])
        out[b] = torch.cat([out[b, :p], torch.ones(n_vec, device=out.device, dtype=out.dtype),
                            out[b, p + 1:]], dim=0)[:out.shape[1]]
    return out


def _encode_with_token(text_encoder, ids, token_id, vec, attention_mask=None):
    """Run frozen CLIP and replace one placeholder row with N learned vectors."""
    emb_module = text_encoder.text_model.embeddings
    vec = vec
    n_vec = vec.shape[0]

    def hook(module, inputs, output):
        out = output.clone()
        for b in range(out.shape[0]):
            pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
            if pos.numel() == 0:
                raise ValueError("placeholder token not found after tokenization")
            p = int(pos[0])
            new_row = torch.cat([out[b, :p], vec.to(out.dtype), out[b, p + 1:]], dim=0)[:out.shape[1]]
            out[b] = new_row
        return out

    handle = emb_module.register_forward_hook(hook)
    try:
        if attention_mask is not None:
            attention_mask = _expand_mask(attention_mask, ids, token_id, n_vec)
        out = text_encoder(ids, attention_mask=attention_mask, output_hidden_states=True)
    finally:
        handle.remove()
    return out

def get_prompt_embeds(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt,
                      token_id, token_id_2, vec1, vec2, device):
    tok = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt")
    tok2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
                       truncation=True, return_tensors="pt")
    ids = tok.input_ids.to(device)
    ids2 = tok2.input_ids.to(device)
    mask = tok.attention_mask.to(device)
    mask2 = tok2.attention_mask.to(device)
    e1 = _encode_with_token(text_encoder, ids, token_id, vec1, mask)
    e2 = _encode_with_token(text_encoder_2, ids2, token_id_2, vec2, mask2)
    prompt_embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1)
    pooled = e2.text_embeds
    return prompt_embeds, pooled

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/sdxl-base-1.0")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--crops-per-image", type=int, default=4)
    p.add_argument("--mask-image", nargs="*", default=[])
    p.add_argument("--mask-loss", action="store_true")
    p.add_argument("--mask-project", default="raw", choices=["raw", "lfp_v3"])
    p.add_argument("--data-enhance", action="store_true", help="dynamic new_crisp-style augmentation per step")
    p.add_argument("--debug-mask-dir", default="", help="save image+mask overlays for geometry checks")
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--prompt", default="a photo of <anomaly>")
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--init-word", "--init_word", dest="init_word", default="anomaly")
    p.add_argument("--num-vectors", type=int, default=4)
    p.add_argument("--random-prompts", action="store_true")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--fp8", action="store_true", help="FP8 training of the UNet; requires torchao")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="textual_inversion_runs/sdxl_token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    rng = random.Random(args.seed)

    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer_2")
    tokenizer.add_tokens([args.placeholder])
    tokenizer_2.add_tokens([args.placeholder])
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=dtype).to(args.device)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.model, subfolder="text_encoder_2", torch_dtype=dtype).to(args.device)
    text_encoder.resize_token_embeddings(len(tokenizer))
    text_encoder_2.resize_token_embeddings(len(tokenizer_2))
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(args.device)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", torch_dtype=dtype).to(args.device)
    if args.fp8:
        from highres.fp8_utils import apply_fp8_training
        unet = apply_fp8_training(unet)
        print("[TI-SDXL] FP8 training enabled on UNet")
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    for m in (vae, unet, text_encoder, text_encoder_2):
        m.requires_grad_(False)
        m.eval()
    token_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    token_id_2 = tokenizer_2.convert_tokens_to_ids(args.placeholder)
    emb1 = text_encoder.get_input_embeddings()
    emb2 = text_encoder_2.get_input_embeddings()
    # Initialize the new token from a semantically meaningful word rather than
    # the random resize_token_embeddings row.
    init_ids = tokenizer(args.init_word, add_special_tokens=False).input_ids
    init_ids2 = tokenizer_2(args.init_word, add_special_tokens=False).input_ids
    print(f"[TI-SDXL] init_word={args.init_word!r} init_ids={init_ids} init_ids2={init_ids2}")
    init_id = init_ids[0] if init_ids else token_id
    init_id2 = init_ids2[0] if init_ids2 else token_id_2
    with torch.no_grad():
        emb1.weight[token_id].copy_(emb1.weight[init_id])
        emb2.weight[token_id_2].copy_(emb2.weight[init_id2])
    n_vec = max(1, int(args.num_vectors))
    vec1 = torch.nn.Parameter(
        emb1.weight[init_id].detach().float().repeat(n_vec, 1).clone())
    vec2 = torch.nn.Parameter(
        emb2.weight[init_id2].detach().float().repeat(n_vec, 1).clone())
    templates = load_crisp_prompt_templates() if args.random_prompts else []

    if args.mask_image and len(args.mask_image) != len(args.image):
        raise ValueError("--mask-image must correspond 1:1 to --image")
    dynamic = bool(args.data_enhance)
    if dynamic:
        latents_list, mask_latents = [], []
        print(f"[TI-SDXL] dynamic data_enhance from {len(args.image)} image(s)")
    else:
        targets = load_targets(args.image, args.mask_image, args.size, args.crops_per_image, args.device)
        with torch.no_grad():
            latents_list = [vae.encode(t[0].to(dtype)).latent_dist.sample() * vae.config.scaling_factor for t in targets]
            mask_latents = []
            for _, m in targets:
                if m is None:
                    mask_latents.append(None)
                    continue
                if args.mask_project == "lfp_v3":
                    from highres.lfp_mask import lfp_pixel_mask_v3, lfp_pixel_mask_v3_cached
                    from torchvision.transforms import ToPILImage
                    m_pil = ToPILImage()((m[0] * 255).to(torch.uint8).cpu())
                    m_pil = lfp_pixel_mask_v3(m_pil, vae, vae.config.scaling_factor, 8)
                    mt = transforms.ToTensor()(m_pil).unsqueeze(0).to(args.device)
                    m = (mt > 0.5).to(dtype)
                ml = F.interpolate(m.to(dtype), size=(latents_list[0].shape[-2], latents_list[0].shape[-1]), mode="nearest")
                mask_latents.append((ml > 0.5).to(dtype))
        print(f"[TI-SDXL] targets={len(latents_list)} from {len(args.image)} image(s) mask_loss={args.mask_loss}")
    add_time_ids = torch.tensor([[args.size, args.size, 0, 0, args.size, args.size]],
                                device=args.device, dtype=dtype)
    opt = torch.optim.AdamW([vec1, vec2], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    ema1 = vec1.detach().clone()
    ema2 = vec2.detach().clone()
    for step in range(1, args.steps + 1):
        train_prompt = args.prompt
        if templates:
            train_prompt = rng.choice(templates).format(args.placeholder)
        prompt_embeds, pooled = get_prompt_embeds(text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                                                  train_prompt, token_id, token_id_2,
                                                  vec1.to(dtype), vec2.to(dtype), args.device)
        if dynamic:
            j = (step - 1) % len(args.image)
            img_t, m_t = load_dynamic_pair(args.image[j], args.mask_image[j] if args.mask_image else None,
                                           args.size, args.device, rng)
            if args.debug_mask_dir and step <= 4:
                from PIL import ImageDraw
                dbg = Path(args.debug_mask_dir); dbg.mkdir(parents=True, exist_ok=True)
                vis = ((img_t[0].detach().float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
                vis = vis.permute(1, 2, 0).numpy()
                mm = (m_t[0, 0].detach().cpu().numpy() > 0.5)
                img_dbg = Image.fromarray(vis).convert("RGB")
                arr = np.asarray(img_dbg).copy(); arr[mm] = (0.5 * arr[mm] + 0.5 * np.array([255, 0, 0])).astype(np.uint8)
                Image.fromarray(arr).save(dbg / f"step{step:04d}.png")
            with torch.no_grad():
                latents = vae.encode(img_t.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
            ml = None
            if args.mask_loss and m_t is not None:
                if args.mask_project == "lfp_v3":
                    from highres.lfp_mask import lfp_pixel_mask_v3, lfp_pixel_mask_v3_cached
                    from torchvision.transforms import ToPILImage
                    mp = ToPILImage()((m_t[0] * 255).to(torch.uint8).cpu())
                    mp = lfp_pixel_mask_v3_cached(mp, vae, vae.config.scaling_factor, 8,
                                                   cache_key=args.mask_image[j])
                    m_t = (transforms.ToTensor()(mp).unsqueeze(0).to(args.device) > 0.5)
                ml = F.interpolate(m_t.to(dtype), size=latents.shape[-2:], mode="nearest")
                ml = (ml > 0.5).to(dtype)
        else:
            idx = (step - 1) % len(latents_list)
            latents = latents_list[idx]
            ml = mask_latents[idx] if args.mask_loss else None
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=args.device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
        if ml is not None:
            per = (pred - noise) ** 2 * ml
            loss = per.sum() / (ml.sum() * pred.shape[1]).clamp_min(1.0)
        else:
            loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            ema1.mul_(args.ema_decay).add_(vec1.detach(), alpha=1.0 - args.ema_decay)
            ema2.mul_(args.ema_decay).add_(vec2.detach(), alpha=1.0 - args.ema_decay)
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI-SDXL] step={step:4d} loss={loss.item():.6f}")

    torch.save({"vec1": vec1.detach().cpu(), "vec2": vec2.detach().cpu(),
                "prompt": args.prompt, "placeholder": args.placeholder, "num_vectors": n_vec, "init_word": args.init_word,
                "token_id": token_id, "token_id_2": token_id_2}, out / "sdxl_token_embedding.pt")
    torch.save({"vec1": ema1.cpu(), "vec2": ema2.cpu(),
                "prompt": args.prompt, "placeholder": args.placeholder, "num_vectors": n_vec, "init_word": args.init_word,
                "token_id": token_id, "token_id_2": token_id_2}, out / "sdxl_token_embedding_ema.pt")
    (out / "manifest.json").write_text(json.dumps({
        "prompt": args.prompt, "placeholder": args.placeholder, "num_vectors": n_vec, "init_word": args.init_word, "random_prompts": args.random_prompts, "steps": args.steps,
        "lr": args.lr, "images": args.image, "crops_per_image": args.crops_per_image,
        "ema_decay": args.ema_decay, "seed": args.seed, "size": args.size,
        "targets": len(latents_list), "dynamic_data_enhance": dynamic, "mask_loss": args.mask_loss, "mask_project": args.mask_project, "mask_image": args.mask_image,
        "note": "only placeholder rows in both SDXL text encoders are trainable",
    }, indent=2))
    print(f"[TI-SDXL] wrote {out}")


if __name__ == "__main__":
    main()
