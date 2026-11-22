#!/usr/bin/env python3
"""Minimal token-level textual inversion for SDXL (dual CLIP encoders).

Only the newly added placeholder token rows in both SDXL text encoders are
trained. VAE, UNet and both text encoders stay frozen.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


def load_image(path, size, device):
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def _encode_with_token(text_encoder, ids, token_id, vec):
    """Run a frozen CLIP text encoder while replacing one token row via a hook."""
    emb_module = text_encoder.text_model.embeddings
    def hook(module, inputs, output):
        out = output.clone()
        pos = (ids == token_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            raise ValueError("placeholder token not found after tokenization")
        for _, idx in pos:
            out[0, idx] = vec.to(out.dtype)
        return out
    handle = emb_module.register_forward_hook(hook)
    try:
        out = text_encoder(ids, output_hidden_states=True)
    finally:
        handle.remove()
    return out


def get_prompt_embeds(text_encoder, text_encoder_2, tokenizer, tokenizer_2, prompt,
                      token_id, token_id_2, vec1, vec2, device):
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    ids2 = tokenizer_2(prompt, padding="max_length", max_length=tokenizer_2.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(device)
    e1 = _encode_with_token(text_encoder, ids, token_id, vec1)
    e2 = _encode_with_token(text_encoder_2, ids2, token_id_2, vec2)
    prompt_embeds = torch.cat([e1.hidden_states[-2], e2.hidden_states[-2]], dim=-1)
    pooled = e2.text_embeds
    return prompt_embeds, pooled

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/sdxl-base-1.0")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="a photo of <anomaly>")
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="textual_inversion_runs/sdxl_token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

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
    scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    for m in (vae, unet, text_encoder, text_encoder_2):
        m.requires_grad_(False)
        m.eval()
    token_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    token_id_2 = tokenizer_2.convert_tokens_to_ids(args.placeholder)
    emb1 = text_encoder.get_input_embeddings()
    emb2 = text_encoder_2.get_input_embeddings()
    vec1 = torch.nn.Parameter(emb1.weight[token_id].detach().float().clone())
    vec2 = torch.nn.Parameter(emb2.weight[token_id_2].detach().float().clone())

    image = load_image(args.image, args.size, args.device).to(text_encoder.dtype)
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    add_time_ids = torch.tensor([[args.size, args.size, 0, 0, args.size, args.size]],
                                device=args.device, dtype=dtype)
    opt = torch.optim.AdamW([vec1, vec2], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    for step in range(1, args.steps + 1):
        prompt_embeds, pooled = get_prompt_embeds(text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                                                  args.prompt, token_id, token_id_2,
                                                  vec1.to(dtype), vec2.to(dtype), args.device)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],),
                                  device=args.device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids}).sample
        loss = F.mse_loss(pred, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI-SDXL] step={step:4d} loss={loss.item():.6f}")

    torch.save({"vec1": vec1.detach().cpu(), "vec2": vec2.detach().cpu(),
                "prompt": args.prompt, "placeholder": args.placeholder,
                "token_id": token_id, "token_id_2": token_id_2}, out / "sdxl_token_embedding.pt")
    (out / "manifest.json").write_text(json.dumps({
        "prompt": args.prompt, "placeholder": args.placeholder, "steps": args.steps,
        "lr": args.lr, "image": args.image, "seed": args.seed, "size": args.size,
        "note": "only placeholder rows in both SDXL text encoders are trainable",
    }, indent=2))
    print(f"[TI-SDXL] wrote {out}")


if __name__ == "__main__":
    main()
