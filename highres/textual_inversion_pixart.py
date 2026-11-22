#!/usr/bin/env python3
"""Token-level textual inversion probe for PixArt-Sigma (T5 encoder)."""
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


def load_image(path, size, device):
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/pixart_sigma_1024")
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="a photo of <anomaly>")
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="textual_inversion_runs/pixart_token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from diffusers import AutoencoderKL, DDPMScheduler, PixArtTransformer2DModel
    from transformers import T5EncoderModel, T5Tokenizer

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=dtype).to(args.device)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(args.device)
    transformer = PixArtTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype).to(args.device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    tokenizer.add_tokens([args.placeholder])
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    embedding = text_encoder.get_input_embeddings().weight
    vec = torch.nn.Parameter(embedding[token_id].detach().float().clone())
    for m in (vae, transformer, text_encoder):
        m.requires_grad_(False)
        m.eval()

    def encode_text():
        tok = tokenizer([args.prompt], padding="max_length", max_length=tokenizer.model_max_length,
                        truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(args.device)
        mask = tok.attention_mask.to(args.device)
        emb = text_encoder.get_input_embeddings()(ids).detach().clone()
        pos = (ids == token_id).nonzero(as_tuple=False)
        for _, idx in pos:
            emb[0, idx] = vec.to(emb.dtype)
        h = text_encoder(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        return h, mask

    image = load_image(args.image, args.size, args.device).to(dtype)
    with torch.no_grad():
        lat = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    opt = torch.optim.AdamW([vec], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    for step in range(1, args.steps + 1):
        h, mask = encode_text()
        noise = torch.randn_like(lat)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=args.device).long()
        noisy = scheduler.add_noise(lat, noise, t)
        pred = transformer(noisy, encoder_hidden_states=h, encoder_attention_mask=mask,
                           timestep=t, added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                           return_dict=False)[0]
        # PixArt-Sigma outputs 8 channels: first 4 are noise prediction.
        loss = F.mse_loss(pred[:, :4], noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI-PixArt] step={step:4d} loss={loss.item():.6f}")

    torch.save({"vec": vec.detach().cpu(), "prompt": args.prompt, "placeholder": args.placeholder,
                "token_id": token_id, "model": args.model}, out / "pixart_token_embedding.pt")
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "prompt": args.prompt, "placeholder": args.placeholder,
        "steps": args.steps, "lr": args.lr, "image": args.image, "seed": args.seed,
        "note": "only the T5 placeholder token row is trainable",
    }, indent=2))
    print(f"[TI-PixArt] wrote {out}")


if __name__ == "__main__":
    main()
