#!/usr/bin/env python3
"""Token-level textual inversion for the official AnomalyDiffusion LDM.

Only a new placeholder-token embedding is optimized. VAE, UNet and BERT are
frozen. Unlike ``textual_inversion_basic.py`` (soft-prompt delta), this replaces
the token embedding before the BERT encoder, so it is a real token-level probe.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    config.model.params.personalization_config = None
    pl_sd = torch.load(ckpt_path, map_location="cpu")
    sd = pl_sd.get("state_dict", pl_sd)
    config.model.params.ckpt_path = ckpt_path
    model = instantiate_from_config(config.model)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[TI-token] loaded {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_image(path, size, device):
    tf = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)


def make_conditioning(model, prompt, token_text, token_embedding):
    """Return BERT conditioning with token_text's embedding replaced."""
    enc = model.cond_stage_model
    ids = enc.tknz_fn([prompt])  # [1, 77]
    tok_id = enc.tknz_fn.tokenizer.convert_tokens_to_ids(token_text)
    pos = (ids == tok_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        raise ValueError(f"placeholder {token_text!r} not found in prompt ids")
    pos = pos[:, 1]
    base = enc.transformer.token_emb(ids)
    emb = base.clone()
    # One token vector shared across all placeholder occurrences.
    for j, idx in enumerate(pos.tolist()):
        emb[0, idx] = token_embedding[0]
    z, _ = enc.transformer(ids, input_embeddings=emb, return_embeddings=True)
    return z, ids, pos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="a photo of anomaly")
    p.add_argument("--placeholder", default="anomaly")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="textual_inversion_runs/token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample_steps", type=int, default=50)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(args.config, args.ckpt, args.device)
    image = load_image(args.image, args.size, args.device)

    enc = model.cond_stage_model
    ids = enc.tknz_fn([args.prompt])
    tok_id = enc.tknz_fn.tokenizer.convert_tokens_to_ids(args.placeholder)
    pos = (ids == tok_id).nonzero(as_tuple=False)
    if pos.numel() == 0:
        raise ValueError(f"placeholder {args.placeholder!r} not found in prompt ids")
    init = enc.transformer.token_emb(torch.tensor([[tok_id]], device=ids.device)).detach().clone()
    token_embedding = torch.nn.Parameter(init)

    with torch.no_grad():
        z = model.get_first_stage_encoding(model.encode_first_stage(image))

    opt = torch.optim.AdamW([token_embedding], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    for step in range(1, args.steps + 1):
        c, _, _ = make_conditioning(model, args.prompt, args.placeholder, token_embedding)
        t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=args.device).long()
        noise = torch.randn_like(z)
        z_t = model.q_sample(z, t, noise)
        eps = model.apply_model(z_t, t, c)
        loss = F.mse_loss(eps, noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI-token] step={step:4d} loss={loss.item():.6f}")

    torch.save({"token_embedding": token_embedding.detach().cpu(), "prompt": args.prompt,
                "placeholder": args.placeholder, "token_id": tok_id}, out / "token_embedding.pt")
    with torch.no_grad():
        c, _, _ = make_conditioning(model, args.prompt, args.placeholder, token_embedding)
        sampler = DDIMSampler(model)
        samples, _ = sampler.sample(S=args.sample_steps, batch_size=1,
                                    shape=(4, args.size // 8, args.size // 8),
                                    conditioning=c, eta=0.0, verbose=False)
        decoded = model.decode_first_stage(samples)
        save_image((decoded + 1) / 2, out / "token_sample.png")
    (out / "manifest.json").write_text(json.dumps({
        "prompt": args.prompt, "placeholder": args.placeholder, "steps": args.steps,
        "lr": args.lr, "image": args.image, "seed": args.seed,
        "note": "only placeholder token embedding is trainable",
    }, indent=2))
    print(f"[TI-token] wrote {out}")


if __name__ == "__main__":
    main()
