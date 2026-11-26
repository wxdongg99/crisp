#!/usr/bin/env python3
"""PixArt-Sigma T5 token TI with CRISP-style multi-vector + mask loss."""
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
from utils import random_transform

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def load_dynamic_pair(img_path, mask_path, size, device, vae=None, mask_project="raw"):
    """Dynamic 1024-native augmentation: resize, flip, rotate, mask-bbox crop."""
    img = Image.open(img_path).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    mask = Image.open(mask_path).convert("L").resize((size, size), Image.Resampling.NEAREST)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
    ang = random.randint(0, 20)
    img = img.rotate(ang, resample=Image.Resampling.BICUBIC)
    mask = mask.rotate(ang, resample=Image.Resampling.NEAREST)
    ma = np.asarray(mask) > 127
    if ma.any():
        ys, xs = np.where(ma)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        cx1, cy1 = random.randint(0, max(0, int(x0))), random.randint(0, max(0, int(y0)))
        cx2, cy2 = random.randint(min(size - 1, int(x1)), size - 1), random.randint(min(size - 1, int(y1)), size - 1)
        if cx2 > cx1 + 32 and cy2 > cy1 + 32:
            img = img.crop((cx1, cy1, cx2, cy2)).resize((size, size), Image.Resampling.BICUBIC)
            mask = mask.crop((cx1, cy1, cx2, cy2)).resize((size, size), Image.Resampling.NEAREST)
    if mask_project == "lfp_v3":
        from highres.lfp_mask import lfp_pixel_mask_v3_cached
        mask = lfp_pixel_mask_v3_cached(mask, vae, float(vae.config.scaling_factor), cache_key=str(mask_path))
    it = transforms.ToTensor()(img).unsqueeze(0)
    mt = (transforms.ToTensor()(mask).unsqueeze(0) > 0.5).float()
    return (it * 2 - 1).to(device), mt.to(device)

def expand_embeddings(emb, ids, token_id, vec, attention_mask):
    out = emb.clone()
    for b in range(out.shape[0]):
        pos = (ids[b] == token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        p = int(pos[0])
        out[b] = torch.cat([out[b, :p], vec.to(out.dtype), out[b, p + 1:]], dim=0)[:out.shape[1]]
        attention_mask[b] = torch.cat([
            attention_mask[b, :p],
            torch.ones(vec.shape[0], device=attention_mask.device, dtype=attention_mask.dtype),
            attention_mask[b, p + 1:],
        ])[:attention_mask.shape[1]]
    return out, attention_mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/pixart_sigma_1024")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--mask-image", nargs="*", default=[])
    p.add_argument("--prompt", required=True)
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--init-word", "--init_word", dest="init_word", default="anomaly")
    p.add_argument("--init-embedding", default="")
    p.add_argument("--num-vectors", type=int, default=4)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--steps", type=int, default=4800)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--mask-loss", action="store_true")
    p.add_argument("--embedding-reg-weight", type=float, default=0.0)
    p.add_argument("--data-enhance", action="store_true")
    p.add_argument("--mask-project", default="lfp_v3", choices=["raw", "lfp_v3"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--output_dir", default="textual_inversion_runs/pixart_token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    from diffusers import AutoencoderKL, DDPMScheduler, PixArtTransformer2DModel
    from transformers import T5EncoderModel, T5Tokenizer

    torch.manual_seed(args.seed); random.seed(args.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(args.model, subfolder="tokenizer", legacy=True)
    text_encoder = T5EncoderModel.from_pretrained(args.model, subfolder="text_encoder", torch_dtype=dtype).to(args.device)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype).to(args.device)
    transformer = PixArtTransformer2DModel.from_pretrained(args.model, subfolder="transformer", torch_dtype=dtype).to(args.device)
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02)
    tokenizer.add_tokens([args.placeholder]); text_encoder.resize_token_embeddings(len(tokenizer))
    token_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    embedding = text_encoder.get_input_embeddings()
    init_ids = tokenizer(args.init_word, add_special_tokens=False).input_ids
    init_id = init_ids[0] if init_ids else token_id
    if args.init_embedding:
        ckpt = torch.load(args.init_embedding, map_location="cpu")
        vec = torch.nn.Parameter(ckpt["vec"].float().to(args.device).clone())
        n_vec = vec.shape[0]
        print(f"[TI-PixArt] resumed {args.init_embedding} n_vec={n_vec}", flush=True)
    else:
        n_vec = int(args.num_vectors)
        with torch.no_grad():
            embedding.weight[token_id].copy_(embedding.weight[init_id])
        vec = torch.nn.Parameter(embedding.weight[init_id].detach().float().repeat(n_vec, 1).clone())
        print(f"[TI-PixArt] init_word={args.init_word!r} ids={init_ids} n_vec={n_vec}", flush=True)
    initial_vec = vec.detach().clone()
    max_vec_norm = float(initial_vec.norm().item()) * 2.0
    for m in (vae, transformer, text_encoder):
        m.requires_grad_(False); m.eval()
    if args.mask_image and len(args.mask_image) != len(args.image):
        raise ValueError("mask-image count must match image count")

    def encode_text():
        max_len = min(120, tokenizer.model_max_length)
        tok = tokenizer([args.prompt], padding="max_length", max_length=max_len,
                        truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(args.device); mask = tok.attention_mask.to(args.device)
        emb = text_encoder.get_input_embeddings()(ids).detach().clone()
        emb, mask = expand_embeddings(emb, ids, token_id, vec, mask)
        h = text_encoder(inputs_embeds=emb, attention_mask=mask).last_hidden_state
        return h, mask

    opt = torch.optim.AdamW([vec], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        j = (step - 1) % len(args.image)
        img, mask_t = load_dynamic_pair(args.image[j], args.mask_image[j], args.size, args.device,
                                        vae=vae, mask_project=args.mask_project)
        with torch.no_grad():
            lat = vae.encode(img.to(dtype)).latent_dist.sample() * vae.config.scaling_factor
        h, am = encode_text()
        noise = torch.randn_like(lat)
        t = torch.randint(0, scheduler.config.num_train_timesteps, (lat.shape[0],), device=args.device).long()
        noisy = scheduler.add_noise(lat, noise, t)
        pred = transformer(noisy, encoder_hidden_states=h, encoder_attention_mask=am,
                           timestep=t, added_cond_kwargs={"resolution": None, "aspect_ratio": None},
                           return_dict=False)[0][:, :4]
        if args.mask_loss and mask_t is not None:
            ml = F.interpolate(mask_t.to(dtype), size=lat.shape[-2:], mode="nearest")
            per = (pred - noise) ** 2 * ml
            loss = per.sum() / (ml.sum() * pred.shape[1]).clamp_min(1.0)
        else:
            loss = F.mse_loss(pred, noise)
        # CRISP embedding_manager2.embedding_to_coarse_loss(), applied to vec.
        diff = vec - initial_vec
        reg = (diff @ diff.T / max(1, diff.shape[0])).mean()
        loss = loss + args.embedding_reg_weight * reg
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([vec], 1.0)
        opt.step()
        with torch.no_grad():
            n = float(vec.norm().item())
            if n > max_vec_norm:
                vec.mul_(max_vec_norm / n)
        if step == 1 or step % 20 == 0 or step == args.steps:
            print(f"[TI-PixArt] step={step:4d} loss={loss.item():.6f} reg={reg.item():.6f} norm={vec.norm().item():.2f}", flush=True)
    torch.save({"vec": vec.detach().cpu(), "prompt": args.prompt, "placeholder": args.placeholder,
                "num_vectors": n_vec, "init_word": args.init_word, "token_id": token_id,
                "model": args.model}, out / "pixart_token_embedding.pt")
    (out / "manifest.json").write_text(json.dumps({
        "model": args.model, "prompt": args.prompt, "placeholder": args.placeholder,
        "num_vectors": n_vec, "init_word": args.init_word, "steps": args.steps,
        "lr": args.lr, "images": args.image, "mask_images": args.mask_image,
        "size": args.size, "mask_loss": args.mask_loss, "data_enhance": args.data_enhance,
        "mask_project": args.mask_project,
        "embedding_reg_weight": args.embedding_reg_weight, "final_vec_norm": float(vec.norm().item()),
        "elapsed_sec": round(time.time()-t0, 2),
    }, indent=2))
    print(f"[TI-PixArt] wrote {out}", flush=True)

if __name__ == "__main__":
    main()
