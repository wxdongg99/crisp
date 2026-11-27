#!/usr/bin/env python3
"""Token-level textual inversion for FLUX.1-dev (dual CLIP-L + T5-XXL encoders).

Only the placeholder token rows of *both* FLUX text encoders are trained; VAE,
FluxTransformer2DModel, CLIP-L and T5-XXL stay frozen. This mirrors the CRISP
PixArt / SDXL textual inversion recipes in this repo:

  highres/textual_inversion_pixart.py   (single encoder, masked flow matching)
  highres/textual_inversion_sdxl.py     (dual encoder, multi-vector, LFP v3)

FLUX specifics (verified against diffusers 0.36 `FluxPipeline` source, not
guessed):

  * 16-channel VAE latents, normalised as
        L = (z - vae.config.shift_factor) * vae.config.scaling_factor
    (inverse of `FluxPipeline.__call__`: `latents / scaling + shift`).
  * Latents are *packed* 2x2: `FluxPipeline._pack_latents(z, b, 16, h_lat, w_lat)`
    gives `[b, (h/2)*(w/2), 64]` tokens; ids come from
    `FluxPipeline._prepare_latent_image_ids(b, h_lat//2, w_lat//2, device, dtype)`.
    For 1024x1024 that is 64x64 = 4096 image tokens.
  * Text conditioning is `encoder_hidden_states` = T5-XXL **last** hidden state
    (4096-d; FLUX does *not* use `hidden_states[-2]` like SDXL) plus
    `pooled_projections` = CLIP-L pooled output (768-d). `txt_ids` are zeros.
  * The frozen transformer is guidance-distilled (`guidance_embeds=True`), so a
    guidance scalar tensor must be passed. The denoiser input timestep is
    `sigma` (the pipeline passes `timestep / 1000` where timestep = sigma*1000).
  * Rectified flow target is `noise - latent` (velocity), i.e. diffusers'
    `train_dreambooth_lora_flux.py` convention with sigma from
    `calculate_shift()` + `FlowMatchEulerDiscreteScheduler.time_shift()`.

Prompts come from `highres/prompt_templates.py` (`imagenet_templates_small`) by
default; `--fixed-prompt` is only an optional escape hatch.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

# The B300 env (/data/solo/wxd/envs/highres_py310) inherits its base
# interpreter's site-packages (/data/solo/Onevision-2/envs/onevision2), where
# sentencepiece 0.1.98 loads a protobuf-3 generated module while protobuf 6.x is
# installed -> "Descriptors cannot be created directly" when transformers
# converts the slow T5 tokenizer to the fast one. Pure-python protobuf parsing
# sidesteps it; see docs/FLUX_TI_SMOKE.md.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import random_transform  # noqa: E402  (new_crisp dynamic augmentation)

_CRISP_DATA_ENHANCE = random_transform()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision import transforms  # noqa: E402

if torch.cuda.is_available():
    # cuDNN's fused MHA (one of the SDPA backends) crashes on this B300
    # (driver 595.84 / torch 2.9.1+cu130): "Expected
    # mha_graph->execute(...).is_good() to be true" -> illegal memory access,
    # which surfaces in the backward pass as CUBLAS_STATUS_EXECUTION_FAILED.
    # Force flash / memory-efficient attention instead.
    torch.backends.cuda.enable_cudnn_sdp(False)

from highres.prompt_templates import canonical_prompt, load_templates  # noqa: E402


# ---------------------------------------------------------------------------
# FLUX prompt encoding with a multi-vector placeholder in both encoders
# ---------------------------------------------------------------------------
class FluxTextEncoders:
    """Encode a prompt with `n` learned vectors spliced into CLIP-L and T5-XXL.

    The placeholder token id occupies exactly one row in each tokenizer output.
    That row is replaced by the `n` learned vectors and the attention mask is
    expanded to match (the sequence is kept at its original length by dropping
    trailing padding rows, so CLIP stays at 77 and T5 at `max_sequence_length`).

    CLIP-L is fed through a forward hook on `text_model.embeddings` because
    transformers 4.46 `CLIPTextModel` has no `inputs_embeds` argument; T5 is fed
    `inputs_embeds` directly (T5 uses relative positions, so splicing is exact).
    The CLIP pooled row is re-gathered at the *shifted* EOS index, because the
    model's own pooler uses `input_ids.argmax()` and would read the wrong row
    after the placeholder becomes `n` rows.
    """

    def __init__(self, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                 placeholder, attn_mode="expanded", max_sequence_length=512,
                 device="cuda"):
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.placeholder = placeholder
        self.token_id = tokenizer.convert_tokens_to_ids(placeholder)
        self.token_id_2 = tokenizer_2.convert_tokens_to_ids(placeholder)
        self.attn_mode = attn_mode
        self.max_sequence_length = int(max_sequence_length)
        self.device = device
        self.clip_max_length = int(tokenizer.model_max_length)
        self.eos_id = tokenizer.eos_token_id

    @staticmethod
    def _first_pos(ids_row, token_id):
        pos = (ids_row == token_id).nonzero(as_tuple=False).flatten()
        if pos.numel() == 0:
            raise ValueError("placeholder token not found after tokenization")
        return int(pos[0])

    @staticmethod
    def _expand_mask(mask, pos, n_vec):
        """Insert `n_vec` ones at `pos`, keep the original length (drop pads)."""
        length = mask.shape[1]
        ones = torch.ones((mask.shape[0], n_vec), dtype=mask.dtype, device=mask.device)
        out = torch.cat([mask[:, :pos], ones, mask[:, pos + 1:]], dim=1)[:, :length]
        if out.shape[1] < length:
            pad = torch.zeros((mask.shape[0], length - out.shape[1]),
                              dtype=mask.dtype, device=mask.device)
            out = torch.cat([out, pad], dim=1)
        return out

    def encode_clip(self, prompt, vec):
        tok = self.tokenizer(prompt, padding="max_length", max_length=self.clip_max_length,
                             truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        length = ids.shape[1]
        n_vec = int(vec.shape[0])
        pos = self._first_pos(ids[0], self.token_id)
        if pos + n_vec > length:
            raise ValueError(f"placeholder+{n_vec} vectors do not fit CLIP length {length}")

        def hook(module, args, output):  # out: [b, 77, 768] (token+pos embeds)
            out = output.clone()
            out[0] = torch.cat([out[0, :pos], vec.to(out.dtype), out[0, pos + 1:]], dim=0)[:length]
            return out

        handle = self.text_encoder.text_model.embeddings.register_forward_hook(hook)
        try:
            kwargs = {}
            if self.attn_mode == "expanded":
                kwargs["attention_mask"] = self._expand_mask(mask, pos, n_vec)
            out = self.text_encoder(ids, output_hidden_states=False, **kwargs)
        finally:
            handle.remove()

        hidden = out.last_hidden_state  # [1, 77, 768]
        eos_pos = int((ids[0] == self.eos_id).nonzero(as_tuple=False)[0, 0])
        shifted = eos_pos + (n_vec - 1) if pos < eos_pos else eos_pos
        shifted = min(shifted, hidden.shape[1] - 1)
        return hidden, hidden[:, shifted]

    def encode_t5(self, prompt, vec):
        tok = self.tokenizer_2(prompt, padding="max_length",
                               max_length=self.max_sequence_length,
                               truncation=True, return_tensors="pt")
        ids = tok.input_ids.to(self.device)
        mask = tok.attention_mask.to(self.device)
        length = ids.shape[1]
        n_vec = int(vec.shape[0])
        pos = self._first_pos(ids[0], self.token_id_2)
        if pos + n_vec > length:
            raise ValueError(f"placeholder+{n_vec} vectors do not fit T5 length {length}")
        with torch.no_grad():
            emb = self.text_encoder_2.shared(ids)  # frozen embedding lookup
        spliced = torch.cat([emb[0, :pos], vec.to(emb.dtype), emb[0, pos + 1:]], dim=0)[:length]
        kwargs = {}
        if self.attn_mode == "expanded":
            kwargs["attention_mask"] = self._expand_mask(mask, pos, n_vec)
        out = self.text_encoder_2(inputs_embeds=spliced.unsqueeze(0), **kwargs)
        return out.last_hidden_state

    def __call__(self, prompt, vec_clip, vec_t5):
        _, pooled = self.encode_clip(prompt, vec_clip)
        t5 = self.encode_t5(prompt, vec_t5)
        text_ids = torch.zeros(t5.shape[1], 3, device=t5.device, dtype=t5.dtype)
        return t5, pooled, text_ids


# ---------------------------------------------------------------------------
# FLUX latent helpers (delegate to FluxPipeline so packing never drifts)
# ---------------------------------------------------------------------------
def flux_pack(latents, height_lat, width_lat):
    from diffusers import FluxPipeline
    b, c = latents.shape[0], latents.shape[1]
    return FluxPipeline._pack_latents(latents, b, c, height_lat, width_lat)


def flux_unpack(packed, height_lat, width_lat):
    from diffusers import FluxPipeline
    return FluxPipeline._unpack_latents(packed, height_lat * 8, width_lat * 8, 8)


def flux_img_ids(batch, height_lat, width_lat, device, dtype):
    from diffusers import FluxPipeline
    return FluxPipeline._prepare_latent_image_ids(batch, height_lat // 2, width_lat // 2,
                                                  device, dtype)


def flux_calculate_shift(image_seq_len, scheduler):
    """Same formula as diffusers.pipelines.flux.pipeline_flux.calculate_shift."""
    cfg = scheduler.config
    base_seq = cfg.get("base_image_seq_len", 256)
    max_seq = cfg.get("max_image_seq_len", 4096)
    base_shift = cfg.get("base_shift", 0.5)
    max_shift = cfg.get("max_shift", 1.15)
    m = (max_shift - base_shift) / (max_seq - base_seq)
    return image_seq_len * m + (base_shift - m * base_seq)


def flux_time_shift(mu, sigma, t):
    """Same formula as FlowMatchEulerDiscreteScheduler._time_shift_exponential."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0) ** sigma)


def sample_sigmas(batch, mode, mu, logit_mean, logit_std, device):
    """Flow-matching sigma ~ U/logit-normal, then FLUX dynamic shift (as diffusers)."""
    if mode == "uniform":
        u = torch.rand(batch, device=device, dtype=torch.float32)
    else:
        u = torch.sigmoid(torch.randn(batch, device=device, dtype=torch.float32) * logit_std + logit_mean)
    u = u.clamp(1e-4, 1.0 - 1e-4)
    if mu is not None:
        u = flux_time_shift(mu, 1.0, u)
    return u.clamp(1e-4, 1.0 - 1e-4)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dynamic_pair(path, mask_path, size, device):
    """new_crisp dynamic augmentation with synchronised image/mask transform."""
    img = Image.open(path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
    img_t = transforms.ToTensor()(img).unsqueeze(0)
    mask_t = (transforms.ToTensor()(mask).unsqueeze(0) > 0.5).float()
    try:
        img_t2, mask_t2 = _CRISP_DATA_ENHANCE(img_t, mask_t)
    except RuntimeError:
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


def load_targets(paths, mask_paths, size, crops_per_image, device, dtype):
    """Deterministic crops (used when --data-enhance is off)."""
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
        mask_img = Image.open(mask_paths[idx]).convert("L") if mask_paths else None
        w, h = img.size
        for c in range(crops_per_image):
            if c == 0:
                crop, mcrop = img, mask_img
            else:
                frac = 0.65 if c % 2 else 0.8
                cw, ch = int(w * frac), int(h * frac)
                x0, y0 = [(0, 0), (w - cw, h - ch), (w - cw, 0), (0, h - ch)][(c - 1) % 4]
                crop = img.crop((x0, y0, x0 + cw, y0 + ch))
                mcrop = mask_img.crop((x0, y0, x0 + cw, y0 + ch)) if mask_img is not None else None
            m = (mask_tf(mcrop).unsqueeze(0) > 0.5).to(device) if mcrop is not None else None
            pairs.append((to_t(crop).unsqueeze(0).to(device, dtype), m))
    return pairs


# ---------------------------------------------------------------------------
# LFP v3 mask projection
# ---------------------------------------------------------------------------
def _lfp_latent_flux16(mask, vae, scale_factor, target_h, target_w):
    """LFP Trim+Bilinear+Fill over all 16 FLUX VAE channels.

    `highres/lfp_mask._lfp_latent` hardcodes `for c in range(4)` because it was
    written for the 4-channel SDXL VAE. FLUX's VAE has 16 channels, so this
    local copy keeps the identical Trim+Bilinear+Fill + Otsu logic but scans all
    channels. Only used when `--mask-project lfp_v3_f16` is selected.
    """
    out = []
    for b in range(mask.shape[0]):
        m = (mask[b:b + 1] >= 0.5).float()
        bil = F.interpolate(m.float(), size=(target_h, target_w), mode="bilinear",
                            align_corners=False)
        m3 = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
        m3 = m3.to(next(vae.parameters()).device, next(vae.parameters()).dtype)
        with torch.no_grad():
            z = vae.encode(m3).latent_dist.mode() * scale_factor
        abs_z = z[0].detach().float().cpu().numpy()
        white_minority = float(m[0, 0].mean().item()) < 0.5
        channels = np.zeros_like(abs_z)
        for c in range(abs_z.shape[0]):
            data = abs_z[c].ravel()
            if data.max() <= 1e-6:
                continue
            rng_ = max(data.max() - data.min(), 1e-8)
            thr = _otsu((data - data.min()) / rng_)
            above = abs_z[c] > data.min() + thr * rng_
            above_minority = above.sum() < above.size / 2
            if white_minority:
                channels[c] = above if above_minority else 1.0 - above
            else:
                channels[c] = above if not above_minority else 1.0 - above
        union = (channels.sum(axis=0) > 0).astype(np.float64)
        union[0, :] = 0
        union[-1, :] = 0
        union[:, 0] = 0
        union[:, -1] = 0
        final = np.maximum(union, bil[0, 0].cpu().numpy())
        while True:
            p = np.pad(final, 1, constant_values=0)
            neigh = ((p[:-2, 1:-1] > .5).astype(int) + (p[2:, 1:-1] > .5).astype(int)
                     + (p[1:-1, :-2] > .5).astype(int) + (p[1:-1, 2:] > .5).astype(int))
            fill = (final < .5) & (neigh >= 3)
            if not fill.any():
                break
            final[fill] = 1.0
        out.append(torch.from_numpy((final >= 0.5).astype(np.float32))[None, None])
    return torch.cat(out, 0)


def _otsu(x):
    hist, edges = np.histogram(x.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    p = hist / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * (edges[:-1] + edges[1:]) / 2)
    denom = omega * (1.0 - omega)
    denom[denom < 1e-12] = 1e-12
    sigma_b = (mu[-1] * omega - mu) ** 2 / denom
    return float(edges[int(np.argmax(sigma_b))])


def lfp_v3_f16(mask_pil, vae, scale_factor, vae_scale=8):
    """LFP v3 with a 16-channel-aware area target (see `_lfp_latent_flux16`)."""
    from highres.lfp_mask import _nearest_dilate_to_area
    w, h = mask_pil.size
    m = torch.from_numpy(np.asarray(mask_pil.convert("L"))).float()[None, None] / 255.0
    latent = _lfp_latent_flux16(m, vae, scale_factor, h // vae_scale, w // vae_scale)
    pix = F.interpolate(latent, size=(h, w), mode="nearest")[0, 0].cpu().numpy()
    v1 = Image.fromarray((pix * 255).astype(np.uint8), mode="L")
    target = int((np.asarray(v1) > 127).sum())
    return _nearest_dilate_to_area(mask_pil, target, vae_scale)


def project_mask(mask_t, mode, vae, scaling_factor, vae_scale, cache_key=""):
    """mask_t: [1,1,H,W] in {0,1} on GPU -> projected [1,1,H,W] on CPU."""
    from torchvision.transforms import ToPILImage
    m_pil = ToPILImage()((mask_t[0] * 255).to(torch.uint8).cpu())
    if mode == "lfp_v3":
        from highres.lfp_mask import lfp_pixel_mask_v3_cached
        m_pil = lfp_pixel_mask_v3_cached(m_pil, vae, scaling_factor, vae_scale, cache_key=cache_key)
    elif mode == "lfp_v3_f16":
        m_pil = lfp_v3_f16(m_pil, vae, scaling_factor, vae_scale)
    return transforms.ToTensor()(m_pil).unsqueeze(0)


def mask_to_tokens(mask_lat_t, reduce="max"):
    """[1,1,H,W] latent mask -> [1, num_img_tokens] packed-token mask."""
    if reduce == "max":
        t = F.max_pool2d(mask_lat_t, 2, 2)
    else:
        t = F.avg_pool2d(mask_lat_t, 2, 2)
    return t.flatten(1)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="/media/cc/U2-Server/crisp-storage/models/FLUX.1-dev")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--mask-image", nargs="*", default=[])
    p.add_argument("--crops-per-image", type=int, default=1)
    p.add_argument("--data-enhance", action="store_true",
                   help="new_crisp dynamic augmentation + fresh VAE encode every step")
    p.add_argument("--mask-loss", action="store_true", help="restrict flow-matching loss to mask")
    p.add_argument("--mask-project", default="raw", choices=["raw", "lfp_v3", "lfp_v3_f16"])
    p.add_argument("--mask-token-reduce", default="max", choices=["max", "mean"])
    p.add_argument("--debug-mask-dir", default="")
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--init-word", "--init_word", dest="init_word", default="anomaly")
    p.add_argument("--num-vectors", type=int, default=4)
    p.add_argument("--init-embedding", default="", help="resume token vectors from a ckpt")
    p.add_argument("--prompt", default="", help="explicit prompt (only with --fixed-prompt)")
    p.add_argument("--fixed-prompt", action="store_true",
                   help="use --prompt/--template-index instead of random templates")
    p.add_argument("--template-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--vae-dtype", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--guidance", type=float, default=3.5,
                   help="guidance embedding value used during training (FLUX dev distilled)")
    p.add_argument("--max-sequence-length", type=int, default=512)
    p.add_argument("--attn-mode", default="expanded", choices=["expanded", "none"],
                   help="expanded: pass expanded attention masks to both encoders")
    p.add_argument("--timestep-sampling", default="logit_normal", choices=["logit_normal", "uniform"])
    p.add_argument("--logit-mean", type=float, default=0.0)
    p.add_argument("--logit-std", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=0, help="0 = only at the end")
    p.add_argument("--output_dir", default="textual_inversion_runs/flux_token")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry-run", action="store_true",
                   help="one forward/backward to validate shapes, then exit without saving")
    args = p.parse_args()

    rng = random.Random(args.seed)
    from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler,
                           FluxTransformer2DModel)
    from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    vae_dtype = torch.float32 if args.vae_dtype == "fp32" else dtype
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ---- tokenizers -------------------------------------------------------
    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    tokenizer_2 = T5TokenizerFast.from_pretrained(args.model, subfolder="tokenizer_2")
    tokenizer.add_tokens([args.placeholder])
    tokenizer_2.add_tokens([args.placeholder])

    # ---- frozen modules ---------------------------------------------------
    text_encoder = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder", torch_dtype=dtype).to(args.device)
    text_encoder_2 = T5EncoderModel.from_pretrained(
        args.model, subfolder="text_encoder_2", torch_dtype=dtype).to(args.device)
    text_encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    text_encoder_2.resize_token_embeddings(len(tokenizer_2), mean_resizing=False)
    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=vae_dtype).to(args.device)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.model, subfolder="transformer", torch_dtype=dtype).to(args.device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")

    for m in (vae, transformer, text_encoder, text_encoder_2):
        m.requires_grad_(False)
        m.eval()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    scaling = float(vae.config.scaling_factor)
    shift = float(vae.config.shift_factor)
    height_lat = args.size // 8
    width_lat = args.size // 8
    img_ids = flux_img_ids(1, height_lat, width_lat, args.device, dtype)
    image_seq_len = (height_lat // 2) * (width_lat // 2)
    mu = None
    if getattr(scheduler.config, "use_dynamic_shifting", False):
        mu = flux_calculate_shift(image_seq_len, scheduler)

    # ---- token vectors ----------------------------------------------------
    if args.mask_image and len(args.mask_image) != len(args.image):
        raise ValueError("--mask-image must correspond 1:1 to --image")
    emb_clip = text_encoder.get_input_embeddings()
    emb_t5 = text_encoder_2.shared
    token_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    token_id_2 = tokenizer_2.convert_tokens_to_ids(args.placeholder)
    init_ids = tokenizer(args.init_word, add_special_tokens=False).input_ids
    init_ids2 = tokenizer_2(args.init_word, add_special_tokens=False).input_ids
    init_id = init_ids[0] if init_ids else token_id
    init_id2 = init_ids2[0] if init_ids2 else token_id_2
    print(f"[TI-FLUX] init_word={args.init_word!r} clip_ids={init_ids} t5_ids={init_ids2} "
          f"token_id={token_id}/{token_id_2}", flush=True)
    with torch.no_grad():
        emb_clip.weight[token_id].copy_(emb_clip.weight[init_id])
        emb_t5.weight[token_id_2].copy_(emb_t5.weight[init_id2])

    if args.init_embedding:
        prev = torch.load(args.init_embedding, map_location="cpu")
        if prev.get("placeholder") != args.placeholder:
            raise ValueError(f"placeholder mismatch: {prev.get('placeholder')} vs {args.placeholder}")
        vec_clip = torch.nn.Parameter(prev["vec_clip"].float().to(args.device).clone())
        vec_t5 = torch.nn.Parameter(prev["vec_t5"].float().to(args.device).clone())
        n_vec = int(prev.get("num_vectors", vec_clip.shape[0]))
        print(f"[TI-FLUX] resumed {args.init_embedding} n_vec={n_vec}", flush=True)
    else:
        n_vec = max(1, int(args.num_vectors))
        vec_clip = torch.nn.Parameter(
            emb_clip.weight[init_id].detach().float().repeat(n_vec, 1).clone())
        vec_t5 = torch.nn.Parameter(
            emb_t5.weight[init_id2].detach().float().repeat(n_vec, 1).clone())

    enc = FluxTextEncoders(text_encoder, text_encoder_2, tokenizer, tokenizer_2,
                           args.placeholder, attn_mode=args.attn_mode,
                           max_sequence_length=args.max_sequence_length, device=args.device)

    # ---- prompt bookkeeping ----------------------------------------------
    templates = load_templates("imagenet_templates_small")
    fixed = bool(args.fixed_prompt)
    if fixed:
        train_prompt = args.prompt or canonical_prompt(args.placeholder, args.template_index)
    prompt_mode = "fixed" if fixed else "imagenet_templates_small"
    print(f"[TI-FLUX] prompt_mode={prompt_mode} templates={len(templates)} "
          f"canonical={canonical_prompt(args.placeholder, args.template_index)!r}", flush=True)

    # ---- deterministic pre-encode (no --data-enhance) --------------------
    pre = []
    if not args.data_enhance:
        pairs = load_targets(args.image, args.mask_image, args.size,
                             args.crops_per_image, args.device, dtype)
        with torch.no_grad():
            for k, (img_t, mask_t) in enumerate(pairs):
                z = vae.encode(img_t.to(vae_dtype)).latent_dist.sample()
                lat = ((z.float() - shift) * scaling).to(dtype)
                m = mask_t
                if m is not None and args.mask_project != "raw":
                    src = args.image[min(k // max(args.crops_per_image, 1), len(args.image) - 1)]
                    m = project_mask(m, args.mask_project, vae, scaling, 8,
                                     cache_key=src).to(args.device)
                pre.append((lat, m))
        print(f"[TI-FLUX] pre-encoded {len(pre)} targets", flush=True)

    opt = torch.optim.AdamW([vec_clip, vec_t5], lr=args.lr, betas=(0.9, 0.999),
                            weight_decay=0.0)
    ema_clip = vec_clip.detach().clone()
    ema_t5 = vec_t5.detach().clone()
    guidance = torch.full((1,), args.guidance, device=args.device, dtype=torch.float32)

    log_path = out / "train_log.csv"
    if not args.dry_run:
        with open(log_path, "w") as f:
            f.write("step,loss,sigma,mask_ratio,step_sec,lr\n")

    t_start = time.time()
    losses, step_times = [], []
    empty_mask_steps = 0
    steps = 1 if args.dry_run else args.steps
    for step in range(1, steps + 1):
        t0 = time.time()
        if fixed:
            prompt = train_prompt
        else:
            prompt = rng.choice(templates).format(args.placeholder)

        # ---- image / mask (dynamic augmentation) ----
        if args.data_enhance:
            j = (step - 1) % len(args.image)
            img_t, mask_t = load_dynamic_pair(args.image[j],
                                              args.mask_image[j] if args.mask_image else None,
                                              args.size, args.device)
            if args.mask_image and args.mask_project != "raw":
                proj = project_mask(mask_t, args.mask_project, vae, scaling, 8,
                                    cache_key=args.mask_image[j])
                mask_t = (proj.to(args.device) > 0.5).float()
            if args.debug_mask_dir and step <= 3:
                dbg = Path(args.debug_mask_dir)
                dbg.mkdir(parents=True, exist_ok=True)
                vis = ((img_t[0].detach().float().cpu() * 0.5 + 0.5).clamp(0, 1) * 255).byte()
                arr = vis.permute(1, 2, 0).numpy()
                mm = mask_t[0, 0].detach().cpu().numpy() > 0.5
                arr = arr.astype(np.float32)
                arr[mm] = 0.5 * arr[mm] + 0.5 * np.array([255, 0, 0])
                Image.fromarray(arr.astype(np.uint8)).save(dbg / f"step{step:04d}.png")
            with torch.no_grad():
                z = vae.encode(img_t.to(vae_dtype)).latent_dist.sample()
                lat = ((z.float() - shift) * scaling).to(dtype)
        else:
            lat, mask_t = pre[(step - 1) % len(pre)]

        # ---- text encoding (grad flows into the token vectors only) ----
        prompt_embeds, pooled, text_ids = enc(prompt, vec_clip, vec_t5)

        # ---- flow matching ----
        packed = flux_pack(lat, height_lat, width_lat)
        noise = torch.randn_like(packed)
        sigma = sample_sigmas(1, args.timestep_sampling, mu, args.logit_mean,
                              args.logit_std, args.device)
        s = sigma.view(1, 1, 1).to(packed.dtype)
        noisy = (1.0 - s) * packed + s * noise
        target = noise - packed
        pred = transformer(hidden_states=noisy.to(dtype), timestep=sigma.to(dtype),
                           guidance=guidance, pooled_projections=pooled.to(dtype),
                           encoder_hidden_states=prompt_embeds,
                           txt_ids=text_ids, img_ids=img_ids,
                           return_dict=False)[0]

        mask_tok = None
        mask_ratio = 0.0
        if args.mask_loss and mask_t is not None:
            ml = F.interpolate(mask_t.to(dtype), size=(height_lat, width_lat), mode="nearest")
            ml = (ml > 0.5).to(dtype)
            mask_tok = mask_to_tokens(ml, args.mask_token_reduce)  # [1, tokens]
            mask_ratio = float(mask_tok.mean().item())
        if mask_tok is not None and mask_ratio > 0.0:
            per = (pred.float() - target.float()) ** 2
            w = mask_tok.float().unsqueeze(-1)
            loss = (per * w).sum() / (w.sum() * per.shape[-1]).clamp_min(1.0)
        else:
            # No mask, or the augmentation rotated the defect fully out of frame:
            # fall back to the full-image flow-matching loss so the step is not
            # wasted (an all-zero weight sum would give loss == 0 and no grad).
            loss = F.mse_loss(pred.float(), target.float())
            if args.mask_loss:
                empty_mask_steps += 1

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            ema_clip.mul_(args.ema_decay).add_(vec_clip.detach(), alpha=1.0 - args.ema_decay)
            ema_t5.mul_(args.ema_decay).add_(vec_t5.detach(), alpha=1.0 - args.ema_decay)

        dt = time.time() - t0
        losses.append(float(loss.item()))
        step_times.append(dt)
        if step == 1 or step % args.log_every == 0 or step == steps:
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if torch.cuda.is_available() else 0
            print(f"[TI-FLUX] step={step:5d} loss={loss.item():.6f} "
                  f"sigma={float(sigma.item()):.3f} mask={mask_ratio:.3f} "
                  f"{dt*1000:.0f}ms peak={peak:.0f}MiB", flush=True)
        if not args.dry_run:
            with open(log_path, "a") as f:
                f.write(f"{step},{loss.item():.8f},{float(sigma.item()):.6f},"
                        f"{mask_ratio:.6f},{dt:.4f},{args.lr}\n")
        if args.save_every and step % args.save_every == 0 and not args.dry_run:
            torch.save({"vec_clip": vec_clip.detach().cpu(), "vec_t5": vec_t5.detach().cpu(),
                        "placeholder": args.placeholder, "num_vectors": n_vec,
                        "step": step}, out / f"step{step:06d}.pt")

    elapsed = time.time() - t_start
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
        peak_res = torch.cuda.max_memory_reserved() / 1024 ** 2
    else:
        peak_alloc = peak_res = 0.0
    avg_step = sum(step_times) / len(step_times)
    print(f"[TI-FLUX] steps={steps} elapsed={elapsed:.1f}s avg_step={avg_step:.3f}s "
          f"peak_alloc={peak_alloc:.0f}MiB peak_reserved={peak_res:.0f}MiB", flush=True)
    if args.dry_run:
        print("[TI-FLUX] dry-run OK (nothing saved)", flush=True)
        return

    meta = {
        "placeholder": args.placeholder, "num_vectors": n_vec,
        "token_id": token_id, "token_id_2": token_id_2, "init_word": args.init_word,
        "prompt": train_prompt if fixed else "",
        "prompt_mode": prompt_mode, "template_index": args.template_index,
        "guidance": args.guidance, "attn_mode": args.attn_mode,
        "max_sequence_length": args.max_sequence_length,
        "mask_project": args.mask_project, "steps": args.steps, "size": args.size,
    }
    torch.save({"vec_clip": vec_clip.detach().cpu(), "vec_t5": vec_t5.detach().cpu(), **meta},
               out / "flux_token_embedding.pt")
    torch.save({"vec_clip": ema_clip.cpu(), "vec_t5": ema_t5.cpu(), **meta},
               out / "flux_token_embedding_ema.pt")
    manifest = {
        **{k: v for k, v in meta.items()},
        "model": args.model, "images": args.image, "mask_image": args.mask_image,
        "crops_per_image": args.crops_per_image, "dynamic_data_enhance": bool(args.data_enhance),
        "mask_loss": args.mask_loss, "mask_token_reduce": args.mask_token_reduce,
        "data_enhance": bool(args.data_enhance),
        "lr": args.lr, "seed": args.seed, "dtype": args.dtype, "vae_dtype": args.vae_dtype,
        "timestep_sampling": args.timestep_sampling, "logit_mean": args.logit_mean,
        "logit_std": args.logit_std, "mu": mu, "image_seq_len": image_seq_len,
        "height_lat": height_lat, "width_lat": width_lat,
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "ema_decay": args.ema_decay, "empty_mask_steps": empty_mask_steps,
        "elapsed_sec": round(elapsed, 2), "avg_step_sec": round(avg_step, 4),
        "loss_first": losses[0], "loss_last": losses[-1],
        "loss_mean": float(np.mean(losses)),
        "peak_vram_allocated_mb": round(peak_alloc, 1),
        "peak_vram_reserved_mb": round(peak_res, 1),
        "note": "only placeholder rows of CLIP-L and T5-XXL are trainable",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[TI-FLUX] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
