#!/usr/bin/env python3
"""Qwen-Image-2.1 token textual inversion (single placeholder, multi-vector, CRISP mask loss).

Reference implementation details are taken from the diffusers sources, not guessed:

* ``diffusers.pipelines.qwenimage21.pipeline_qwenimage21.QwenImage21Pipeline``
  - prompt template: ``prompt_template_t2i`` (raw string, *not* ``apply_chat_template``)
  - ``_drop_idx`` leading system-role tokens are removed from the text-encoder states
  - ``hidden_states[-1]`` is read with the text encoder's final RMSNorm neutralised by a
    forward hook (transformers>=5 ties ``hidden_states[-1]`` to ``last_hidden_state``)
  - the transformer consumes ``hidden_states`` = packed latents ``(B, h*w, 64)``, plus
    ``img_shapes``/``img_mask``; the timestep handed to the transformer is ``sigma`` in [0, 1]
* ``diffusers.models.transformers.transformer_qwenimage21.QwenImage21Transformer2DModel``
  - one vision slot stands for a 2x2 group of latent tokens, so ``img_mask`` gets
    ``h*w//4`` trailing ``True`` slots for the target image
* ``diffusers.models.autoencoders.autoencoder_kl_qwenimage21.AutoencoderKLQwenImage21``
  - 4-channel input (RGB + alpha), ``latents_mean``/``latents_std`` normalisation, 16x spatial
  - latents keep the temporal axis: encode ``(B, 4, 1, H, W)`` -> ``(B, 64, 1, h, w)``

Only the ``num_vectors`` learned token vectors are trained; text encoder, transformer and VAE
stay frozen. LFP v3 mask projection is the CRISP version from ``highres/lfp_mask.py`` adapted
to the 64-channel Qwen VAE (``lfp_mask.py`` itself is reused unmodified for the shared parts).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from highres.prompt_templates import canonical_prompt, load_templates, random_prompt  # noqa: E402
from highres.lfp_mask import _nearest_dilate_to_area, _otsu  # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


# --------------------------------------------------------------------------------------
# prompt / text encoder (Qwen3-VL)
# --------------------------------------------------------------------------------------
def build_pipeline_shell(scheduler, vae, text_encoder, processor, transformer):
    """Instantiate the real pipeline object *without* loading weights again.

    The pipeline is used as the single source of truth for the prompt template, the number of
    leading system tokens to drop and the processor call, so this script cannot drift from it.
    """
    from diffusers import QwenImage21Pipeline

    return QwenImage21Pipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        processor=processor,
        transformer=transformer,
    )


def encode_prompt_embeds(
    text_encoder,
    processor,
    pipe,
    prompt: str,
    device,
    placeholder: str,
    placeholder_id: int | None,
    vec: torch.Tensor | None,
    num_vectors: int = 0,
):
    """Encode one prompt, splicing ``num_vectors`` learned vectors at the placeholder.

    Mirrors ``QwenImage21Pipeline._get_qwen_prompt_embeds`` (raw template string, left padding,
    ``hidden_states[-1]`` with the final RMSNorm neutralised, leading system tokens dropped).
    The placeholder token only exists in the tokenizer, never in the model vocabulary: its row
    is replaced by ``vec`` so ``embed_tokens`` is never called with an out-of-range id.
    """
    prompt = prompt if prompt else " "
    # The pipeline wraps the prompt in its raw template string before tokenizing.
    full_prompt = pipe.prompt_template_t2i.format(prompt)
    model_inputs = processor(
        text=[full_prompt], padding=True, padding_side="left", return_tensors="pt"
    ).to(device)
    ids = model_inputs.input_ids
    attn = model_inputs.attention_mask
    if ids.shape[0] != 1:
        raise ValueError("encode_prompt_embeds only supports a single prompt")

    insert = vec is not None and num_vectors > 0
    if insert:
        pos = (ids[0] == placeholder_id).nonzero(as_tuple=False).flatten()
        if pos.numel() != 1:
            raise ValueError(
                f"expected exactly one placeholder {placeholder!r} token, found {pos.numel()}"
            )
        p = int(pos[0])
    embed = text_encoder.get_input_embeddings()
    safe_ids = ids.clone()
    if insert:
        safe_ids[ids == placeholder_id] = 0
    inputs_embeds = embed(safe_ids)
    if insert:
        inputs_embeds = torch.cat(
            [inputs_embeds[:, :p], vec.to(inputs_embeds.dtype).unsqueeze(0),
             inputs_embeds[:, p + 1:]],
            dim=1,
        )
        attn = torch.cat(
            [attn[:, :p], attn.new_ones(1, num_vectors), attn[:, p + 1:]], dim=1
        )

    # The pipeline neutralises the text encoder's final RMSNorm so that hidden_states[-1] is the
    # last decoder layer output. transformers>=5 ties hidden_states[-1] to last_hidden_state.
    text_model = getattr(text_encoder.model, "language_model", text_encoder.model)
    handle = text_model.norm.register_forward_hook(lambda module, args, output: args[0])
    try:
        outputs = text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            output_hidden_states=True,
        )
    finally:
        handle.remove()

    hidden = outputs.hidden_states[-1]
    # Batch of one with left padding: trim padding rows, then drop the leading system tokens.
    valid = hidden[attn.bool()].unsqueeze(0)
    hidden = valid[:, pipe._drop_idx:]
    # Text-only conditioning: the pipeline returns None when nothing is padded.
    enc_mask = None
    return hidden, enc_mask


def image_pad_mask(num_text: int, num_latent: int, batch: int, device) -> torch.Tensor:
    """``img_mask`` for a text-only sample with one 2x2-slot group per 4 target latents."""
    slots = num_latent // 4
    mask = torch.zeros(batch, num_text + slots, dtype=torch.bool, device=device)
    mask[:, num_text:] = True
    return mask


def transformer_velocity(
    transformer,
    latents_packed: torch.Tensor,
    sigma: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_mask: torch.Tensor | None,
    latent_h: int,
    latent_w: int,
):
    """One QwenImage21Transformer2DModel forward, matching ``QwenImage21Pipeline.__call__``."""
    batch, num_latent, channels = latents_packed.shape
    if num_latent != latent_h * latent_w:
        raise ValueError(f"packed latent length {num_latent} != {latent_h}x{latent_w}")
    out = transformer(
        hidden_states=latents_packed,
        timestep=sigma.to(latents_packed.dtype).expand(batch),
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        img_shapes=[[(1, latent_h, latent_w)]] * batch,
        img_mask=image_pad_mask(
            encoder_hidden_states.shape[1], num_latent, batch, latents_packed.device
        ),
        return_dict=False,
    )[0]
    # The joint sequence is [text, target]; the pipeline keeps the trailing target tokens.
    return out[:, -num_latent:]


# --------------------------------------------------------------------------------------
# VAE helpers (64-channel, 16x, temporal axis of length 1)
# --------------------------------------------------------------------------------------
def vae_latents(vae, image: torch.Tensor) -> torch.Tensor:
    """``image`` in [-1, 1], shape ``(B, 4, H, W)`` -> normalised latents ``(B, 64, h, w)``."""
    param = next(vae.parameters())
    image = image.to(device=param.device, dtype=param.dtype)
    stat_shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=image.device, dtype=image.dtype).view(
        stat_shape
    )
    std = torch.tensor(vae.config.latents_std, device=image.device, dtype=image.dtype).view(
        stat_shape
    )
    with torch.no_grad():
        z = vae.encode(image.unsqueeze(2)).latent_dist.mode()
        z = (z - mean) / std
    return z[:, :, 0]


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``(B, C, h, w)`` -> ``(B, h*w, C)`` (2.1 consumes latents unpatched)."""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).transpose(1, 2)


def unpack_latents(latents: torch.Tensor, h: int, w: int) -> torch.Tensor:
    b, _, c = latents.shape
    return latents.transpose(1, 2).reshape(b, c, h, w)


# --------------------------------------------------------------------------------------
# LFP v3 for the Qwen VAE (lfp_mask.py's helpers reused; the VAE probe is adapted to 64ch/16x)
# --------------------------------------------------------------------------------------
def _qwen_lfp_latent(mask: torch.Tensor, vae, target_h: int, target_w: int) -> torch.Tensor:
    """LFP v1 support at latent resolution for one binary mask, ``mask`` of shape ``(H, W)``.

    The reference code decides the polarity per channel, which works for a 4-channel VAE. The
    Qwen VAE has 64 latent channels and a per-channel union saturates (every cell gets covered
    by some channel's complement). Instead the response magnitude ``mean_c |z_c|`` is split by
    Otsu, the polarity is chosen by correlation with the max-pooled mask footprint, and the true
    footprint is always included. Trim / bilinear merge / hole fill are then the same as
    ``highres/lfp_mask.py``.
    """
    m = (mask[None, None] >= 0.5).float()
    bil = F.interpolate(m, size=(target_h, target_w), mode="bilinear", align_corners=False)
    rgb = (m * 2.0 - 1.0).expand(-1, 3, -1, -1).clone()
    alpha = torch.ones_like(rgb[:, :1])  # Qwen VAE reads RGB + alpha, opaque images use +1
    probe = torch.cat([rgb, alpha], dim=1).to(
        next(vae.parameters()).device, next(vae.parameters()).dtype
    )
    with torch.no_grad():
        z = vae.encode(probe.unsqueeze(2)).latent_dist.mode()[:, :, 0]
    resp = z[0].detach().float().cpu().numpy().__abs__().mean(axis=0)  # (th, tw)
    footprint = (
        F.max_pool2d(m, kernel_size=mask.shape[0] // target_h, stride=mask.shape[0] // target_h)
        [0, 0].cpu().numpy() > 0.5
    )
    lo, hi = float(resp.min()), float(resp.max())
    if hi - lo <= 1e-6:
        support = footprint.copy()
    else:
        thr = _otsu((resp - lo) / (hi - lo))
        support = resp > lo + thr * (hi - lo)
        if (support & footprint).sum() < 0.5 * support.sum():
            support = ~support
        support = support | footprint
    final = support.astype(np.float64)
    final[0, :] = 0
    final[-1, :] = 0
    final[:, 0] = 0
    final[:, -1] = 0
    final = np.maximum(final, bil[0, 0].cpu().numpy())
    while True:  # 4-neighbour hole filling
        p = np.pad(final, 1, constant_values=0)
        neigh = (
            (p[:-2, 1:-1] > 0.5).astype(int)
            + (p[2:, 1:-1] > 0.5).astype(int)
            + (p[1:-1, :-2] > 0.5).astype(int)
            + (p[1:-1, 2:] > 0.5).astype(int)
        )
        fill = (final < 0.5) & (neigh >= 3)
        if not fill.any():
            break
        final[fill] = 1.0
    return torch.from_numpy((final >= 0.5).astype(np.float32))[None, None]


def lfp_v1_mask_qwen(mask_pil: Image.Image, vae, vae_scale: int = 16) -> Image.Image:
    w, h = mask_pil.size
    m = torch.from_numpy(np.array(mask_pil.convert("L"), dtype=np.float32)) / 255.0
    th, tw = h // vae_scale, w // vae_scale
    latent = _qwen_lfp_latent(m, vae, th, tw)
    pix = F.interpolate(latent, size=(h, w), mode="nearest")[0, 0].cpu().numpy()
    return Image.fromarray((pix * 255).astype(np.uint8), mode="L")


_LFP_V3_AREA_RATIO: dict[str, float] = {}


def lfp_area_ratio_qwen(mask_pil: Image.Image, vae, vae_scale: int = 16,
                        cache_key: str = "") -> float:
    """LFP v1 area / raw mask area, cached per source mask.

    Measured on the (small) augmented mask so the per-step VAE probe stays cheap; the ratio is
    resolution independent enough to set the v3 target area at the training resolution.
    """
    if cache_key and cache_key in _LFP_V3_AREA_RATIO:
        return _LFP_V3_AREA_RATIO[cache_key]
    raw_area = int((np.asarray(mask_pil.convert("L")) > 127).sum())
    v1 = lfp_v1_mask_qwen(mask_pil, vae, vae_scale)
    v1_area = int((np.asarray(v1) > 127).sum())
    ratio = v1_area / max(raw_area, 1)
    if cache_key:
        _LFP_V3_AREA_RATIO[cache_key] = ratio
    return ratio


def lfp_v3_mask_qwen(
    mask_pil: Image.Image, vae, vae_scale: int = 16, cache_key: str = "",
    ratio: float | None = None,
) -> Image.Image:
    """LFP v3: max-pool support of the raw mask, nearest-dilated to the LFP v1 area."""
    if ratio is None:
        ratio = lfp_area_ratio_qwen(mask_pil, vae, vae_scale, cache_key)
    raw_area = int((np.asarray(mask_pil.convert("L")) > 127).sum())
    target = int(round(raw_area * ratio))
    return _nearest_dilate_to_area(mask_pil, target, vae_scale)


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
def load_dynamic_pair(img_path, mask_path, size, device, vae=None, mask_project="lfp_v3",
                      cache_key=""):
    """CRISP-style dynamic augmentation, applied identically to image and mask."""
    img = Image.open(img_path).convert("RGB").resize((512, 512), Image.Resampling.BICUBIC)
    if mask_path is None:
        it = transforms.ToTensor()(img.resize((size, size), Image.Resampling.BICUBIC)).unsqueeze(0)
        it4 = torch.cat([it * 2 - 1, torch.ones_like(it[:, :1])], dim=1)
        return it4.to(device), None
    mask = Image.open(mask_path).convert("L").resize((512, 512), Image.Resampling.NEAREST)
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
        cx1 = random.randint(0, max(0, int(x0)))
        cy1 = random.randint(0, max(0, int(y0)))
        cx2 = random.randint(min(511, int(x1)), 511)
        cy2 = random.randint(min(511, int(y1)), 511)
        if cx2 > cx1 + 32 and cy2 > cy1 + 32:
            img = img.crop((cx1, cy1, cx2, cy2)).resize((size, size), Image.Resampling.BICUBIC)
            mask = mask.crop((cx1, cy1, cx2, cy2)).resize(
                (size, size), Image.Resampling.NEAREST
            )
    # LFP v1 area ratio is measured on the augmented 512 mask (one-off, cached per source mask),
    # then the v3 support is built at the training resolution.
    ratio = lfp_area_ratio_qwen(mask, vae, vae_scale=16, cache_key=cache_key)
    img = img.resize((size, size), Image.Resampling.BICUBIC)
    mask = (np.asarray(mask.resize((size, size), Image.Resampling.NEAREST)) > 127).astype(
        np.uint8
    ) * 255
    mask = Image.fromarray(mask, mode="L")
    if mask_project == "lfp_v3":
        mask = lfp_v3_mask_qwen(mask, vae, vae_scale=16, ratio=ratio)
    it = transforms.ToTensor()(img).unsqueeze(0)
    mt = (transforms.ToTensor()(mask).unsqueeze(0) > 0.5).float()
    it4 = torch.cat([it * 2 - 1, torch.ones_like(it[:, :1])], dim=1)  # RGB + opaque alpha
    return it4.to(device), mt.to(device)


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def shift_sigma(sigma: torch.Tensor, mu: float) -> torch.Tensor:
    """Same exponential time shift as FlowMatchEulerDiscreteScheduler.time_shift."""
    return math.exp(mu) / (math.exp(mu) + (1.0 / sigma - 1.0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data/solo/wxd/models/Qwen-Image-2.1")
    p.add_argument("--image", required=True, nargs="+")
    p.add_argument("--mask-image", nargs="*", default=[])
    p.add_argument("--mask-project", default="lfp_v3", choices=["raw", "lfp_v3"])
    p.add_argument("--placeholder", default="<anomaly>")
    p.add_argument("--init-word", "--init_word", dest="init_word", default="anomaly")
    p.add_argument("--init-embedding", default="")
    p.add_argument("--num-vectors", type=int, default=4)
    p.add_argument("--fixed-prompt", action="store_true",
                   help="use only the canonical imagenet template (index) instead of a random one")
    p.add_argument("--template-index", type=int, default=0)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--mask-loss", action="store_true", default=True,
                   help="masked flow-matching loss (on by default, as CRISP uses it)")
    p.add_argument("--no-mask-loss", dest="mask_loss", action="store_false")
    p.add_argument("--embedding-reg-weight", type=float, default=0.0)
    p.add_argument("--sigma-shift", default="auto",
                   help="'auto' = calculate_shift() at this token count, 'none' = 1.0, or a float")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--save-every", type=int, default=0,
                   help="also checkpoint every N steps (0 = only at the end)")
    p.add_argument("--grad-ckpt", action="store_true", default=True)
    p.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    p.add_argument("--vae-tiling", action="store_true")
    p.add_argument("--verify-encode", action="store_true",
                   help="compare this script's encoder against QwenImage21Pipeline._get_qwen_prompt_embeds")
    p.add_argument("--output_dir", default="textual_inversion_runs/qwen21_token")
    args = p.parse_args()

    if args.mask_image and len(args.mask_image) != len(args.image):
        raise ValueError("mask-image count must match image count")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
    from diffusers import QwenImage21Transformer2DModel
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    t_load = time.time()
    text_encoder = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, subfolder="text_encoder", torch_dtype=dtype
        )
        .to(args.device)
        .eval()
    )
    transformer = (
        QwenImage21Transformer2DModel.from_pretrained(
            args.model, subfolder="transformer", torch_dtype=dtype
        )
        .to(args.device)
        .eval()
    )
    vae = (
        AutoencoderKLQwenImage21.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
        .to(args.device)
        .eval()
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model, subfolder="scheduler")
    processor = Qwen3VLProcessor.from_pretrained(args.model, subfolder="processor")
    if args.vae_tiling:
        vae.enable_tiling()
    for m in (text_encoder, transformer, vae):
        m.requires_grad_(False)
    if args.grad_ckpt:
        transformer.enable_gradient_checkpointing()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        mem_after_load = torch.cuda.memory_allocated() / 2**30
    else:
        mem_after_load = 0.0
    pipe = build_pipeline_shell(scheduler, vae, text_encoder, processor, transformer)
    print(f"[TI-Qwen21] models loaded in {time.time() - t_load:.1f}s "
          f"(vae_scale_factor={pipe.vae_scale_factor}, drop_idx={pipe._drop_idx}, "
          f"resident={mem_after_load:.1f} GiB)", flush=True)

    # ---- placeholder: tokenizer-only token, learned rows replace it in the embedding output ----
    tokenizer = processor.tokenizer
    # The Qwen tokenizer is shorter than the model vocabulary (reserved slots), so a new token id
    # still lands inside the embedding table; its row is never used because we splice `vec` in.
    n_added = tokenizer.add_tokens([args.placeholder])
    if n_added != 1:
        raise ValueError(f"placeholder {args.placeholder!r} is already a token")
    placeholder_id = tokenizer.convert_tokens_to_ids(args.placeholder)
    init_ids = tokenizer(args.init_word, add_special_tokens=False).input_ids
    if not init_ids:
        raise ValueError(f"init word {args.init_word!r} tokenizes to nothing")
    embedding = text_encoder.get_input_embeddings()
    with torch.no_grad():
        init_vec = embedding.weight[torch.tensor(init_ids, device=embedding.weight.device)]
    if args.init_embedding:
        ckpt = torch.load(args.init_embedding, map_location="cpu")
        vec = torch.nn.Parameter(ckpt["vec"].float().to(args.device).clone())
        n_vec = vec.shape[0]
        print(f"[TI-Qwen21] resumed {args.init_embedding} n_vec={n_vec}", flush=True)
    else:
        n_vec = int(args.num_vectors)
        # Mean of the init word's sub-token rows, repeated: the token starts out reading as the
        # init word and learns its own multi-vector codes.
        base = init_vec.mean(dim=0)
        vec = torch.nn.Parameter(base.repeat(n_vec, 1).clone())
        print(f"[TI-Qwen21] init_word={args.init_word!r} init_ids={init_ids} "
              f"placeholder_id={placeholder_id} n_vec={n_vec}", flush=True)
    vec.requires_grad_(True)
    initial_vec = vec.detach().clone()
    max_vec_norm = float(initial_vec.norm().item()) * 2.0

    # ---- optional fidelity check against the pipeline's own prompt encoder ----
    if args.verify_encode:
        prompt = canonical_prompt(args.init_word, args.template_index)
        with torch.no_grad():
            h_mine, _ = encode_prompt_embeds(
                text_encoder, processor, pipe, prompt, args.device, args.placeholder,
                placeholder_id, None, 0,
            )
            h_ref, ref_mask, _ = pipe._get_qwen_prompt_embeds([prompt], None, args.device)
        diff = (h_mine.float() - h_ref.float()).abs().max().item()
        print(f"[TI-Qwen21] verify-encode prompt={prompt!r} shapes "
              f"{tuple(h_mine.shape)} vs {tuple(h_ref.shape)} max_abs_diff={diff:.3e}",
              flush=True)
        if diff > 2e-2:
            raise RuntimeError(f"encoder mismatch vs pipeline: {diff}")

    templates = load_templates()
    rng = random.Random(args.seed)
    opt = torch.optim.AdamW([vec], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    latent_h = args.size // pipe.vae_scale_factor
    latent_w = args.size // pipe.vae_scale_factor
    num_latent = latent_h * latent_w
    # sampler shift for this token count, exactly as QwenImage21Pipeline computes mu
    mu = None
    if args.sigma_shift == "auto":
        base_seq, max_seq, base_shift, max_shift = 256, 8192, 0.5, 0.9
        m = (max_shift - base_shift) / (max_seq - base_seq)
        mu = num_latent * m + (base_shift - m * base_seq)
    elif args.sigma_shift != "none":
        mu = float(args.sigma_shift)
    print(f"[TI-Qwen21] size={args.size} latents={latent_h}x{latent_w} tokens={num_latent} "
          f"mu={mu if mu is not None else 1.0:.4f}", flush=True)

    if (args.mask_project == "lfp_v3" or args.mask_loss) and not args.mask_image:
        raise ValueError("--mask-loss / --mask-project lfp_v3 need --mask-image")

    t0 = time.time()
    peak = 0.0
    losses = []
    step_times = []
    try:
        for step in range(1, args.steps + 1):
            t_step = time.time()
            j = (step - 1) % len(args.image)
            img, mask_t = load_dynamic_pair(
                args.image[j], args.mask_image[j] if args.mask_image else None,
                args.size, args.device, vae=vae, mask_project=args.mask_project,
                cache_key=str(args.mask_image[j]) if args.mask_image else "",
            )
            lat = vae_latents(vae, img)  # (1, 64, h, w)
            packed = pack_latents(lat)

            if args.fixed_prompt:
                train_prompt = canonical_prompt(args.placeholder, args.template_index)
            else:
                train_prompt = random_prompt(args.placeholder, rng)
            hidden, enc_mask = encode_prompt_embeds(
                text_encoder, processor, pipe, train_prompt, args.device, args.placeholder,
                placeholder_id, vec, n_vec,
            )

            noise = torch.randn_like(packed)
            sigma = torch.rand(1, device=args.device)
            if mu is not None:
                sigma = shift_sigma(sigma, mu)
            sigma = sigma.clamp(1e-4, 1.0 - 1e-4)
            sigma_lat = sigma.to(packed.dtype)  # keep the blend in the latent dtype
            noisy = sigma_lat * noise + (1.0 - sigma_lat) * packed
            target = noise - packed

            pred = transformer_velocity(
                transformer, noisy, sigma.reshape(1), hidden, enc_mask, latent_h, latent_w
            )
            if args.mask_loss and mask_t is not None:
                # latents are packed (B, h*w, C) -> the mask must be a per-token weight (B, h*w, 1)
                ml = F.interpolate(
                    mask_t.float(), size=(latent_h, latent_w), mode="nearest"
                ).reshape(1, -1, 1)
                per = (pred.float() - target.float()) ** 2 * ml
                loss = per.sum() / (ml.sum() * pred.shape[1]).clamp_min(1.0)
            else:
                loss = F.mse_loss(pred.float(), target.float())
            diff = vec - initial_vec
            reg = (diff @ diff.T / max(1, diff.shape[0])).mean()
            loss = loss + args.embedding_reg_weight * reg

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([vec], 1.0)
            opt.step()
            with torch.no_grad():
                norm = float(vec.norm().item())
                if norm > max_vec_norm:
                    vec.mul_(max_vec_norm / norm)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(time.time() - t_step)
            losses.append(loss.item())
            peak = max(peak, torch.cuda.max_memory_allocated() / 2**30 if
                       torch.cuda.is_available() else 0.0)
            if args.save_every and step % args.save_every == 0 and step != args.steps:
                torch.save(
                    {
                        "vec": vec.detach().cpu(),
                        "placeholder": args.placeholder,
                        "num_vectors": n_vec,
                        "init_word": args.init_word,
                        "prompt_mode": "canonical" if args.fixed_prompt else "imagenet_templates_small",
                        "template_index": args.template_index,
                        "model": args.model,
                        "size": args.size,
                        "steps_done": step,
                    },
                    out / "qwen21_token_embedding.pt",
                )
                print(f"[TI-Qwen21] checkpoint step={step} -> {out}/qwen21_token_embedding.pt",
                      flush=True)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                if step == 1:
                    print(f"[TI-Qwen21] prompt={train_prompt!r}", flush=True)
                avg_t = sum(step_times[1:]) / max(1, len(step_times) - 1)
                print(
                    f"[TI-Qwen21] step={step:4d} loss={loss.item():.6f} reg={reg.item():.6f} "
                    f"norm={vec.norm().item():.2f} sigma={sigma.item():.3f} "
                    f"step_s={step_times[-1]:.2f} avg_s={avg_t:.2f} "
                    f"peak_GiB={peak:.1f}",
                    flush=True,
                )
    except torch.cuda.OutOfMemoryError as exc:
        info = {
            "oom": True,
            "error": str(exc),
            "size": args.size,
            "latent_hw": [latent_h, latent_w],
            "tokens": num_latent,
            "peak_GiB": torch.cuda.max_memory_allocated() / 2**30,
            "reserved_GiB": torch.cuda.max_memory_reserved() / 2**30,
            "total_GiB": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "steps_done": len(losses),
        }
        (out / "oom_report.json").write_text(json.dumps(info, indent=2))
        print(f"[TI-Qwen21] OOM after {len(losses)} steps: {info}", flush=True)
        raise SystemExit(2)

    elapsed = time.time() - t0
    torch.save(
        {
            "vec": vec.detach().cpu(),
            "placeholder": args.placeholder,
            "num_vectors": n_vec,
            "init_word": args.init_word,
            "prompt_mode": "canonical" if args.fixed_prompt else "imagenet_templates_small",
            "template_index": args.template_index,
            "model": args.model,
            "size": args.size,
            "steps_done": args.steps,
        },
        out / "qwen21_token_embedding.pt",
    )
    manifest = {
        "model": args.model,
        "placeholder": args.placeholder,
        "num_vectors": n_vec,
        "init_word": args.init_word,
        "prompt_mode": "canonical" if args.fixed_prompt else "imagenet_templates_small",
        "fixed_prompt": bool(args.fixed_prompt),
        "template_index": args.template_index,
        "mask_project": args.mask_project,
        "mask_loss": args.mask_loss,
        "steps": args.steps,
        "size": args.size,
        "latent_grid": [latent_h, latent_w],
        "num_latent_tokens": num_latent,
        "lr": args.lr,
        "dtype": args.dtype,
        "sigma_shift_mu": mu,
        "images": args.image,
        "mask_images": args.mask_image,
        "embedding_reg_weight": args.embedding_reg_weight,
        "grad_ckpt": bool(args.grad_ckpt),
        "final_loss": losses[-1] if losses else None,
        "mean_loss_last20": float(np.mean(losses[-20:])) if losses else None,
        "final_vec_norm": float(vec.norm().item()),
        "mem_after_load_GiB": mem_after_load,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "peak_GiB": peak,
        "step_sec_mean": float(np.mean(step_times[1:])) if len(step_times) > 1 else None,
        "step_sec_last": step_times[-1] if step_times else None,
        "elapsed_sec": round(elapsed, 2),
        "text_encoder": "Qwen3VLForConditionalGeneration",
        "transformer": "QwenImage21Transformer2DModel",
        "vae": "AutoencoderKLQwenImage21",
        "diffusers_version": __import__("diffusers").__version__,
        "transformers_version": __import__("transformers").__version__,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[TI-Qwen21] wrote {out} ({elapsed:.1f}s)", flush=True)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
