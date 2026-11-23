# CRISP → high-resolution migration assessment

**Date**: 2026-11-22  
**Scope**: decision record only; no CRISP training code was changed.

## Why migration is not immediate

Current CRISP stages are tightly coupled to:

```text
LDM UNet f8, 4 latent channels
BERT text encoder
TI embedding manager
PSP spatial encoder
Stage-0 / Stage-A / Stage-B residual v10
```

The high-res candidates differ materially:

| model | latent | text encoder | denoiser | mask/PSP port |
|---|---|---|---|---|
| SDXL | f8, 4ch | dual CLIP | SDXL UNet | **easiest**, UNet + inpaint/ControlNet |
| PixArt-Σ | f8, 4ch | T5 | DiT | token/conditioning rewrite |
| SD3.5 | 16ch VAE | CLIP+T5 | MMDiT | major rewrite |
| FLUX.1 | 16ch VAE | T5+CLIP | MMDiT/flow | major rewrite |

## Recommended order

```text
1. SDXL: TI + mask + 512/1024 high-res pipeline
2. SDXL: 1024→2048 high-res fix
3. Only then port Stage-A (TI init) to SDXL UNet
4. Stage-B branch-mask residual after Stage-A is stable
5. PixArt only as native high-res quality comparison
```

## Migration gates

Do not start CRISP Stage-A/B port until all pass:

- SDXL token TI save/reload + fixed-seed quality check
- SDXL mask background preservation at 512/1024/2048
- fixed-seed SDXL-vs-PixArt mask comparison
- peak VRAM and wall-clock budget measured

## Risks

- SDXL dual-CLIP token vectors are not interchangeable with BERT TI.
- SDXL mask inpaint is sampler-level; it is not PSP.
- PixArt T5 TI quality at single-image budget is currently poor.
- High-res Stage-B training cost may exceed the current A100 quota.
- Old CRISP checkpoints are not reusable after a text-encoder/latent change.

## Current decision

```text
Keep crisp-cvpr stable.
Keep highres-exploration isolated.
Continue SDXL as the primary migration candidate.
Do not migrate Stage-A/B yet.
```

## Update 2026-11-23: new high-res evidence

### Per-pattern dynamic SDXL TI

| pattern | setup | result |
|---|---|---|
| 16 | dynamic `random_transform`, raw mask loss, 4800 steps | coherent fabric texture, no defect |
| 17 | dynamic + LFP v3 loss support, 4800 steps | seed0 localized blue/white blob; seeds 1–2 noisy |
| 18 | dynamic raw mask loss, 4800 steps | textured background with colored blobs, not realistic defects |
| 19 | dynamic raw mask loss, 4800 steps | in progress |

Conclusion: token-only SDXL TI still does not learn a transferable anomaly even
with correct image/mask-synchronized dynamic augmentation and 800 views/image.

### Mask generation

- PixArt 1024 CFG=4.5 with SDPA/bf16 is clearly better than SDXL mask on the
  fixed-seed stain comparison.
- PixArt 1024 CFG FP8 inference works at ~3.2 GB peak VRAM.
- PixArt FP8 training fails on 4090 in torchao float8 matmul.
- LFP v3 nearest-point support matches v1 area; composite confines edit to the
  fine-mask region, but generated content is still a bright artifact.

### Revised migration gate

Do **not** port Stage-B yet. Before any CRISP Stage-A/B work:

1. mask-covered region must produce a plausible defect, not bright bands;
2. SDXL or PixArt must beat the other on a fixed-seed, same-mask benchmark;
3. per-pattern tokens must show transfer on held-out images;
4. FP8 training must be validated on supported hardware or dropped.

If migration resumes, start with SDXL + LFP v3 loss support, not PixArt: SDXL
keeps the f8/4ch latent and 4-channel UNet path closest to CRISP Stage-A, while
PixArt needs T5/DiT and token/conditioning rewrites.
