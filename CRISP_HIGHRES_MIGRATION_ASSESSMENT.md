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
