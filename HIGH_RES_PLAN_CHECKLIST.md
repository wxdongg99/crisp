# High-resolution exploration plan checklist

Updated: 2026-11-23  
Status legend: `[ ]` todo, `[~]` in progress, `[x]` done.

## Current stage

```text
Phase: high-resolution generation / mask control explored
Primary route: SDXL (f8, 4ch) + per-pattern token + mask loss
Comparison route: PixArt-Sigma 1024 mask (CFG, SDPA, FP8)
Stage-A / Stage-B CRISP migration: paused by gate
```

Current evidence:

- SDXL per-pattern token TI with dynamic new_crisp `random_transform`,
  4800 steps, lr=5e-3, 800 views/image: texture works, defect transfer fails.
- PixArt 1024 mask with CFG=4.5 + SDPA: coherent stain generation at
  ~3.8 GB bf16 / ~3.2 GB FP8, 4–7 s.
- LFP v3 nearest-point support and fine-mask composite implemented, but the
  generated defect content is still not plausible.
- FP8 inference works; FP8 training fails on 4090 in torchao float8 matmul.

## P0. Evidence and environment

- [x] Isolated Python 3.8 env: `/media/cc/U2-Server/crisp-storage/envs/highres_py38`
- [x] Python 3.12 FP8 env: `/media/cc/U2-Server/crisp-storage/envs/highres_py312`
  (`torch 2.4.1+cu121`, `diffusers 0.36.0`, `transformers 4.46.3`,
  `torchao 0.7.0`)
- [x] Company 4090 48G overlay env and PixArt model staged
- [x] Record model download sources (ModelScope T5/VAE, HF mirror transformer)
- [~] Record exact git SHA/dependency versions/component hashes per run
- [~] Peak VRAM recorded for PixArt mask 512/1024 bf16/FP8; SDXL TI not all
- [x] Confirm local GPU0/GPU1 parallel training and company 4090 availability

## P1. True resolution validation

- [x] Disable resolution binning (`use_resolution_binning=False`)
- [x] Record requested/denoise/latent/output size in manifests
- [x] Rerun 2048 with 1024 weights: latent 256×256, output 2048
- [x] Download and validate official 2K PixArt transformer (`sample_size=256`)
- [x] Verify 512/1024 latent sizes with binning disabled
- [~] Compare 1024@2048 vs official 2K@2048 visually; not the current blocker

## P2. Token-level textual inversion

- [x] Soft-prompt TI control (`tools/textual_inversion_basic.py`)
- [x] LDM placeholder-token TI (`tools/textual_inversion_token.py`)
- [x] SDXL dual-CLIP token TI (`highres/textual_inversion_sdxl.py`)
- [x] PixArt T5 token TI smoke (`highres/textual_inversion_pixart.py`)
- [x] SDXL save/reload + generation validated
- [x] Dynamic `utils.random_transform` reuse for image/mask-synchronized aug
- [x] Masked latent loss:
  `loss = sum((pred-noise)^2 * latent_mask) / (mask_sum * C)`
- [x] Training budget: 6 source images × 800 = 4800 steps/pattern, lr=5e-3
- [x] Pattern16: raw mask loss, 4800 steps, fixed-seed generation
- [x] Pattern17: LFP v3 only as latent loss support, 4800 steps, fixed-seed
- [x] Pattern18: raw mask loss, 4800 steps, fixed-seed
- [x] Pattern19: raw mask loss, 4800 steps, fixed-seed
- [~] Held-out denoise loss / transfer metrics; not yet
- [x] Negative result recorded: token-only SDXL TI does not learn transferable defect

## P3. SDXL baseline and mask

- [x] SDXL base downloaded; refiner not used
- [x] SDXL 512/1024/2048 mask probe with background preservation
- [x] SDXL masked inpainting probe (`highres/sdxl_inpaint_mask.py`)
- [x] LFP v1 mask projection probe; v1 has border-support artifacts
- [x] LFP v3 maxpool + exact nearest-point dilation to v1 area
- [x] LFP v3 cached area-ratio speedup: 5.2 s first / 0.083 s cached
- [x] Fine-mask composite option (`--composite-mask`)
- [~] SDXL mask content quality: still bright artifacts, not defects
- [ ] SDXL 1024→2048 high-res fix baseline

## P4. PixArt baseline and mask

- [x] PixArt 512/1024 fixed-seed baseline
- [x] PixArt mask manual sampler with source-latent blend
- [x] Frozen params + `torch.set_grad_enabled(False)` inference
- [x] bf16 + SDPA (`AttnProcessor2_0`)
- [x] 1024 no-CFG and CFG=4.5 fit on 24 GB (peak ~3.8 GB)
- [x] FP8 weight-only inference on Python 3.12/torchao 0.7
- [x] PixArt 1024 CFG FP8 multi-seed 0/1/2: coherent stains
- [x] Fixed-seed SDXL-vs-PixArt 1024 mask comparison: PixArt better for stain
- [ ] PixArt per-pattern token TI (not attempted; current route is SDXL)
- [~] Two-device placement `--tx-device/--vae-device` code exists, not benchmarked

## P5. Experiment matrix

Current focused matrix, not the original 24-cell grid:

| item | SDXL | PixArt |
|---|---|---|
| 512 mask | done | done |
| 1024 mask | done | done, CFG/FP8 |
| 2048 mask | not main | native control only |
| per-pattern TI | 4 patterns, 4800 steps | no |
| LFP v3 loss | pattern17 | no |
| fixed-seed compare | done | done |

- [ ] Full 2 models × 3 resolutions × 2 TI × 2 mask grid
- [ ] 3 seeds per cell / boundary-shift prompts
- [x] Fixed-seed stain comparison at 1024
- [~] Multi-pattern fixed-seed comparison: all four patterns generated, all weak

## P6. Isolation and provenance

- [x] Work on `highres-exploration`; stable `crisp-cvpr@9686ec1` untouched
- [ ] Record checkpoint SHA + dataset manifest hash per run
- [x] External run dirs under `/media/cc/U2-Server/crisp-storage`
- [x] Preserve historical notes; append corrections (LFP v3 nearest-point,
  dynamic `random_transform`, fixed-24-pair correction)
- [ ] Do not merge branch into `crisp-cvpr`; cherry-pick verified commits only

## P7. Deliverables and tags

- [ ] `highres-resolution-v1`
- [ ] `highres-ti-contract-v1`
- [ ] `highres-sdxl-ti-v1` (negative transfer result recorded, not a quality baseline)
- [~] `highres-mask-v1`: SDXL/PixArt/LFP v3 evidence collected
- [ ] `highres-comparison-v1`
- [x] `PIXART_MEMORY_OPTIMIZATION.md`
- [x] `FP8_AND_TWO_GPU.md`
- [x] `CRISP_HIGHRES_MIGRATION_ASSESSMENT.md` updated

## P8. Risks / blockers

- [x] True 2048 inference validated with official 2K weights
- [~] Fine defect detail survives VAE encode/decode: partially checked
- [x] TI learns defect, not only fabric background: **currently fails**
- [~] SDXL/PixArt implementations comparable: same mask/prompt/seed at 1024
- [x] Mask control localizes edit: yes; **content quality still fails**
- [ ] FP8 training: blocked on 4090 torchao float8 matmul
- [ ] Multi-source PixArt components fully reproducible

## Migration gate (Stage-A/B)

Do not port CRISP Stage-A/B yet. Required before migration:

```text
1. mask region produces a plausible fabric defect, not bright bands/blobs
2. per-pattern token shows held-out transfer
3. fixed-seed SDXL vs PixArt winner is clear
4. FP8 training validated on supported hardware or abandoned
5. LFP v3 used as loss/condition support has stable gradients and no location shift
```

Recommended first migration if gate passes:

```text
SDXL + LFP v3 loss support + Stage-A mask-conditioned TI
```

not PixArt (requires T5/DiT/conditioning rewrite).

## Next actions (ordered)

1. Decide whether token-only SDXL TI route is stopped after four negative patterns.
2. Try mask-conditioned generation: LFP v3 support + defect-region composite,
   instead of free token generation.
3. Validate FP8 training on A100/H100 or newer torchao; otherwise drop FP8.
4. Only then run a minimal Stage-A port on SDXL.
