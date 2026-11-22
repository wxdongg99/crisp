# High-resolution exploration changes

Branch: `highres-exploration`

## 2026-11-21 highres-v1

### Scope

Standalone PixArt-Σ probe for native 512/1024/2048 generation.

### Added

- `highres/pixart_sigma_generate.py`
- `highres/prompts/zju_fabric.txt`
- `highres/requirements-highres.txt`
- `highres/README.md`

### Not changed

- CRISP LDM Stage-A/Stage-B code
- Stable branch `crisp-cvpr`
- Existing model repository

### Next

1. Run 1024 smoke test and inspect quality.
2. Run 2048 smoke test with CPU offload if needed.
3. Compare with SDXL 1024 + high-res fix before any migration decision.

## 2026-11-21 highres-v2 smoke results

### Model staging

PixArt-Sigma weights staged locally at:

```text
/media/cc/U2-Server/crisp-storage/models/pixart_sigma_1024
```

Source split:

- T5 text encoder, tokenizer, VAE, scheduler: ModelScope
  `AI-ModelScope/pixart_sigma_sdxlvae_T5_diffusers`
- transformer + model_index: HuggingFace mirror
  `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`

Environment:

```text
/media/cc/U2-Server/crisp-storage/envs/highres_venv
```

Required compatibility fix: `kornia>=0.7.3` (old 0.6.0 lacks
`build_laplacian_pyramid` needed by current diffusers).

### Smoke outputs

| resolution | output | images | steps | status |
|---|---|---:|---:|---|
| 512 | `highres_runs/pixart_sigma/smoke_512` | 1 | 10 | OK |
| 1024 | `highres_runs/pixart_sigma/smoke_1024` | 1 | 10 | OK |
| 2048 | `highres_runs/pixart_sigma/smoke_2048` | 1 | 10 | OK, CPU offload + attention slicing |

### Fabric prompt runs

| resolution | output | images | prompts | steps |
|---|---|---:|---:|---:|
| 1024 | `highres_runs/pixart_sigma/fabric_1024` | 8 | 4 | 20 |
| 2048 | `highres_runs/pixart_sigma/fabric_2048` | 4 | 4 | 20 |

All images are under:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_sigma/
```

### Observation

Native 2048 is feasible on one local GPU with CPU offload + attention slicing;
1024 runs comfortably without offload. Next comparison should be against SDXL
1024 + high-res fix, plus a mask-conditioned PixArt probe.

## 2026-11-22 native-resolution correction

### Fix

- `highres/pixart_sigma_generate.py` now defaults to
  `use_resolution_binning=False`.
- Manifests now record requested/denoise/latent/output sizes,
  `vae_scale_factor`, `transformer_sample_size`, and the binning flag.

### Native 2048 results

| run | transformer | denoise | latent | output |
|---|---|---|---:|---:|---:|
| `native2048_1024weights` | 1024 weights, sample_size=128 | 2048² | 256² | 2048² |
| `native2048_official2k` | official 2K weights, sample_size=256 | 2048² | 256² | 2048² |

The 1024-weight 2048 run is visibly degraded and should not be reported as
native 2K capability. The official 2K transformer produces a materially better
texture at the same requested resolution.

Comparison image:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_sigma/
  native2048_1024weights_vs_2k.jpg
```

### Download provenance

- T5 / VAE / tokenizer / scheduler: ModelScope
  `AI-ModelScope/pixart_sigma_sdxlvae_T5_diffusers`
- 2K transformer: HuggingFace mirror
  `PixArt-alpha/PixArt-Sigma-XL-2-2K-MS` (aria2c, 4 connections)
- ModelScope does not currently expose the official Sigma 2K transformer repo.
