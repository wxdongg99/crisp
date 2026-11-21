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
