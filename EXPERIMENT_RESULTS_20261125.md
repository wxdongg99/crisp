# new_crisp ZJU Stage-A / Stage-B reproduction results

Date: 2026-11-25

## Status

- Stage-0 (new_crisp, nobgsub): complete, 4 bg tokens, no bg_sub tokens.
- Stage-A 40k on new Stage-0: complete (`embeddings_gs-39999.pt`).
- Stage-A B300 regeneration: complete, 504 images x 4 patterns, `cfg_a=3.0`, LFP, uniform base rotation.
- Stage-A 5-seed classification: running on B300 GPU7.
- Stage-B 40k: complete (`embeddings_gs-39999.pt`).
- Stage-B arbitrary `cfg_a` / `cfg_b` inference: implemented; 2-image smoke test passed.
- Stage-B full generation + classification: not started yet.

## Stage-A classification

### Previous A100 result (already completed)

| seed | Acc |
|---|---:|
| 42 | 81.25 |
| 7 | 85.42 |
| 19 | 83.33 |
| 0 | 81.25 |
| 1 | 81.25 |

```
mean 82.50, std 1.86
```

Old CRISP Stage-A reference: `76.25 +/- 5.02`.

### Fresh B300 regeneration

Same Stage-A 40k checkpoint and final generation code, newly generated pool:

```
/pfs/pfs-7jnepv/solo/wxd/new_crisp/eval_runs/stageA40k_regen/
```

Current partial results:

| seed | Acc |
|---|---:|
| 42 | 83.33 |
| 7 | running |
| 19 | pending |
| 0 | pending |
| 1 | pending |

## Stage-B

- Training run: `new_crisp_zju_stageB_40k_s0new_v1`
- Steps: 40,000
- Final checkpoint: `embeddings_gs-39999.pt`
- Retained backup: local A100 archive and B300 `checkpoints_stageB40k`.

Stage-B inference formula now implemented:

```
eps = eps_uc
      + cfg_a * (eps_A - eps_uc)
      + cfg_b * eps_R
```

where:

```
eps_A = UNet(x_t, t, cond_a)
eps_R = UNet(x_t, t, cond_b)
eps_uc = UNet(x_t, t, empty)
```

Smoke test:

```
cfg_a=3.0, cfg_b=1.0, 2 images, rc=0
```

## Paths

- Project (local B300): `/data/solo/wxd/projects/new_crisp`
- Project (U2 local): `/home/cc/Documents/new_crisp`
- B300 storage: `/pfs/pfs-7jnepv/solo/wxd/new_crisp`
- Stage-A fresh generation: `/pfs/pfs-7jnepv/solo/wxd/new_crisp/eval_runs/stageA40k_regen/generated_dataset`
- Stage-A eval: `/pfs/pfs-7jnepv/solo/wxd/new_crisp/eval_runs/stageA40k_regen/eval_cls40`

## Notes

- The previous A100 Stage-A run is archived locally.
- Fresh B300 5-seed classification is being collected; this file should be updated with the final values.
