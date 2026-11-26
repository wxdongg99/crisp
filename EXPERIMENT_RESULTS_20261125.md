# new_crisp ZJU Stage-A / Stage-B reproduction results

Date: 2026-11-25

## Status

- Stage-0 (new_crisp, nobgsub): complete, 4 bg tokens, no bg_sub tokens.
- Stage-A 40k on new Stage-0: complete (`embeddings_gs-39999.pt`).
- Stage-A B300 regeneration: complete, 504 images x 4 patterns, `cfg_a=3.0`, LFP, uniform base rotation.
- Stage-A 5-seed classification: complete on B300 GPU7.
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

Fresh B300 5-seed results:

| seed | Acc |
|---|---:|
| 42 | 83.33 |
| 7 | 79.17 |
| 19 | 83.33 |
| 0 | 85.42 |
| 1 | 81.25 |

```
mean 82.50, std 2.37
```

This fresh B300 run reproduces the previous A100 mean (82.50).

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
- Fresh B300 5-seed classification is complete and recorded above.

## Stage-B full evaluation (2026-11-25)

Two Stage-B inference configurations were generated on B300, 500 images per
pattern (`pattern16`-`pattern19`), with Stage-A 40k principal frozen.

### cfg_a=3.0, cfg_b=1.0

| seed | Acc |
|---|---:|
| 42 | 72.92 |
| 7 | 75.00 |
| 19 | 85.42 |
| 0 | 79.17 |
| 1 | 79.17 |

```
mean 78.34, std 4.80
```

### cfg_a=1.0, cfg_b=1.0

| seed | Acc |
|---|---:|
| 42 | 72.92 |
| 7 | 70.83 |
| 19 | 64.58 |
| 0 | 60.42 |
| 1 | 58.33 |

```
mean 65.42, std 6.35
```

### Comparison

| Experiment | 5-seed Acc |
|---|---:|
| Old CRISP Stage-A | 76.25 +/- 5.02 |
| new_crisp Stage-A 40k | 82.50 +/- 2.37 |
| new_crisp Stage-B, cfg_a=3, cfg_b=1 | 78.34 +/- 4.80 |
| new_crisp Stage-B, cfg_a=1, cfg_b=1 | 65.42 +/- 6.35 |

### Control

A-only (`cfg_a=3`, no Stage-B) vs Stage-B with identical base image, mask and
`seed=123`, `cfg_a=3, cfg_b=0`:

```
mask/ori: bit-identical
generated images: max diff 1-2 / 255, mean diff < 0.02
```

`cfg_b=0` therefore reproduces A-only within numerical/JPEG tolerance; the
Stage-B residual path does not silently change Stage-A guidance.

### Observation

The cfg_a=1 run loses the Stage-A guidance and drops to 65.42, close to the
earlier Stage-A-at-cfg1 result.  The cfg_a=3 run retains A guidance but is still
4.16 points below A-only, so the current residual is not yet a net positive.
The Stage-B residual was trained against an un-CFG Stage-A prediction, while
inference applies it after cfg_a=3; this mismatch is the first hypothesis to
test (e.g. residual target/scale under CFG, or a smaller cfg_b).

## Stage-B cfg_a=3.0, cfg_b=3.0 (2026-11-26)

Classification, 5 seeds:

| seed | Acc |
|---|---:|
| 42 | 58.33 |
| 7 | 58.33 |
| 19 | 68.75 |
| 0 | 62.50 |
| 1 | 58.33 |

```
mean 61.25, std 4.57
```

## Final Stage-B comparison

### Classification

| Experiment | 5-seed Acc |
|---|---:|
| Stage-A 40k | 82.50 +/- 2.37 |
| Stage-B cfg_a=3, cfg_b=1 | 78.34 +/- 4.80 |
| Stage-B cfg_a=1, cfg_b=1 | 65.42 +/- 6.35 |
| Stage-B cfg_a=3, cfg_b=3 | 61.25 +/- 4.57 |

### Localization (best checkpoint, fabric)

| Experiment | AUROC-I | AP-I | F1-I | AUROC-P | AP-P | F1-P | PRO |
|---|---:|---:|---:|---:|---:|---:|---:|
| cfg_a=3, cfg_b=1 | 100.0 | 100.0 | 100.0 | 86.4 | 51.9 | 49.3 | 48.4 |
| cfg_a=1, cfg_b=1 | 100.0 | 100.0 | 100.0 | 80.1 | 35.7 | 38.3 | 42.0 |
| cfg_a=3, cfg_b=3 | 96.9 | 99.2 | 94.7 | 84.8 | 40.7 | 39.4 | 45.1 |

The localization test was run with `test-localization.py` on the saved best
checkpoint.  An A-only localization baseline has not yet been run, so the table
compares Stage-B configurations only.

### Conclusion

- Among the three Stage-B settings, `cfg_a=3, cfg_b=1` is best for both
  classification and pixel localization.
- Increasing `cfg_b` to 3 hurts classification (61.25) and does not improve
  localization over `cfg_b=1`.
- Stage-B still does not beat the A-only classification result (82.50).
