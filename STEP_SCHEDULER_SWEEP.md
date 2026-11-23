# Euler vs DDIM step sweep

Fixed: source, mask, seed 0, token=muti4 anomaly, CFG=12, strength=0.9.

## EulerDiscrete

| steps | background MAE |
|---:|---:|
| 20 | 0.65 |
| 35 | 0.65 |
| 50 | 0.67 |
| 75 | 0.65 |
| 100 | 0.67 |

## DDIM

| steps | background MAE |
|---:|---:|
| 50 | 0.66 |
| 100 | 0.65 |
| 200 | 0.66 |

Grid:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/step_sweep/
  step_sweep_grid.jpg
```

## Finding

Increasing steps makes the mask-region structure more coherent (repeated
stitch-like marks), and DDIM-200 matches new_crisp's default step count. But
the content remains an artifact, not a plausible fabric defect. Background MAE
is essentially flat.

Conclusion: steps were not the primary blocker. The next variables are
strength, prompt/loss protocol, mask selection, and token training with an
inpainting objective.

Script now supports:

```text
--scheduler {euler, ddim}
```
