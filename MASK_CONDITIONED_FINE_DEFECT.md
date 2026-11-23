# Mask-conditioned fine-defect generation

## Setup

Real thin defect:

```text
fabric/test/pattern16/001.jpg
fabric/ground_truth/pattern16/001.png
```

PixArt-Sigma 1024, CFG=4.5, SDPA, FP8, T5 CPU. Two masks:

1. raw fine mask;
2. LFP v3 nearest-point support (area matched to v1) with fine-mask composite.

Command additions:

```text
--mask-image
--composite-mask
```

## Results

| mask | background MAE | composite |
|---|---:|---|
| raw fine mask | 1.06 / 1.09 | faint dark line |
| LFP v3 support | 1.05 / 1.09 | faint dark line, slightly stronger |

Comparison:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_mask/
  fine_mask_composite_compare.jpg
```

## Observation

- Fine-mask composite avoids repainting the whole LFP v3 band.
- PixArt produces a subtle dark line inside the real defect mask, much less
  destructive than the earlier SDXL bright band.
- The edit is still faint and not a convincing fabric defect; the content
  quality gate is not passed.

This supports the decision to stop token-only TI and continue only with
mask-conditioned generation.
