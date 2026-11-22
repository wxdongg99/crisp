# SDXL mask-conditioned probe

## Script

`highres/sdxl_inpaint_mask.py`

Uses `StableDiffusionXLInpaintPipeline` with the SDXL base 4-channel UNet.
The pipeline's 4-channel path preserves background outside the mask; inside
the mask it regenerates with the prompt.

Optional `--embedding` loads the two-SDXL-token TI checkpoint and installs the
same embedding hooks as `sdxl_token_generate.py`.

## Runs

Source image:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_sigma/
  goodprompt_1024w_1024/images/p00_0000.png
```

Prompt:

```text
a realistic macro photograph of a dark red wine stain on natural beige linen fabric, high detail, photorealistic
```

| run | mask | background MAE | observation |
|---|---|---:|---|
| `stain_mask_2` | center 30% | 1.88–1.97 | stain generated inside mask, background preserved |
| `mask_empty` | all black | 1.98 | no intended edit |
| `mask_full` | all white | n/a | full re-generation, no background constraint |
| `mask_shift2` | top-left box | 2.06 | stain follows shifted mask |

Comparison images:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  stain_mask_2/comparison.jpg
  mask_variants.jpg
  mask_shift2/comparison.jpg
```

## Notes

- This is sampler/inpaint-level mask control, not learnable spatial condition.
- Background MAE is computed on decoded RGB outside the mask, without pasting
  the source back.
- Empty mask still has small numeric drift (~2/255), expected from VAE.
- Full mask intentionally removes background preservation.
