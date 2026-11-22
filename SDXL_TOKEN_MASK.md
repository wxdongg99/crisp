# SDXL learned-token + mask probe

## Setup

`highres/sdxl_inpaint_mask.py --embedding ...` combines:

- SDXL token-level TI vectors (`vec1`, `vec2`) via forward hooks;
- `StableDiffusionXLInpaintPipeline` 4-channel background preservation.

## Fixed-seed resolution comparison

Source:

```text
highres_runs/pixart_sigma/goodprompt_1024w_1024/images/p00_0000.png
```

Mask: center box `0.35 0.35 0.65 0.65`, seed 0, 20 steps, strength 0.9.

| run | resolution | background MAE |
|---|---:|---:|
| `tokenmask_512` | 512 | 4.67 |
| `tokenmask_1024` | 1024 | 1.90 |
| `tokenmask_2048` | 2048 | 1.50 |

Output:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_token_mask/
  tokenmask_512/
  tokenmask_1024/
  tokenmask_2048/
  tokenmask_comparison.jpg
```

Observation: background outside mask is preserved at all three resolutions.
The learned token was trained on a single fabric image, so the edit inside the
mask is a soft texture change rather than a convincing anomaly.

## Notes

- 512 has higher background MAE because the source is resized aggressively and
  the mask covers a larger relative area of the texture.
- 2048 uses the same SDXL base UNet, so it is direct high-resolution inference;
  it is slower but did not OOM with CPU offload.
