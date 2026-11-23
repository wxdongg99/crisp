# FP8 and two-device support

## Python 3.12 FP8 environment

```text
/media/cc/U2-Server/crisp-storage/envs/highres_py312
python 3.12.16
torch 2.4.1+cu121
diffusers 0.36.0
transformers 4.46.3
torchao 0.7.0
kornia 0.7.3
```

`torchao.quantization.float8_weight_only` and
`torchao.float8.convert_to_float8_training` both import successfully.

## FP8 inference validated

PixArt 1024 mask, CFG=4.5, SDPA, bf16 base + FP8 transformer:

| run | generation | peak VRAM |
|---|---:|---:|
| bf16 `stain_1024_cfg45_sdpa` | 4.05s | 3,797 MB |
| FP8 `fp8_1024_cfg` | 7.20s | 3,216 MB |

Visual comparison:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_mask/fp8_comparison.jpg
```

FP8 output is visually close to bf16 and lower VRAM, though slower in this
unoptimized path.

## FP8 training

`highres/textual_inversion_sdxl.py --fp8` applies:

```python
from torchao.float8 import convert_to_float8_training
unet = convert_to_float8_training(unet)
```

A 5-step smoke was attempted while the 4800-step masked TI was still using
12+ GB on the same GPU; it OOMed during model conversion. The code path is
present but needs a free GPU to validate.

## Two-device placement

`highres/pixart_mask.py` supports:

```text
--tx-device cuda:1
--vae-device cuda:0
```

This is explicit placement, not model sharding; source latents are moved to the
transformer device, and decoding is moved to the VAE device.

## FP8 multi-seed PixArt mask

PixArt 1024 mask, CFG=4.5, FP8 transformer, seeds 0–2:

```text
fp8_seed0, fp8_seed1, fp8_seed2
```

Montage:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_mask/
  fp8_multiseed_montage.jpg
```

All three seeds produce coherent dark stain-like regions inside the mask while
preserving background. FP8 inference is therefore validated visually on PixArt
1024 mask.
