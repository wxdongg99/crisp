# SDXL TI on anomaly-region crops

## Goal

Test whether token-level textual inversion can learn a defect concept when the
training crops are centered on real ZJU fabric anomalies, instead of normal
fabric texture.

## Data

Anomaly images + ground-truth masks:

```text
fabric/test/pattern16/*.jpg
fabric/ground_truth/pattern16/*.png
```

Crop rule: mask bbox expanded by 0.8×bbox, resized to 1024².

Generated crops:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/anomaly_crops/
```

## Training

```text
8 crops
300 steps
lr 2e-4
init_word defect
prompt "a realistic macro photograph of a fabric defect, <anomaly>, high detail"
```

Checkpoint:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/sdxl_token_anomaly_300/
```

## Result

Raw and EMA samples still look like normal fabric textures; the defect is not
visibly learned. This is a negative result for the current single-token,
300-step, 8-crop setup.

## Next

- use larger defect crops and diverse anomaly patterns
- add mask conditioning during training
- try separate tokens per defect pattern
- compare against a normal-texture token control
