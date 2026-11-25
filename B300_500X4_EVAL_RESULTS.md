# B300 500×4 generated dataset evaluation

## Generation

```text
4 patterns × 500 images = 2000 images
SDXL inpainting 512, steps40, strength1.0, CFG5
```

Per-pattern generation time (4 processes in parallel):

| pattern | elapsed |
|---|---:|
| pattern16 | 2373 s (~39.6 min) |
| pattern17 | 2232 s (~37.2 min) |
| pattern18 | 2436 s (~40.6 min) |
| pattern19 | 2243 s (~37.4 min) |

Dataset:

```text
/data/solo/wxd/generated_dataset/zju512/fabric/
  pattern16..19/image/*.jpg
  pattern16..19/mask/*.jpg
```

## Classification

Trained ResNet34 on generated anomalies, tested on real ZJU fabric test split
(4 classes: pattern16–19).

Best test accuracy:

```text
0.5625
```

Chance level = 0.25. The classifier learns something but is far from saturated.

## Localization

Trained DRAEM-style segmenter on generated image-mask pairs, tested on real
fabric test + good split.

Best checkpoint (epoch 56 by sum):

| metric | value |
|---|---:|
| AUC-I | 0.8958 |
| AP-I | 0.9716 |
| AUC-P | 0.8770 |
| AP-P | 0.3187 |
| PRO | 0.4571 |

Per-metric maxima over epochs:

| metric | max |
|---|---:|
| AUC-I | 0.9613 |
| AP-I | 0.9896 |
| AUC-P | 0.8770 |
| AP-P | 0.3187 |
| PRO | 0.4592 |

Interpretation: image-level localization is strong (AUC-I ~0.9–0.96), but
pixel-level AP/PRO is poor (~0.32 AP-P, ~0.46 PRO). This is consistent with
generated images being coarse artifacts rather than pixel-faithful defects:
the segmenter can tell anomaly image vs good image, but cannot localize the
real defect mask accurately.

Logs:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/
  b300_eval_500x4/classification.log
  b300_eval_500x4/localization.log
```
