# CRISP additions in `new_crisp`

**Repo**: `https://github.com/sjtuplayer/anomalydiffusion`  
**Local path**: `/home/cc/Documents/anomalydiffusion/new_crisp`  
**Base commit**: `b85297f Update IC-LPIPS`  
**Last update**: 2026-11-19

This document records every CRISP change added on top of the official
AnomalyDiffusion repository. Official AnomalyDiffusion is intentionally a
**single-stage** TI + PSP model; Stage-0/Stage-A/Stage-B, LFP and BPRC are
CRISP extensions.

---

## 1. New files

| File | Purpose |
|---|---|
| `ldm/models/diffusion/lfp.py` | CRISP LFP/TBF mask projection |
| `ldm/models/diffusion/bprc.py` | BPRC residual composition/loss core |
| `ldm/data/dataset_paths.py` | robust image/mask sorting, pairing, VisA-friendly mask normalization |
| `configs/latent-diffusion/txt2img-1p4B-stage0-bgtoken.yaml` | Stage-0 bg-token config |
| `configs/latent-diffusion/txt2img-1p4B-stageA-lfp.yaml` | Stage-A principal config |
| `configs/latent-diffusion/txt2img-1p4B-stageB-residual.yaml` | Stage-B residual config |
| `CRISP_CHANGES.md` | this document |

---

## 2. Modified files

### `generate_with_mask.py`

- Added `--downsample_method {bilinear,tbf}`.
  - default: `tbf` → CRISP LFP
  - explicit `bilinear` → original official behavior
- Added `--force_lfp_mask`; disables the >20% large-mask bilinear fallback.
- Passes `downsample_method` into `model.log_images(...)`.

### `ldm/models/diffusion/ddpm.py`

- Imports `tbf_resize_mask`, `bprc_residual_loss`, `copy`.
- Added model attributes:
  - `training_stage`
  - `mask_pool_type` (default `tbf`)
  - `stage_b_residual_enabled`
  - `v10_bprc_alpha`
  - `_stage_a_embedding_manager`
- Added LFP mask projection:
  - `_resize_mask_to_latent(...)`
  - default `tbf`, explicit `bilinear` restores official behavior
  - >20% white mask falls back to bilinear unless `force_lfp_mask=True`
- `p_losses(...)` now converts the input mask with LFP instead of a hard-coded
  `transforms.Resize((32,32))`.
- `log_images(...)` accepts `downsample_method` and uses LFP for inpainting.
- `get_learned_conditioning(...)` accepts optional `embedding_manager` so the
  frozen Stage-A and trainable Stage-B paths can be evaluated separately.
- Added Stage-B residual training:
  - `enable_stage_b_residual(stage_a_ckpt_root, init_from, alpha)`
  - `p_losses_stage_b(...)`
  - `shared_step(...)` dispatches to Stage-B residual loss when enabled
- Stage-B loss:
  - `eps_A` from frozen Stage-A path under `torch.no_grad()`
  - `eps_R` from trainable Stage-B path
  - target `noise - stop_gradient(eps_A)`
  - masked mean over `(C,H,W)`

### `ldm/modules/embedding_manager2.py`

- Added per-class background token bank:
  - `bg_token_embeddings`
  - `bg_token_k`
  - `bg_token_frozen`
- Added methods:
  - `prepare_bg_token_embeddings(...)`
  - `freeze_bg_token_embeddings(...)`
  - `get_bg_token(...)`
- Stage-0 condition branch in `forward`: `[bg_token]` only.
- Stage-A/other branch prepends `[bg_token]` to `[class_TI | PSP]`.
- `save/load` now persists/restores bg tokens and `bg_token_k`.
- `embedding_parameters()` includes bg tokens only when trainable.
- `name-anomaly.txt` parsing now uses `parse_name_anomaly_file`.

### `ldm/data/personalized.py`

- Robust dataset loading:
  - image extensions filtered
  - numeric-aware sorting
  - image/mask pairing by stem rather than independent list index
  - clear error when an image has no matching mask
- Mask normalization supports MVTec and VisA via
  `normalize_gt_mask_array` (VisA 0/1/4 labels no longer erased by `/255`).
- Added Stage-0 `set='bg_only'`:
  - anomaly masks inverted (defect=0, background=1)
  - normal images appended with all-one mask
- `set in ('train','bg_only')` handles the train split.

### `main.py`

- Added `--stage {single,0,A,B}`.
1. `single`: official single-stage TI + PSP training.
2. `0`: Stage-0 bg-token only.
3. `A`: Stage-A class TI + PSP with frozen bg tokens.
4. `B`: Stage-B residual with frozen Stage-A principal.
- Added:
  - `--bg_token_enable`, `--bg_token_k`, `--bg_token_freeze`,
    `--bg_token_ckpt`
  - `--stage_a_freeze_class_ti`, `--stage_a_freeze_psp`
  - `--mask_pool_type {bilinear,tbf}`
  - `--override_max_steps`
  - `--stage_a_ckpt_root`
  - `--stage_b_init_from {random,stage_a}`
  - `--bprc_alpha`
- Stage-0 wiring: prepare/load bg tokens, freeze TI/PSP, `data.set=bg_only`.
- Stage-A wiring: freeze bg tokens, train class TI + PSP.
- Stage-B wiring: load frozen Stage-A embedding manager and enable BPRC
  residual loss.
- Dataset selection still uses `--spatial_encoder_embedding` for the official
  mask-encoder dataset path.

### `environment.yaml`

- Added `scikit-image==0.21.0` for LFP Otsu thresholding.

---

## 3. Stage commands

### Stage-0: train bg token

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/txt2img-1p4B-stage0-bgtoken.yaml \
  -t --actual_resume models/ldm/text2img-large/model.ckpt \
  --spatial_encoder_embedding --data_enhance \
  --mvtec_path /path/to/mvtec --init_word anomaly \
  --stage 0 --bg_token_enable 1 --bg_token_k 4
```

Output: `checkpoints/embeddings.pt` containing `bg_token_embeddings`.

### Stage-A: train principal (class TI + PSP)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/txt2img-1p4B-stageA-lfp.yaml \
  -t --actual_resume models/ldm/text2img-large/model.ckpt \
  --spatial_encoder_embedding --data_enhance \
  --mvtec_path /path/to/mvtec --init_word anomaly \
  --stage A --bg_token_enable 1 --bg_token_k 4 --bg_token_freeze 1 \
  --bg_token_ckpt <stage0>/checkpoints/embeddings.pt \
  --override_max_steps 50000
```

Output: `checkpoints/embeddings.pt` + `checkpoints/spatial_encoder.pt`.

### Stage-B: train residual (BPRC)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --base configs/latent-diffusion/txt2img-1p4B-stageB-residual.yaml \
  -t --actual_resume models/ldm/text2img-large/model.ckpt \
  --spatial_encoder_embedding --data_enhance \
  --mvtec_path /path/to/mvtec --init_word anomaly \
  --stage B \
  --stage_a_ckpt_root <stageA_run> \
  --stage_b_init_from random \
  --bprc_alpha 1.0 \
  --override_max_steps 50000
```

Stage-B fits:

```text
target_R = noise - stop_gradient(eps_A)
eps      = eps_A + alpha * eps_R
```

---

## 4. Configuration defaults

| Config | train set | max_steps | checkpoint interval |
|---|---|---|---|
| Stage-0 | `bg_only` | 40,000 (4×10k) | 5,000 |
| Stage-A | `train`, repeats 200 | 50,000 (4×12.5k) | 6,250 |
| Stage-B | `train`, repeats 200 | 50,000 | 6,250 |

Change `max_steps` and checkpoint interval together when the number of defect
classes changes:

```text
Stage-0 max_steps = num_defect_classes × 10,000
checkpoint interval = max_steps / 8
```

---

## 5. LFP behavior

```text
downsample_method=tbf (default):
    VAE encode mask
    → per-channel Otsu
    → union
    → trim border
    → merge bilinear support
    → fill holes
    → binarize

mask white ratio > 20%:
    fallback to bilinear unless force_lfp_mask=True
```

`--downsample_method bilinear` restores the official AnomalyDiffusion
bilinear behavior for both generation and Stage-A training masks.

---

## 6. Verification performed

```text
py_compile OK:
  main.py
  ldm/models/diffusion/ddpm.py
  ldm/models/diffusion/lfp.py
  ldm/models/diffusion/bprc.py
  ldm/modules/embedding_manager2.py
  ldm/data/personalized.py
  ldm/data/dataset_paths.py
  generate_with_mask.py

LFP unit smoke:
  input  (2,1,256,256)
  output (2,1,32,32), binary=True

Dataset pairing smoke:
  images 001.JPG / 2.JPG / 10.JPG
  masks  001.png / 2.png / 10.png
  paired correctly

BPRC core smoke:
  eps_A + alpha*eps_R
  residual target has requires_grad=False
  masked residual loss finite

bg token smoke:
  prepare 2 classes × K=4
  get_bg_token shape (2,4,1280)
  save/load 2/2
  Stage-0 forward output (2,8,1280)
```

Not yet run end-to-end here: full Stage-0/Stage-A/Stage-B training on GPU.
Those commands are provided above and should be smoke-tested with
`--override_max_steps 10` before long runs.

---

## 7. Git

The modifications are currently **uncommitted** in the local clone.
The official remote is:

```text
https://github.com/sjtuplayer/anomalydiffusion
```

To review:

```bash
cd /home/cc/Documents/anomalydiffusion/new_crisp
git diff --stat
git status
```

Do not push without explicit authorization; this clone currently only has
read access configured.
