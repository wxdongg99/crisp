# Basic textual-inversion harness

## What is implemented

`tools/textual_inversion_basic.py` is a minimal, standalone inversion probe for
the official AnomalyDiffusion LDM.

It currently optimizes a **soft prompt embedding** on top of the frozen BERT
conditioning:

```text
c = BERT(prompt) + delta
```

and keeps VAE / UNet / BERT frozen. This is intentionally smaller than the
official Stage-A TI manager, because the goal is to validate the inversion
optimization loop before touching the full CRISP training stack.

## Why not the full official TI manager yet

The official `embedding_manager2` path imports the PSP/stylegan2 custom CUDA
ops. On this machine those ops try to JIT-compile and fail with the current
CUDA/GCC combination. The basic probe avoids that dependency.

Two small guards were added to make a no-manager path possible:

- `LatentDiffusion.instantiate_embedding_manager(None)` returns `None`.
- `TransformerWrapper.forward` defines `position = None` when no embedding
  manager is passed.

## Usage

```bash
PY=/home/cc/anaconda3/envs/anomalydiffusion/bin/python

CUDA_VISIBLE_DEVICES=0 $PY tools/textual_inversion_basic.py \
  --ckpt /mnt/hdd8t/crisp-research/20261113_pix500ft_local/assets/base_model.ckpt \
  --image /path/to/target.jpg \
  --steps 200 \
  --lr 1e-3 \
  --output_dir /media/cc/U2-Server/crisp-storage/textual_inversion_runs/fabric_basic_200
```

Outputs:

```text
inverted_prompt_embedding.pt   # delta + prompt
inverted_sample.png            # DDIM sample after inversion
```

## Smoke result

A 200-step fabric run completed end-to-end:

```text
loss: 0.035 -> variable in [0.002, 0.34]
```

It does not yet reconstruct the target well; this is expected for a first
soft-prompt probe with one image and no regularization. The next step is a
token-level inversion (placeholder token embedding) with:
- lower LR / EMA of delta
- multiple target crops
- fixed prompt template
- held-out reconstruction check

## Run output

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/
├── basic_smoke/
└── fabric_basic_200/
```
