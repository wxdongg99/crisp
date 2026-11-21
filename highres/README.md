# High-resolution exploration: PixArt-Σ

This folder is a standalone high-resolution probe inside the
`highres-exploration` branch. It does not touch the CRISP LDM Stage-A/B
training code.

## Goal

Validate native high-resolution generation before paying the cost of a model
migration:

| resolution | purpose |
|---|---|
| 512 | low-cost sanity check |
| 1024 | main comparison point |
| 2048 | native high-res stress test |

## Model

```text
PixArt-alpha/PixArt-Sigma-XL-2-1024-MS
```

PixArt-Σ is a compact DiT (≈0.6B transformer) with a T5 text encoder. It is a
good efficiency probe for native high resolution, but its textual-inversion
tooling is less mature than SDXL's.

## Install

```bash
pip install -r highres/requirements-highres.txt
```

## Quick run

```bash
python highres/pixart_sigma_generate.py \
  --resolution 1024 \
  --num_images 8 \
  --batch_size 1 \
  --steps 20 \
  --output_dir highres_runs/pixart_sigma \
  --run_name pixart_1024_smoke
```

2048:

```bash
python highres/pixart_sigma_generate.py \
  --resolution 2048 \
  --num_images 4 \
  --batch_size 1 \
  --steps 20 \
  --output_dir highres_runs/pixart_sigma \
  --run_name pixart_2048_smoke
```

If the 2048 run is close to OOM, add:

```bash
--enable_cpu_offload --enable_attention_slicing
```

## Prompt list

`highres/prompts/zju_fabric.txt` contains fabric prompts for the first ZJU
smoke test.
