# Token-level textual inversion (LDM probe)

## Status

Implemented `tools/textual_inversion_token.py`.

Unlike `textual_inversion_basic.py` (soft-prompt delta), this replaces the
placeholder token embedding **before** the BERT encoder:

```text
token ids -> token_emb(ids) -> replace placeholder row -> BERT -> conditioning
```

Only the placeholder vector is trainable; VAE, UNet and BERT are frozen.

## Required code change

`ldm/modules/x_transformer.py: TransformerWrapper.forward` now accepts an
optional `input_embeddings` argument. Existing callers are unchanged.

## Smoke run

```bash
PY=/home/cc/anaconda3/envs/anomalydiffusion/bin/python
CUDA_VISIBLE_DEVICES=1 $PY tools/textual_inversion_token.py \
  --ckpt /mnt/hdd8t/crisp-research/20261113_pix500ft_local/assets/base_model.ckpt \
  --image /path/to/fabric.jpg \
  --prompt "a photo of anomaly" \
  --placeholder anomaly \
  --steps 200 \
  --output_dir /media/cc/U2-Server/crisp-storage/textual_inversion_runs/token_fabric_200
```

Outputs:

```text
token_embedding.pt
token_sample.png
manifest.json
```

## Observation

The 10-step gradient check and 200-step optimization both run, but the first
200-step single-image result still looks unlike the target. This is a smoke
probe, not a converged inversion. Next tuning should use:
- multiple target crops
- lower/EMA-smoothed learning rate
- fixed prompt template and validation set
- VAE round-trip check before blaming the token

## Environment caveat

The official `embedding_manager2` still cannot be used directly because its
PSP/stylegan2 CUDA extension fails to JIT-compile in this environment. This
token probe intentionally bypasses that manager.
