# SDXL token-level textual inversion (probe)

## Clean environment

A dedicated environment was created **without** system-site-packages:

```text
/media/cc/U2-Server/crisp-storage/envs/highres_py38
python 3.8.20
torch 2.4.1+cu121
diffusers 0.36.0
transformers 4.46.3
accelerate 1.0.1
kornia 0.7.3
```

This env is independent of `anomalydiffusion`; no packages were added to the
AnomalyDiffusion environment.

## Implementation

`highres/textual_inversion_sdxl.py`:

- add one placeholder token to both SDXL tokenizers;
- resize both text-encoder embedding tables;
- keep VAE / UNet / both text encoders frozen;
- replace the placeholder embedding row with a trainable vector **inside a
  forward hook** before the CLIP transformer;
- optimize only the two placeholder vectors with SDXL denoising loss;
- save both vectors plus the prompt/placeholder metadata.

Key detail: this transformers version's `CLIPTextModel` does not expose
`inputs_embeds`, so the token row is injected with a forward hook on
`text_model.embeddings`.

## Smoke

```bash
PY=/media/cc/U2-Server/crisp-storage/envs/highres_py38/bin/python

CUDA_VISIBLE_DEVICES=1 $PY highres/textual_inversion_sdxl.py \
  --image /path/to/fabric.jpg \
  --prompt "a photo of <anomaly>" \
  --placeholder "<anomaly>" \
  --steps 200 --lr 5e-4 --dtype bf16 \
  --output_dir /media/cc/U2-Server/crisp-storage/textual_inversion_runs/sdxl_token_200
```

Result:

```text
5-step run: finite loss
200-step run: finite loss, saved vectors
```

Outputs:

```text
sdxl_token_embedding.pt
manifest.json
```

Generation from the learned token is the next step; this run only validates
the token-level training path.

## Generation from the learned token

Added `highres/sdxl_token_generate.py`: loads the saved dual-CLIP token vectors,
adds the placeholder to both tokenizers, resizes the embedding tables, installs
the same forward hooks, and samples with the normal SDXL pipeline.

Run:

```bash
PY=/media/cc/U2-Server/crisp-storage/envs/highres_py38/bin/python
CUDA_VISIBLE_DEVICES=1 $PY highres/sdxl_token_generate.py \
  --embedding /media/cc/U2-Server/crisp-storage/textual_inversion_runs/sdxl_token_200/sdxl_token_embedding.pt \
  --resolution 1024 --num_images 4 --steps 20 --dtype bf16 \
  --output_dir /media/cc/U2-Server/crisp-storage/highres_runs/sdxl_token \
  --run_name sdxl_token_200_1024x4 \
  --enable_attention_slicing --enable_cpu_offload
```

Output:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_token/
  sdxl_token_200_1024x4/images/p00_0000.png ... p00_0003.png
  sdxl_token_200_1024x4/manifest.json
  sdxl_token_200_1024x4/montage.jpg
```

This also serves as the save/reload check: the generated images use only the
serialized token vectors, not the training process state.
