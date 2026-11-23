# CRISP-style textual inversion understanding

## What new_crisp actually does

### Embedding manager

`ldm/modules/embedding_manager2.py`:

- Stage-A/Stage-0/Stage-B config: `num_vectors_per_token: 4`
  (`configs/latent-diffusion/txt2img-1p4B-stageA-lfp.yaml`).
- Placeholder string is `"*"`; initialized from `"sculpture"`.
- Each training class gets its own parameter `[4, token_dim]` in
  `string_to_param_dict` (key is `sample+anomaly`).
- In `forward`, the single `*` token row is replaced by **4 inserted vectors**;
  following text tokens are shifted. This is multi-vector TI, not one vector.

### Prompt

`ldm/data/personalized.py`:

- Training caption is randomly sampled from `imagenet_templates_small`,
  e.g. `"a photo of a *"`; the template text is not a fixed hand-written prompt.
- `data_enhance` in `ddpm.py` lazily calls `utils.random_transform`; the same
  random transform is applied to image and mask.
- `get_learned_conditioning` passes the condition image and class name into the
  embedding manager; only embedding parameters are trainable.

### Initialization / budget

- Token vectors initialized from `"sculpture"` (Stage-A), 4 vectors per token.
- Stage-A config: 50,000 steps for 4 classes, 4 vectors/token.
- This is why the prior single-token SDXL TI was structurally too weak.

## SDXL adaptation implemented

`highres/textual_inversion_sdxl.py` now supports:

```text
--num-vectors 4
--random-prompts
--data-enhance
--mask-loss
```

- one placeholder token is expanded to 4 learned vectors in both CLIP encoders;
- prompt templates are read from the existing
  `imagenet_templates_small` list in `ldm/data/personalized.py` via AST
  (avoids importing `imgaug`);
- dynamic `utils.random_transform` is reused for image/mask;
- mask loss uses the same latent mask support.

Generation script `highres/sdxl_token_generate.py` expands the 4 vectors the same
way.

## PixArt adaptation

Not yet implemented. PixArt T5 has a different tokenizer/encoder; the same
multi-vector insertion and prompt-template scheme can be ported, but SDXL is the
first target because CRISP's f8/4ch UNet path is closer to SDXL.

## Current run

```text
pattern16
6 source images, dynamic random_transform
4800 steps, lr=5e-3
num_vectors=4
random_prompts=True
raw mask loss
```

Output:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/
  sdxl_token_pattern16_multi4_dynamic_4800/
```

## Attention mask handling for multi-vector expansion

new_crisp BERT path does **not** pass an attention mask. For SDXL, the pipeline
normally does, so a token-row hook alone would leave the mask unexpanded.

We now expand the mask together with the token block:

```python
mask_new = cat([mask[:p], ones(n_vectors), mask[p+1:]])[:77]
```

Training: `_encode_with_token` builds and passes the expanded mask to both CLIP
encoders.

Generation: a forward pre-hook on each text encoder expands
`attention_mask` before the encoder runs; the embedding hook expands the token
rows by the same `n_vectors`.

Smoke tested:

```text
training 3 steps + 5-step generation from the saved 4-vector embedding
```

Current 4800-step run:

```text
sdxl_token_pattern16_multi4_mask_4800
```

## Initialization word

The learned token block must start from a semantically meaningful word, not the
random row created by `resize_token_embeddings`.

Default is now:

```text
--init_word anomaly
```

Both SDXL CLIP encoders initialize their N vectors by repeating that word's
embedding. `--init-word` and `--init_word` are aliases. The manifest and
checkpoint record the init word.

For PixArt T5, the placeholder token row is also initialized from
`--init_word anomaly` before training.

## Multi-vector 4 + anomaly initialization result

Run:

```text
pattern16, 6 images, dynamic random_transform, 4800 steps, lr=5e-3
num_vectors=4, init_word=anomaly, random imagenet templates, mask loss
attention_mask expanded
```

Training finished at step 4800.

Generation:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_token/
  pattern16_multi4_anomaly_1024/
  pattern16_multi4_anomaly_fixedprompt/
```

Result: with the saved prompt, seeds produce unrelated objects (wooden deck,
olive fabric, brown garment); with an explicit fabric-defect prompt, the result
is gray fabric with a button/seam, no defect. Multi-vector + anomaly init alone
still does not learn a transferable fabric defect.

Interpretation: the token block is learning a generic "anomaly/object" concept
from too few images and random prompt templates; this is not yet a defect token.

## Generation must be inpainting, not text-to-image

new_crisp `log_images` calls:

```python
self.sample_log(cond=c, x0=z[:N], mask=mask, ...)
```

DDIM inpainting blends every step:

```python
img = img * mask + (1 - mask) * img_orig
```

with `img_orig = q_sample(x0, ts)` and `mask` = defect support. So the source
image latent is fixed outside the mask; only the mask region is generated.

The earlier SDXL token generations used text-to-image and were therefore the
wrong protocol. Added multi-vector token + expanded attention mask to
`highres/sdxl_inpaint_mask.py` and ran:

```text
source: normal fabric image
mask: pattern16 real defect mask
prompt: a photo of a fabric defect, <pattern16>, high detail
token: multi4 anomaly-initialized SDXL token
```

Result directory:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  token_multi4_inpaint_1024/
```

The inpainting path now matches the intended protocol; current content is still
weak and needs prompt/strength/mask tuning.
