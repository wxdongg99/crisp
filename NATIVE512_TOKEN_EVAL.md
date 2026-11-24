# Native512 multi4 token: first inpainting evaluation

Token:

```text
sdxl_native512_pattern16_multi4_4800
size=512
num_vectors=4
init_word=anomaly
prompt=a photo of a fabric defect, <pattern16>, high detail
dynamic random_transform, mask loss, 4800 steps
```

Generation:

```text
source: random native512 good image
mask: random generated_mask/fabric/pattern16 mask
SDXL inpainting 512
steps=20, strength=0.9, CFG=12
```

Grid:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  native512_random10/native512_random10_grid.jpg
```

Observation:

- Background is preserved.
- Mask region shows colorful stitch/edge artifacts and high-frequency bands.
- The native512 token did not fix the content-quality problem; the generated
  region still does not look like a real fabric defect.
- The 500-mask set contains many large strip/blob masks, so some outputs are
  dominated by long seams.

This is a negative evaluation of the current token + inpainting setup, not a
reason to stop the data/resolution work.
