# CRISP LFP mask on SDXL: first probe

## Implementation

`highres/lfp_mask.py` ports the LDM LFP (Trim + Bilinear + Fill) idea to a
diffusers VAE:

1. binary pixel mask;
2. VAE-encode mask as 3-channel image;
3. per-channel Otsu after VAE projection;
4. union channels, trim latent border;
5. max with bilinear support;
6. 4-neighbour hole filling;
7. upsample latent support back to pixels for the SDXL inpaint pipeline.

`highres/sdxl_inpaint_mask.py` now accepts:

```text
--mask-image
--mask-project {raw,lfp}
```

## Real thin-defect test

Source/mask:

```text
fabric/test/pattern16/001.jpg
fabric/ground_truth/pattern16/001.png
```

Resolution 512, 20 steps, strength 0.9, one seed.

| run | mask | background MAE |
|---|---|---:|
| `lfp_raw_512` | raw thin line | 1.00 |
| `lfp_lfp_512` | LFP projected | 1.61 |

Output:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_lfp/
  raw_vs_lfp_comparison.jpg
```

## Result

The naive SDXL-VAE LFP port produces thick border/edge artifacts in the mask
(visible as horizontal bands at the top/bottom). LFP's latent Otsu/threshold
rules were designed for the LDM VAE and do not transfer directly to the SDXL
VAE without calibration.

Current conclusion:

```text
raw thin mask: conservative, weak edit
naive LFP: spurious border support, not usable yet
```

Next: calibrate LFP per-channel thresholds/trim for SDXL VAE, or use a
pixel-space LFP-like morphology rule instead of direct VAE-space Otsu.

## LFP mask v3

User-requested v3:

1. max-pool the input mask at the latent stride;
2. upsample the pooled support back to pixels;
3. rank pixels by Euclidean distance to the max-pooled support and add the nearest pixels until the area exactly matches v1 LFP;
4. use this as the SDXL mask support.

Implementation: `highres/lfp_mask.py::lfp_pixel_mask_v3`.

### Real thin-defect comparison

| mask | area (512²) | background MAE | output |
|---|---:|---:|---|
| raw input | 4,716 | 1.00 | almost unchanged |
| LFP v1 | 27,072 | 1.61 | strong border artifacts |
| LFP v3 | 26,944 | 1.72 | nearest-point support, but bright jagged band inside mask |

Comparison:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_lfp/lfp_v3b_comparison.jpg
```

Conclusion: v3 removes v1's border-support artifacts and matches v1's area,
but the SDXL inpaint result inside the band is still a bright artefact, not a
plausible defect. This suggests the problem is not only mask support but also
the model/prompt conditioning for thin defects.

## LFP v3 defect-region composite

Added `--composite-mask` to `sdxl_inpaint_mask.py`: generate with LFP v3 but
paste back only the original fine-mask support.

Run:

```text
lfp_v3_composite_512
```

Result:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_lfp/
  lfp_v3_composite_512/comparison.jpg
```

The composite confines the edit to the thin original defect line instead of the
whole v3 band, but the generated content is still a bright line-like artifact,
not a plausible fabric defect.

## v3 correction

The v3 expansion is **not** blur/Gaussian "smooth dilation". It is an exact
nearest-point dilation:

```python
dist = distance_transform_edt(~base)
nearest = np.argpartition(dist.ravel(), target - 1)[:target]
```

i.e. rank all pixels by distance to the max-pooled mask and add the closest
pixels until the support area equals v1 LFP. The earlier "smooth" wording was
misleading.

## LFP v3 speed optimization

The per-step v3 projection was dominated by the VAE encode + Otsu used to obtain
the v1 LFP area. Added:

```python
lfp_pixel_mask_v3_cached(..., cache_key=source_mask_path)
```

The v1/raw area ratio is computed once per **source mask** and cached. Dynamic
augmentation changes the mask footprint, so v3 target area is:

```text
target = augmented_raw_area * cached(v1_area / raw_area)
```

The nearest-point dilation itself runs on CPU with scipy distance transform.

Benchmark on 1024² mask, CPU VAE:

| call | time |
|---|---:|
| first, no cache | 5.24 s |
| cached first | 5.80 s |
| cached subsequent | **0.083 s** |

So after the first call per source mask, v3 projection is ~60× faster. The
first-call cost remains; in training this is paid only once per source image.

Note: for dynamically augmented masks this uses the cached area ratio, so the
support area is an approximation of exact per-step v1. For loss support this is
acceptable; if exact area is required, compute v1 per step.
