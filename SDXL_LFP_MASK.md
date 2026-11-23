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
3. dilate with a distance-transform disk until the area matches v1 LFP;
4. use this as the SDXL mask support.

Implementation: `highres/lfp_mask.py::lfp_pixel_mask_v3`.

### Real thin-defect comparison

| mask | area (512²) | background MAE | output |
|---|---:|---:|---|
| raw input | 4,716 | 1.00 | almost unchanged |
| LFP v1 | 27,072 | 1.61 | strong border artifacts |
| LFP v3 | 26,944 | 1.72 | smooth support, but bright jagged band inside mask |

Comparison:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_lfp/lfp_v3b_comparison.jpg
```

Conclusion: v3 removes v1's border-support artifacts and matches v1's area,
but the SDXL inpaint result inside the band is still a bright artefact, not a
plausible defect. This suggests the problem is not only mask support but also
the model/prompt conditioning for thin defects.
