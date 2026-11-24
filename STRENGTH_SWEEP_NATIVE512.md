# Native512 token inpainting: strength sweep at CFG=5

Fixed source/mask/seed/token, resolution 512, prompt:

```text
a photo of a fabric defect, <pattern16>, high detail
```

Grid:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  strength_sweep_native512/strength_sweep_grid.jpg
```

## Observations

- CFG=5 removes the colorful stitch artifacts seen at CFG=12.
- strength 0.4–0.6: almost no visible change inside the mask.
- strength 0.7–0.9: a faint horizontal band/ghost appears, but still not a
  plausible fabric defect.
- steps 40 vs 20 at CFG=5 does not create a qualitative improvement.
- The mask used here is a large strip, so the edit is dominated by a low-frequency
  band rather than a localized defect.

Conclusion: CFG=5 is the correct conservative setting, but the token/content
problem remains. Next is prompt-specific inpainting training/eval with fine
masks, not raising CFG or steps.

## Mask visualization correction

The first grid overlapped the top-row reference panels with the output rows, so
the mask looked invisible. The mask itself is valid: 24,600 white pixels,
a horizontal strip.

Corrected view:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  strength_sweep_native512/strength_sweep_maskfix2.jpg
```

Red = mask-overlay on source; white panel = mask. The mask is a long horizontal
band, which explains why outputs become a low-frequency horizontal ghost rather
than a localized defect.
