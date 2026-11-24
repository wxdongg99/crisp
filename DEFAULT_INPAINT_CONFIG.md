# Default SDXL inpainting config

`highres/sdxl_inpaint_mask.py` defaults updated:

```text
steps=40
strength=1.0
guidance_scale=5.0
resolution default still passed by caller
```

A 10-pair random native512 evaluation was run with these defaults:

```text
source: native512 good images
mask: generated_mask/fabric/pattern16 masks
token: native512 pattern16 multi4 anomaly
```

Output:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/
  native512_default_random10/grid.jpg
```

Observation: mask white bands are present; outputs show faint horizontal bands
or edge changes, still not localized realistic defects. Background visually
stays close to source; grayscale MAE is 3–6 due low contrast.
