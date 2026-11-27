# explore

## LFP v1/v3 visualization on Qwen-Image-2.1

Script: `explore/visualize_lfp_qwen21.py`

Output directory: `explore/lfp_qwen21_vis/`

Key images:

```text
lfp_qwen21_v1_steps.png      # 10-step LFP v1 construction + Otsu histogram
vae_all_channels.png         # all 64 Qwen VAE latent channels
vae_all_channels_otsu.png    # per-channel Otsu diagnostic
lfp_qwen21_pipeline.png      # older compact v1/v3 overview
```

Stats: `stats_v1.txt`, `per_channel_otsu.txt`.

Run on B300:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD \
/data/solo/wxd/envs/qwen21_py310/bin/python \
explore/visualize_lfp_qwen21.py \
  --model /data/solo/wxd/models/Qwen-Image-2.1 \
  --image /data/solo/wxd/data/zju_eval512/fabric/train/good/075428.jpg \
  --mask /data/solo/wxd/data/generated_mask_fabric/pattern16/136.jpg \
  --size 2048 --out_dir explore/lfp_qwen21_vis
```

LFP v1 steps:

```text
1 source
2 raw mask
3 latent response mean_c |z_c|
4 aggregate Otsu: resp > threshold
5 polarity correction using maxpool footprint
6 union with footprint
7 trim border + bilinear merge
8 4-neighbour hole fill -> LFP v1
9 v1 pixel overlay
10 Otsu threshold histogram
```

Important: the standard 4-channel LFP (`highres/lfp_mask.py`) does
per-channel Otsu + polarity, then **unions the channels**. That is correct for a
4-channel VAE (SDXL/PixArt).

Qwen-Image-2.1 has a 64-channel VAE. Applying the literal 64-channel union here
covers 49% of the latent grid (`union_all64_area_latent=8018/16384`), i.e. it
saturates into a huge region. The Qwen helper therefore deviates: it computes
`mean_c |z_c|` and performs one aggregate Otsu, followed by maxpool-footprint
polarity correction, trim, bilinear merge and hole fill. The per-channel Otsu
grid is diagnostic only.

So:

```text
4ch VAE LFP v1: per-channel Otsu -> union
Qwen 64ch LFP v1: mean response -> aggregate Otsu (deviation, because union saturates)
```

Measured on this 2048 mask:

```text
raw area = 366,720 px (8.74%)
v1 area  = 457,472 px (10.91%)
v1 / raw = 1.247
```

So at 2048 the area ratio is 1.25x, not the 4–6x seen at 512. The ratio must
be measured at the target resolution.


## Smaller-side foreground rule (user correction)

The standard LFP polarity is: for each channel, after Otsu, keep the
**smaller-area side** as foreground, then union channels. The earlier per-channel
Otsu grid was only a threshold diagnostic and did not apply that polarity.

Re-ran with the smaller-side rule on Qwen-Image-2.1 2048:

```text
channels = 64
union_smaller_side_area_latent = 8018 / 16384
union_smaller_side_fraction_latent = 0.4894
literal 64ch union = 8018 / 16384  (same result)
aggregate Otsu v1 = 1787 / 16384 = 0.1091 latent
```

Visuals:

```text
vae_all_channels_smaller_side.png
v1_union_smaller_side_raw.png
```

Result: for Qwen's 64-channel VAE, the smaller-side per-channel union still
covers 48.94% of the latent grid, so the standard 4-channel union rule does not
transfer directly. Many channels select the same large Otsu side.

Options:

```text
1. keep aggregate Otsu for Qwen 2.1 (current code)
2. test selecting a subset of channels (e.g. top-4 by footprint overlap)
3. use a 4-channel VAE (SDXL/PixArt) when strict LFP-4ch is required
```


## LFP VAE fill/reconstruct + DINOv2 score (corrected definition)

Script: `explore/test_lfp_vae_recon_dinov2.py`

Corrected pipeline (anomaly latent is kept INSIDE the LFP mask):

```text
I_ori
-> z_orig = VAE.encode(I_ori)

I_fill = I_ori with LFP mask replaced by solid color (mean of normal pixels)
-> z_fill = VAE.encode(I_fill)

z_recon = z_fill.clone()
z_recon[mask inside]  = z_orig[mask inside]   # keep anomaly latent
z_recon[mask outside] = z_fill[mask outside]  # clean fill latent outside

I_recon = VAE.decode(z_recon)
DINOv2(I_recon, I_ori) and DINOv2(I_recon, I_fill)
```

Latent check (pattern16/001): inside the LFP mask `z_recon == z_orig` exactly,
outside it comes from z_fill:

```text
latent_absdiff_inside_mask_vs_orig  = 0.0000
latent_absdiff_outside_mask_vs_orig = 4.8420
```

DINOv2 on 3 pattern16 samples:

```text
sample LFP frac  cls(vs_ori)  mask_patch(vs_ori)  outside_patch(vs_ori)  mask_patch(vs_fill)  outside_patch(vs_fill)
001    6.71%     0.8099       0.5111              0.9022                 0.4670               0.9101
002    6.23%     0.6925       0.3868              0.8485                 0.4699               0.8574
003    1.61%     0.7202       0.4901              0.9098                 0.4811               0.9108
```

Interpretation:

- Inside the LFP latent mask, z_recon is the original anomaly latent.
- DINOv2 mask-patch cosine vs I_ori is only 0.39-0.51 after VAE decode,
  so the anomaly is preserved but its decoded appearance changes.
- DINOv2 mask-patch cosine vs I_fill is ~0.47-0.48, so the mask region remains
  closer to the original anomaly than to the solid fill.
- Outside, similarity to I_ori is 0.85-0.91, so the background is preserved.
- `mask_anomaly_retention = mask_patch_cosine_vs_ori` (higher = more anomaly retained).
- `mask_anomaly_strength_vs_fill = 1 - mask_patch_cosine_vs_fill`.

Visuals:

```text
recon_compare.png         # I_ori / LFP mask / I_fill / I_recon
latent_recon_compare.png  # z_orig / z_fill / z_recon / |z_recon-z_orig|*outside
dinov2_patch_cosine.png   # patch cosine map vs I_ori
```


## LFP v1 vs LFP v3 replacement comparison

Script: `explore/compare_lfp_v1_v3_recon_dinov2.py`

This is the direct check: same z_orig / z_fill, only the replacement mask differs.

```text
z_recon = z_fill.clone()
z_recon[mask inside]  = z_orig[mask inside]
z_recon[mask outside] = z_fill[mask outside]
```

pattern16/001, 1024, mean-fill [75,75,75]:

| mask | area fraction | cls vs I_ori | mask patch vs I_ori | outside patch vs I_ori | mask patch vs I_fill | latent diff inside | latent diff outside |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFP v1 | 6.7139% | 0.9454 | 0.7819 | 0.9601 | 0.5592 | 0.0000 | 2.4527 |
| LFP v3 | 6.7139% | 0.9285 | 0.7509 | 0.9584 | 0.5447 | 0.0000 | 2.5386 |

Key point: v1 and v3 have the same area by construction (v3 target area = v1
area), but different shapes. v1 preserves the decoded anomaly slightly better
(mask patch cosine vs I_ori 0.782 vs 0.751); v3 changes a bit more of the
surrounding support. Both keep latent inside exactly equal to z_orig.

Outputs:

```text
v1_v3_recon_compare.png
mask_v1.png
mask_v3.png
mask_v1_v3_overlay.png
I_recon_v1.png
I_recon_v3.png
stats_v1_v3.txt
```
