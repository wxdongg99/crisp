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


## LFP VAE fill/reconstruct + DINOv2 score

Script: `explore/test_lfp_vae_recon_dinov2.py`

Pipeline:

```text
I_ori
-> z_orig = VAE.encode(I_ori)
I_fill = I_ori with LFP mask replaced by solid color (mean of normal pixels)
-> z_fill = VAE.encode(I_fill)
z_recon = z_fill.clone()
z_recon[mask outside] = z_orig[mask outside]   # only LFP-mask latent is replaced
I_recon = VAE.decode(z_recon)
DINOv2 cosine(I_recon, I_ori)
```

Run on B300:

```bash
CUDA_VISIBLE_DEVICES=2 TORCH_HOME=/data/solo/wxd/torch_hub PYTHONPATH=$PWD /data/solo/wxd/envs/qwen21_py310/bin/python explore/test_lfp_vae_recon_dinov2.py   --model /data/solo/wxd/models/Qwen-Image-2.1   --image /data/solo/wxd/data/zju_leaper_native512/fabric/test/pattern16/001.jpg   --mask /data/solo/wxd/data/zju_leaper_native512/fabric/ground_truth/pattern16/001.png   --size 1024 --out_dir explore/lfp_recon_dinov2/pattern16_001
```

DINOv2 score on 3 pattern16 samples:

```text
sample  LFP frac  global_cls_cos  mask_patch_cos  outside_patch_cos  mask_change
001     6.71%     0.6986          0.3154          0.8754             0.6846
002     6.23%     0.6147          0.3210          0.8158             0.6790
003     1.61%     0.7537          0.3877          0.9080             0.6123
```

Interpretation: mask-region DINOv2 patch similarity drops to ~0.32–0.39 while
outside stays ~0.82–0.91, so the fill/reconstruct operation changes the masked
defect region far more than the background. `mask_change = 1 - mask_patch_cos`
is the restoration/change score (higher = more changed inside LFP mask).

On pattern16/001 the latent check is exact:

```text
latent_absdiff_inside_mask  = 32.6478
latent_absdiff_outside_mask = 0.0000
```

So outside the LFP latent mask `z_orig` is byte-identical; only the mask region
is replaced by `z_fill`. Visuals:

```text
recon_compare.png        # I_ori / mask / I_fill / I_recon
latent_recon_compare.png # z_orig / z_fill / z_recon / |z_recon-z_orig|*mask
```
