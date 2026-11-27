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
