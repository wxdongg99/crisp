# explore

## LFP v3 visualization on Qwen-Image-2.1

Script:

```text
explore/visualize_lfp_qwen21.py
```

Output:

```text
explore/lfp_qwen21_vis/lfp_qwen21_pipeline.png
explore/lfp_qwen21_vis/stats.txt
```

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

The script visualizes:

```text
source
raw mask
Qwen VAE latent response
LFP v1 support
v3 max-pool base
distance to max-pool base
LFP v3 final
raw/v3 overlay
```
