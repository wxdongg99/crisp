# Native-512 ZJU data and training setup

## Normal image ID resolution

Source folder:

```text
/home/cc/Documents/anomalydiffusion/anomalydiffusion/zju_leaper_native512/
  fabric/train/good/all
```

Most files already carry the pool ID. Two were unnamed and were matched by exact
512×512 grayscale content hash against `/media/cc/U2-Server/ZJU-Leaper/Images`:

```text
img.png   -> 079666.jpg
img_1.png -> 079708.jpg
```

All 16 were copied to:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/
  zju_native512_good_ids/
  mapping.csv
```

## SDXL TI training on native512

Data: native512 pattern16 first 6 anomaly images + matching ground-truth masks.

```text
size=512
num_vectors=4
init_word=anomaly
prompt=a photo of a fabric defect, <pattern16>, high detail
data_enhance=dynamic utils.random_transform
mask_loss
4800 steps
lr=5e-3
EMA=0.999
```

Prompt is now exactly the generation template; no random ImageNet template.

Generation should use the same prompt and SDXL inpainting with a normal source
image + mask.

Output:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/
  sdxl_native512_pattern16_multi4_4800/
```
