# Resume native512 multi4 TI to 10k steps per pattern

Added `--init-embedding` to `textual_inversion_sdxl.py`: it loads the previous
`vec1/vec2` token vectors and continues training with a fresh optimizer/EMA.

Resume command: previous `_multi4_4800` EMA checkpoint + 5,200 additional steps
=> 10,000 total steps per pattern.

Data/config unchanged:

```text
native512 pattern16-19
6 images, dynamic random_transform, mask loss
size=512, num_vectors=4, anomaly init
prompt=a photo of a fabric defect, <patternN>, high detail
lr=5e-3
```

Running on B300 GPU4-7:

```text
sdxl_native512_pattern16_multi4_10k
sdxl_native512_pattern17_multi4_10k
sdxl_native512_pattern18_multi4_10k
sdxl_native512_pattern19_multi4_10k
```
