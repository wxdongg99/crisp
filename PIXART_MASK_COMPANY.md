# PixArt mask on company 4090 (48 GB)

## Environment

Company host has 8× RTX 4090 48 GB. GPU7 was free (48 GB) for the mask run.

Clean, non-invasive package overlay:

```text
/home/train/CRISP/work/highres_pkgs
```

Used with the existing AnomalyDiffusion Python 3.8 env via `PYTHONPATH`; no
packages were installed into `/home/train/CRISP/envs/anomalydiffusion`.

Model:

```text
/home/train/CRISP/work/pixart_sigma_1024
```

Downloaded from HF mirror; 18 files, T5 + VAE + transformer + tokenizer.

## Results

| run | GPU | CFG | resolution | background MAE | status |
|---|---|---:|---:|---:|---|
| `stain_512_company` | 4090 48G GPU7 | 1.0 | 512 | 5.04 | succeeded |
| `stain_512_cfg45` | 4090 48G GPU7 | 4.5 | 512 | — | OOM |
| `stain_1024_nocfg` | 4090 48G GPU7 | 1.0 | 1024 | — | started |

Findings:

- PixArt mask with T5 on CPU + mask blending runs on 48 GB at 512 with CFG=1.
- CFG=4.5 doubles activation memory and OOMs at 512 on the 48 GB card.
- The no-CFG 512 result preserves background but the in-mask content is not a
  plausible stain; it is an engineering smoke, not a quality result.
- T5-on-GPU also OOMs at 48 GB during 1024 because T5+transformer+attention
  exceed the card.

## Next

- test 1024 no-CFG (running)
- try `attention_slicing`/smaller transformer chunking, or a 2-GPU split
- do not claim PixArt mask quality until CFG or an equivalent guidance path runs
