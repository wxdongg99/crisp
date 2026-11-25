# PixArt T5 textual inversion adaptation

`highres/textual_inversion_pixart.py` now mirrors the CRISP-style token setup:

- 1 placeholder token expanded to 4 learned T5 vectors;
- attention mask expanded with the token block;
- placeholder initialized from `anomaly`;
- dynamic `utils.random_transform` image/mask augmentation;
- masked latent loss;
- prompt fixed and reused at generation time.

Important stability finding: bf16 T5/DiT training produced NaN loss after a few
steps even at lr=1e-4. Training in fp32 is stable:

```text
step 1 loss=0.000268
step 20 loss=0.385882
```

FP32 training of pattern16 was launched on B300 GPU4:

```text
4800 steps, size=512, lr=1e-4, mask loss, dynamic aug
output: /data/solo/wxd/runs/ti/pixart_native512_pattern16_multi4_4800
```

Generation script `pixart_token_generate.py` now supports the same multi-vector
insertion and expanded attention mask; a 512 result will be generated when the
token finishes.
