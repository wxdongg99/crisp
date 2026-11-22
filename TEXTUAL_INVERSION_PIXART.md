# PixArt-Sigma T5 token-level TI probe

## Implementation

`highres/textual_inversion_pixart.py`

- T5 tokenizer gets one placeholder token;
- T5 input embedding table is resized;
- the placeholder row is replaced before `T5EncoderModel(inputs_embeds=...)`;
- PixArt transformer / VAE / T5 are frozen;
- PixArt-Sigma outputs 8 channels; the first 4 are used for noise MSE.

Training output:

```text
/media/cc/U2-Server/crisp-storage/textual_inversion_runs/pixart_token_200/
  pixart_token_embedding.pt
  manifest.json
```

200-step loss was finite, but generated samples are not semantically tied to the
fabric target (one building, one portrait). This validates the T5 gradient path
but not the quality of a single-image PixArt TI.

Generation from the saved token:

```text
highres/pixart_token_generate.py
highres_runs/pixart_token/pixart_token_200_1024/
```

## Conclusion

```text
PixArt token TI engineering path: works
PixArt token TI quality: not usable yet
```

Next: use multiple target crops, a semantically meaningful initializer word,
and lower/EMA-smoothed LR before judging PixArt TI.
