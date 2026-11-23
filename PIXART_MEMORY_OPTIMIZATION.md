# PixArt mask memory optimization

## First finding: autograd graph, not only precision

The original manual PixArt mask loop did **not** freeze the transformer/VAE.
Although no optimizer was used, parameters had `requires_grad=True`, so the
20-step manual loop could retain a long autograd graph. On 48 GB this looked like
"attention OOM" but was partly graph retention.

Fix:

```python
pipe.transformer.requires_grad_(False)
pipe.vae.requires_grad_(False)
```

Optional additional guard for inference:

```python
with torch.inference_mode():
    ...
```

After this fix, PixArt mask **1024 no-CFG** fits on a local 24 GB card and
produces a coherent defect-like edit; before the fix it OOMed at 48 GB.

Comparison:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/pixart_mask/
  pixart_mask_frozen_comparison.jpg
```

## Memory components (PixArt-Sigma 1024)

| component | fp16/bf16 size |
|---|---:|
| T5 text encoder | 9.5 GB |
| transformer | 1.2 GB |
| VAE | 0.17 GB |

At 1024, latent is 128×128; PixArt patch size 2 gives 4096 tokens. Attention
activations dominate, not weights. CFG doubles the batch/activation footprint.

## Practical options, in order

1. **Freeze all inference models / `torch.inference_mode()`**  
   Biggest and cheapest win; avoids graph retention.

2. **bf16 or fp16**  
   Halves weight and activation bytes. bf16 is preferred for stability; 4090
   supports it. FP8 transformer weights are possible on newer stacks but
   activation/attention remains the bottleneck and support is fragile.

3. **Memory-efficient attention**  
   Use PyTorch SDPA / xformers / FlashAttention if available. This changes
   attention from materializing N×N to a fused path and is the main lever for
   4096-token PixArt.

4. **Attention slicing**  
   `pipe.enable_attention_slicing()` and/or setting a sliced attention
   processor on the transformer. It trades speed for lower peak memory; verify
   it is actually applied to the manual transformer call.

5. **Keep T5 on CPU**  
   T5 is 9.5 GB and used once. Encode prompts on CPU, delete T5, then move only
   transformer/VAE to GPU. This is what made 1024 no-CFG fit on 24 GB.

6. **Avoid CFG at high resolution**  
   CFG doubles the transformer batch. no-CFG 1024 works with frozen params;
   CFG=4.5 is the next thing to test on 48 GB. If still tight, use guidance
   rescaling / distilled guidance or stream negative and positive passes.

7. **Two-GPU placement, not naive model sharding**  
   Put T5 on GPU0 and transformer+VAE on GPU1, or split the CFG batch across two
   GPUs. Standard diffusers model sharding is not available for the manual loop;
   device placement must be explicit.

8. **Quantized transformer / text encoder**  
   int8/fp8 T5 and transformer reduce weight memory but do not solve the
   activation peak. Useful after attention and placement are optimized.

9. **Resolution/tiling fallback**  
   Run 512 for smoke tests and 1024 for the main mask comparison; do not claim
   native 2048 until the 1024 path is stable.

## Recommended next test order

```text
1. 1024 no-CFG + frozen params (done, 24GB)
2. 1024 CFG=4.5 + frozen params on 48GB
3. SDPA/xformers attention
4. T5 CPU + transformer/VAE GPU
5. If still OOM: two-GPU T5/UNet split
```
