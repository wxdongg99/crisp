# Mask-conditioned inpainting tuning plan

## Status: baseline only, do not tune yet

10 random pairs generated with current fixed settings:

```text
normal source images: random from ZJU fabric train/good
masks: random from local generated_mask/fabric/pattern16 (500-mask dir)
token: pattern16 multi4 anomaly-initialized SDXL TI
SDXL inpainting 1024, 20 steps, strength 0.9, CFG 12
```

Output:

```text
/media/cc/U2-Server/crisp-storage/highres_runs/sdxl_mask/zju500_random10/
```

Result: background is preserved; mask regions become folds/lines/edge artifacts.
The mask support is often a large strip/blob, not a fine defect.

## Planned tuning order (not executed yet)

### 1. strength / steps first

Current `strength=0.9`, `steps=20` means only ~2 effective denoise steps after
inpaint start; it destroys source detail inside the mask.

Sweep, fixed source/mask/seed/token:

```text
strength: 0.35, 0.50, 0.65, 0.80
steps:    20, 30, 40
```

Keep the same scheduler/CFG while sweeping. Evaluate:

- background MAE outside mask;
- masked-region difference from source;
- whether texture continues naturally across mask boundary;
- defect plausibility.

### 2. prompt templates

The ImageNet templates (`a photo of a *`) are too broad for fabric defects.
Use a small controlled set, e.g.:

```text
a photo of a fabric with a stain
a photo of a fabric with a tear
a photo of a fabric with a crease
a close-up of a fabric defect
```

Train/generate with the same defect-type taxonomy. Keep token placeholder and
class context.

### 3. CFG

Current 12 already produces edge artifacts. After strength/prompt are fixed,
sweep:

```text
5, 7.5, 10, 12
```

Pick the lowest CFG that reliably edits the mask.

### 4. mask selection

The local 500-mask directory is mixed: many masks are large strips/blobs.
Filter by:

- area fraction;
- aspect ratio / thinness;
- distance from border;
- pattern consistency.

Keep raw fine masks and LFP v3 as separate arms.

### 5. token training with inpainting objective

Current TI was trained with text-to-image latent loss. It should be trained
with the inpainting objective:

```text
source latent x0 (normal image)
mask = defect support
x_t = q_sample(x0, t)
condition = token prompt + mask-conditioned model
loss = MSE(eps, eps_pred) only in mask, and/or full latent masked loss
```

Also pair normal source and anomalous target from the same pattern; otherwise
the token learns generic anomaly/object concepts.

### 6. evaluation protocol

- fixed 10 random source/mask pairs;
- 3 seeds each;
- compare strength/steps/prompt/CFG/mask variants;
- report background MAE, masked L1, and visual pass/fail.

Do not run these sweeps until the baseline is reviewed.
