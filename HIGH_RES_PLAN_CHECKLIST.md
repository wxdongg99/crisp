# High-resolution exploration plan checklist

Source: Codex read-only planning session, 2026-11-21/22.  
Status legend: `[ ]` todo, `[~]` in progress, `[x]` done.

## P0. Evidence and environment

- [x] Use isolated env `/media/cc/U2-Server/crisp-storage/envs/highres_venv`
- [x] Record model split: ModelScope T5/VAE + HF mirror transformer
- [ ] Record exact git SHA, dependency versions, component hashes in a run manifest
- [ ] Build a non-system-site-packages env for final reproducibility
- [ ] Confirm GPU availability and document peak VRAM per resolution

## P1. True resolution validation

- [x] Set `use_resolution_binning=False` in `highres/pixart_sigma_generate.py`
- [x] Record requested size, denoise size, latent shape, output size in manifest
- [x] Rerun 2048 with current 1024 weights: `latent=256x256`, output 2048x2048
- [x] Verify 512/1024 latent sizes with binning disabled
- [x] Download official 2K transformer `PixArt-alpha/PixArt-Sigma-XL-2-2K-MS`
- [x] Run true native 2048 with official 2K weights: `latent=256x256`, `sample_size=256`
- [~] Compare 1024-weights@2048 vs official 2K@2048 visually (first comparison done; metrics next)

## P2. Token-level textual inversion

- [ ] Keep `tools/textual_inversion_basic.py` as soft-prompt control
- [ ] Add `tools/textual_inversion_token.py` for LDM placeholder-token TI
- [ ] Add `highres/textual_inversion_sdxl.py` (dual CLIP, separate token banks)
- [ ] Add `highres/textual_inversion_pixart.py` (T5 input embedding, attention mask)
- [ ] Add `highres/token_embedding.py` only if shared logic is real
- [ ] Validate placeholder not split/truncated; only token vector changes
- [ ] Validate save/reload consistency, gradient finite/nonzero
- [ ] 10-step engineering check -> 200-step -> 1000-step budget
- [ ] Metrics: held-out denoise loss, fixed seeds, VAE round-trip detail check

## P3. SDXL baseline

- [x] Download SDXL base from ModelScope; refiner not yet
- [x] PixArt 512/1024 fixed-seed baseline (validation runs)
- [ ] SDXL 1024 token TI (200 -> 1000 steps)
- [ ] SDXL 1024 -> 2048 high-res fix baseline
- [ ] Mask probe: background preservation + mask shift + empty/full mask

## P4. PixArt baseline

- [x] PixArt 512/1024 fixed-seed baseline (validation runs)
- [ ] PixArt token TI after T5 gradient/VRAM validation
- [ ] Native 2048 with official 2K weights
- [ ] Mask probe: same protocol as SDXL

## P5. Experiment matrix

- [ ] 2 models x 3 resolutions x 2 TI states x 2 mask states = 24 cells
- [ ] 1 template x 3 seeds per cell = 72 images
- [ ] Add PixArt 1024-weights@2048 control
- [ ] Add SDXL 1024->2048 high-res-fix control
- [ ] Full denominator: 32 cells / 96 main images
- [ ] Boundary/mask-shift/generalization prompts separately counted

## P6. Isolation and provenance

- [x] Keep work on `highres-exploration`; stable `crisp-cvpr@9686ec1` untouched
- [ ] Record checkpoint code SHA + dataset manifest hash per run
- [ ] External run dirs by model/protocol/version/run_id; never overwrite
- [ ] Do not merge whole branch into `crisp-cvpr`; cherry-pick only verified commits
- [ ] Preserve historical `HIGH_RES_CHANGES.md`; append corrections

## P7. Deliverables and tags

- [ ] `highres-resolution-v1`: resolution evidence corrected
- [ ] `highres-ti-contract-v1`: token TI engineering checks pass
- [ ] `highres-sdxl-ti-v1`: SDXL 1024 TI baseline + fixed samples
- [ ] `highres-mask-v1`: mask boundary/shift evidence
- [ ] `highres-comparison-v1`: all matrix cells and failures resolved

## P8. Risks to validate first

- [ ] True 2048 training/inference resource requirement
- [ ] Fine defect detail survives VAE encode/decode
- [ ] TI learns defect, not only fabric background
- [ ] SDXL/PixArt training implementations are actually comparable
- [ ] Mask control really localizes defect, not just preserves background
- [ ] Multi-source PixArt components reproducible and fully probed
