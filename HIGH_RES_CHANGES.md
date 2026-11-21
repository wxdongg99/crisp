# High-resolution exploration changes

Branch: `highres-exploration`

## 2026-11-21 highres-v1

### Scope

Standalone PixArt-Σ probe for native 512/1024/2048 generation.

### Added

- `highres/pixart_sigma_generate.py`
- `highres/prompts/zju_fabric.txt`
- `highres/requirements-highres.txt`
- `highres/README.md`

### Not changed

- CRISP LDM Stage-A/Stage-B code
- Stable branch `crisp-cvpr`
- Existing model repository

### Next

1. Run 1024 smoke test and inspect quality.
2. Run 2048 smoke test with CPU offload if needed.
3. Compare with SDXL 1024 + high-res fix before any migration decision.
