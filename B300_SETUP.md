# B300 / B20Z workspace setup

## Connect

Local wrapper:

```bash
ssh-b300
```

or:

```bash
ssh-b300 'nvidia-smi'
ssh-b300 'bash -l'
```

The wrapper uses `/home/cc/.ssh/b300_wxd`, port 9031, user `solo`.

## Directory layout

```text
/data/solo/wxd/
  envs/highres_py310
  projects/new_crisp_highres
  models/
  runs/
  logs/
  data -> /pfs/pfs-7jnepv/solo/wxd/data

/pfs/pfs-7jnepv/solo/wxd/
  data/ models/ runs/ archive/
```

Python environment:

```text
Python 3.10.22
torch 2.9.1+cu130
diffusers 0.36.0
transformers 5.7.0
accelerate 1.9.0
```

Run:

```bash
cd /data/solo/wxd/projects/new_crisp_highres
/data/solo/wxd/envs/highres_py310/bin/python highres/sdxl_inpaint_mask.py --help
```

## Hardware (reported by nvidia-smi)

```text
8 × NVIDIA B20Z, compute capability 10.0 (Blackwell)
183,359 MiB VRAM per GPU (~1.43 TiB total)
driver 580.126.09 / CUDA 13.0
180 CPU cores
2.8 TiB RAM
/data/solo: 14 TB (4.9 TB free)
/pfs/pfs-7jnepv/solo: 201 TB (21 TB free)
```

## Strengths

- 183 GB per GPU: SDXL/PixArt 1024–2048 can run without CPU offload.
- 8 GPUs: parameter sweeps and multi-seed generation can run in parallel.
- 180 CPU cores / 2.8 TiB RAM: large dataloaders and preprocessing.
- Compute capability 10.0 and CUDA 13 stack: modern Blackwell kernels.

## Caveats

- External PyPI/HuggingFace downloads from the B300 are extremely slow; stage
  models via rsync from the local workstation.
- GPUs are shared; check `nvidia-smi` before dispatching.
- The env is an overlay on Onevision-2's Python 3.10/torch install; do not
  upgrade the base environment in place.
