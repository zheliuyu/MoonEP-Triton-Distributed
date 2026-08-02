# MoonEP-Triton-Distributed

**Languages:** **English** | [简体中文](README.zh-CN.md)

## Contents

- [Overview](#overview)
- [Supported devices](#supported-devices)
- [Usage](#usage)
- [Build & Test](#build--test)
  - [Prerequisites](#prerequisites)
  - [Install](#install)
  - [Environment (before `torchrun`)](#environment-before-torchrun)
  - [Tests](#tests)
  - [Benchmarks](#benchmarks)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)

---

## Overview

[Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) port of [MoonEP](https://github.com/MoonshotAI/MoonEP): same **`Buffer` API and buffer layout**, with Triton kernels and **NVSHMEM** symmetric memory instead of the upstream CUDA extension (`moonep._C`).

| | This repo | Upstream MoonEP |
|---|-----------|-----------------|
| Package | **`moonep_td`** (`from moonep_td import Buffer`) | **`moonep`** |
| Backend | Triton-distributed + NVSHMEM | CUDA VMM + CuTe |

Algorithm, performance numbers, buffer diagrams, and full API semantics are documented in the **[MoonEP README](https://github.com/MoonshotAI/MoonEP/blob/master/README.md)**. This repository focuses on building and running the port.

## Supported devices

- **NVIDIA GPU**, multi-GPU **NCCL** (NVLink recommended for intranode EP)
- **[Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)** and **NVSHMEM** installed (see below)

Other hardware and the CUDA/VMM build: [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP).

## Usage

Install and set [environment variables](#environment-before-torchrun), then use the same call patterns as MoonEP with **`moonep_td`**:

```python
from moonep_td import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8, num_sms=32, token_padding=128)
# dispatch → prefetch_weight → expert compute → combine → (training) reduce_grad
# See upstream MoonEP README for tensor shapes, zero_copy, and destroy().
```

## Build & Test

### Prerequisites

- Linux, **Python ≥ 3.11**
- **CUDA ≥ 12.4**, PyTorch with CUDA (see `setup.py`; e.g. `torch>=2.6,<2.7` from the [cu124 index](https://download.pytorch.org/whl/cu124))
- **≥ 2 GPUs** with NCCL (4+ GPUs recommended for the default test matrix)
- Triton-distributed built and installed: `pip install -e 'python[build]'` in that repo

### Install

```bash
git clone https://github.com/ByteDance-Seed/Triton-distributed.git
git clone <this-repo> MoonEP-Triton-Distributed
cd MoonEP-Triton-Distributed

pip install -e .
```

### Environment (before `torchrun`)

**Required for this port** (upstream MoonEP only needs `pip install -e .` and `torchrun`).

In **every** shell that runs tests or benchmarks:

```bash
export TRITON_DIST_ROOT=/path/to/Triton-distributed
export MOONEP_TD_ROOT=/path/to/MoonEP-Triton-Distributed

source "$TRITON_DIST_ROOT/scripts/setenv.sh"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"  # required on some hosts (GLIBCXX vs libtriton)
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-34359738368}"   # 32 GiB; raise for H=7168 stress cases
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export PYTHONPATH="${MOONEP_TD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$MOONEP_TD_ROOT"
```

Optional:

```bash
export TRITON_CACHE_DIR="${MOONEP_TD_ROOT}/triton_cache"
```

### Tests

Use **`torchrun`** (plain `pytest` skips distributed tests).

```bash
torchrun --nproc_per_node=4 -m pytest tests/ -q
```

**2 GPUs** (minimum hardware): use the same [environment](#environment-before-torchrun) block, then:

```bash
torchrun --nproc_per_node=2 -m pytest tests/ -q
```

`--nproc_per_node` must match the EP world size (tests use `dist.get_world_size()` as `num_ep_ranks`). On **2 GPUs**, expect **~67 passed, 1 skipped** (planning case `step1_segment_tail_full_tile` requires **R ≥ 4**). On **4 GPUs**, expect **67 passed, 1 skipped** instead (`experts_gt_block_size`, **max_R=2**). The `large_hidden_stride_spotcheck` case is very slow in a full run.

Per-module runs match upstream MoonEP (`tests/test_planning.py`, `test_dispatch.py`, `test_combine.py`, `test_e2e.py`, `test_grad_reduce.py`, `test_prefetch.py`).

### Benchmarks

Scripts live under [benchmarks/](benchmarks/); see each file’s docstring for GPU count and options. They are inherited from the MoonEP tree for parity — **numbers are not reproduced in this README**.

```bash
torchrun --nproc_per_node=8 benchmarks/bench_comm.py
```

## Acknowledgments

- [MoonEP](https://github.com/MoonshotAI/MoonEP)
- [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)

## Citation

If you use MoonEP in research, cite [upstream MoonEP](https://github.com/MoonshotAI/MoonEP) (bibtex in their README).
