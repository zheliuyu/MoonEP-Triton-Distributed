# MoonEP-Triton-Distributed

**语言：** [English](README.md) | **简体中文**

## 目录

- [概述](#概述)
- [支持设备](#支持设备)
- [用法](#用法)
- [构建与测试](#构建与测试)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [运行前环境变量](#运行前环境变量)
  - [测试](#测试)
  - [基准测试](#基准测试)
- [致谢](#致谢)
- [引用](#引用)

---

## 概述

基于 [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) 的 [MoonEP](https://github.com/MoonshotAI/MoonEP) 移植版：**`Buffer` API 与 buffer 布局与上游一致**，算子与对称内存由 Triton + **NVSHMEM** 实现，替代上游 CUDA 扩展（`moonep._C`）。

| | 本仓库 | 上游 MoonEP |
|---|--------|-------------|
| Python 包 | **`moonep_td`**（`from moonep_td import Buffer`） | **`moonep`** |
| 后端 | Triton-distributed + NVSHMEM | CUDA VMM + CuTe |

算法说明、性能数据、buffer 示意图与完整 API 语义见 **[MoonEP README](https://github.com/MoonshotAI/MoonEP/blob/master/README.md)**。本仓库文档侧重如何构建与运行移植版。

## 支持设备

- **NVIDIA GPU**，多卡 **NCCL**（机内 EP 推荐 NVLink）
- 已安装 **[Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)** 与 **NVSHMEM**（见下文）

CUDA/VMM 上游及其他硬件：[MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)。

## 用法

完成[环境变量](#运行前环境变量)配置后，调用方式与 MoonEP 相同，仅改用 **`moonep_td`**：

```python
from moonep_td import Buffer

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8, num_sms=32, token_padding=128)
# dispatch → prefetch_weight → expert 计算 → combine →（训练）reduce_grad
# 张量 shape、zero_copy、destroy 等见上游 MoonEP README
```

## 构建与测试

### 环境要求

- Linux，**Python ≥ 3.11**
- **CUDA ≥ 12.4**，带 CUDA 的 PyTorch（见 `setup.py`；可从 [cu124 index](https://download.pytorch.org/whl/cu124) 安装）
- **≥ 2 张 GPU** + NCCL（默认测试矩阵推荐 4 卡及以上）
- 已构建安装 Triton-distributed：在该仓库执行 `pip install -e 'python[build]'`

### 安装

```bash
git clone https://github.com/ByteDance-Seed/Triton-distributed.git
git clone <this-repo> MoonEP-Triton-Distributed
cd MoonEP-Triton-Distributed

pip install -e .
```

### 运行前环境变量

**本移植版必做**（上游 MoonEP 只需 `pip install -e .` + `torchrun`）。

每次跑测试或 benchmark 的 shell：

```bash
export TRITON_DIST_ROOT=/path/to/Triton-distributed
export MOONEP_TD_ROOT=/path/to/MoonEP-Triton-Distributed

source "$TRITON_DIST_ROOT/scripts/setenv.sh"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"  # 部分机器必设（GLIBCXX / libtriton）
export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-34359738368}"   # 32 GiB；H=7168 压测可调大
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export PYTHONPATH="${MOONEP_TD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "$MOONEP_TD_ROOT"
```

可选：

```bash
export TRITON_CACHE_DIR="${MOONEP_TD_ROOT}/triton_cache"
```

### 测试

须用 **`torchrun`**（直接 `pytest` 会 skip 分布式用例）。

```bash
torchrun --nproc_per_node=4 -m pytest tests/ -q
```

**2 卡**（最低配置）：同样先完成[运行前环境变量](#运行前环境变量)，再执行：

```bash
torchrun --nproc_per_node=2 -m pytest tests/ -q
```

`--nproc_per_node` 须与 EP 进程数一致（测试里 `num_ep_ranks` 取 `dist.get_world_size()`）。**2 卡** 典型 **~67 passed, 1 skipped**（planning 用例 `step1_segment_tail_full_tile` 要求 **R ≥ 4**）。**4 卡** 则为 **67 passed, 1 skipped**（`experts_gt_block_size`，**max_R=2**）。全量含 `large_hidden_stride_spotcheck` 时耗时会明显变长。

分模块与上游 MoonEP 相同（`tests/test_planning.py` 等）。

### 基准测试

脚本在 [benchmarks/](benchmarks/)，参数与 GPU 数量见各文件 docstring。目录与 MoonEP 对齐；**不在此 README 重复 MoonEP 的 benchmark 结论或图表**。

```bash
torchrun --nproc_per_node=8 benchmarks/bench_comm.py
```

## 致谢

- [MoonEP](https://github.com/MoonshotAI/MoonEP)
- [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)

## 引用

研究中使用 MoonEP 请引用[上游 MoonEP](https://github.com/MoonshotAI/MoonEP)（bibtex 见其 README）。
