# MoonEP-Triton-D

MoonEP reimplemented on [Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed) with NVSHMEM symmetric memory.

目录与 MoonEP 的对照见 [`STRUCTURE.md`](STRUCTURE.md)；迁移进度与待办见 [`STATUS.md`](STATUS.md)。

---

## 两个项目，别混

文档里会出现两个名字，指的是**两个不同的仓库/包**：

| | **本仓库（你要用的）** | **上游 MoonEP（可选对照）** |
|---|------------------------|----------------------------|
| 仓库 | 本 repo（`MoonEP-Triton-Distributed`） | [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) |
| Python 包 | **`moonep_td`**（`pip install -e .` 装的是它） | **`moonep`**（另一个 pip 包） |
| 后端 | Triton-distributed + NVSHMEM | CUDA VMM + CuTe |
| 是否必装 | **是** — 日常开发、跑测试、用 `Buffer` 全靠它 | **否** — 仅可选的 API 签名校验 |

**结论**：

- 安装、运行、GPU 回归：**只需要本仓库 + Triton-distributed**，不需要 clone/安装上游 MoonEP。
- 上游 MoonEP **不是** `moonep_td` 的运行依赖；只有跑 `tests/test_api_signatures.py` 时才需要临时安装 `moonep` 做参数签名对比（见 §1 可选步骤）。

---

## 硬件与软件要求

| 项目 | 最低要求 | 已验证环境 |
|------|----------|------------|
| GPU | ≥2 张，NCCL 互通（NVLink 推荐） | 2× Tesla V100S-32GB |
| CUDA | ≥12.4 | 12.4 |
| Python | ≥3.11 | 3.12 |
| PyTorch | ≥2.6，带 CUDA（cu124 index） | 2.6.0+cu124 |
| torchvision | 与 PyTorch 匹配（cu124） | 0.21.0+cu124 |
| Triton-distributed | 源码 `-e python[build]` | 3.4.0（内置 triton 3.4.0） |
| 网络 | 多卡 `torchrun` 需可用 TCP（`MASTER_PORT`） | 单机 |

> 完整 **测试矩阵**（8-rank、H=7168、i64_offset）需要 **8×GPU** 或 **80GB 大显存**；与是否安装上游 MoonEP 无关。见 [STATUS.md](STATUS.md)。

---

## 新机器安装与复现（逐步）

完成 **§1–§7** 全部步骤后，在仓库根目录执行：

```bash
bash scripts/static_check.sh          # 预期: 4 passed
bash scripts/run_gpu_tests.sh         # 预期: 67 passed, 1 skipped, 6 deselected
```

路径以 `$HOME` 为例；可按需替换。下文用 **`$MOONEP_TD_ROOT`** 指本仓库 clone 目录。

### 1. 克隆仓库

**必做** — 本仓库与 Triton-distributed：

```bash
git clone <your-remote>/MoonEP-Triton-Distributed.git "$HOME/MoonEP-Triton-Distributed"
git clone https://github.com/ByteDance-Seed/Triton-distributed.git "$HOME/Triton-distributed"

export MOONEP_TD_ROOT="$HOME/MoonEP-Triton-Distributed"
export TRITON_DIST_ROOT="$HOME/Triton-distributed"
```

**可选** — 上游 MoonEP（`moonep` 包），**仅**用于 API 签名校验测试 `test_api_signatures.py`；不装也能完成全部 GPU 回归：

```bash
# 可选：与上游 moonep.api.Buffer 对比方法签名是否一致
git clone https://github.com/MoonshotAI/MoonEP.git "$HOME/MoonEP"
pip install -e "$HOME/MoonEP"
# 注意：会升级 cuda-python；签名校验后请执行：
#   pip install cuda-python==12.4.0
```

不装上游 MoonEP 时，`static_check.sh` 会打印 `skip: moonep not installed`，**属正常**。

### 2. 检查 GPU 与 CUDA

```bash
nvidia-smi
nvcc --version   # 应 ≥ 12.4；或设置 CUDA_HOME 指向 12.4
```

### 3. Python 环境

建议使用 conda/venv：

```bash
python3 -m venv "$HOME/venv-moonep-td"
source "$HOME/venv-moonep-td/bin/activate"
pip install -U pip wheel setuptools
```

安装 **PyTorch（CUDA 12.4）** — 使用 [PyTorch cu124 index](https://download.pytorch.org/whl/cu124)（当前最高 **2.6.0**）：

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

### 4. NVSHMEM 与 Triton-distributed 依赖

版本需与 Triton-distributed `setup.py` 一致：

```bash
pip install \
  cuda.core==1.0.1 \
  cuda-python==12.4.0 \
  nvidia-nvshmem-cu12==3.6.5 \
  Cython==0.29.24 \
  nvshmem4py-cu12==0.3.0 \
  pytest
```

验证 NVSHMEM Python 包：

```bash
python -c "import nvidia.nvshmem; print('nvshmem ok')"
```

### 5. 编译安装 Triton-distributed（源码，推荐）

> **顺序**：须先完成 §3（PyTorch）和 §4（NVSHMEM 依赖），再执行本节。  
> 上游 [build.md](https://github.com/ByteDance-Seed/Triton-distributed/blob/main/docs/build.md) 中的 NVSHMEM 版本可能偏旧；以本节 §4 及 Triton-distributed 当前 `setup.py` 为准。

**编译工具**（`--no-build-isolation` 要求已预装）：

```bash
pip install "cmake>=3.20,<4.0" "ninja>=1.11.1" "pybind11>=2.13.1" wheel setuptools
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
```

**拉取源码与子模块**（首次约数分钟，需稳定网络；偶发超时可重试）：

```bash
cd "$HOME/Triton-distributed"
git submodule update --init --recursive
```

**编译安装**（本仓库只需 Triton-distributed 的 `[build]` extra，不必装 `[tests,tutorials]`）：

```bash
cd "$HOME/Triton-distributed"

# PyTorch 会安装 pip triton；须卸载，改用 Triton-distributed 内置 triton 3.4.0
pip uninstall -y triton triton_dist 2>/dev/null || true
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')"/triton \
       "$(python -c 'import site; print(site.getsitepackages()[0])')"/triton-*.dist-info 2>/dev/null || true

export USE_TRITON_DISTRIBUTED_AOT=0
echo 'numpy<2' > /tmp/pip_install_constraint.txt
MAX_JOBS=$(nproc) pip install -c /tmp/pip_install_constraint.txt \
  -e "python[build]" --verbose --no-build-isolation --use-pep517

# pip 可能在编译过程中再次装上 triton；完成后务必清掉
pip uninstall -y triton 2>/dev/null || true
rm -rf "$(python -c 'import site; print(site.getsitepackages()[0])')"/triton \
       "$(python -c 'import site; print(site.getsitepackages()[0])')"/triton-*.dist-info 2>/dev/null || true
```

首次编译约 **5–10 分钟**（视 CPU 与磁盘而定）；请预留 **≥10 GB** 可用空间。

验证（须先 `source scripts/setenv.sh`，见 §7）：

```bash
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
source scripts/setenv.sh
python -c "import triton; import triton_dist; print('triton', triton.__version__, 'triton_dist ok')"
# 预期: triton 3.4.0  triton_dist ok
```

**升级 PyTorch 后**：一般只需再次 `pip uninstall -y triton` 并 `source setenv.sh`，**无需**重编 Triton-distributed（除非改动了其源码或 clone 路径）。

**更换 clone 路径或重装**：若 editable 指向的旧路径已删除，在 Triton-distributed 目录重新执行本节 `pip install -e "python[build]" ...`。

### 6. 安装本仓库（Python 包 `moonep-td` / import 名 `moonep_td`）

```bash
pip install -e "$MOONEP_TD_ROOT"
```

这是**唯一必需**的本项目安装步骤；与上游 `moonep` 包无关。

### 7. 每次打开 shell 的环境变量

**必须**在运行 §8–§9 测试前设置（可写入 `~/.bashrc` 或脚本）：

```bash
# 若 §1 未 export，在此设置：
export MOONEP_TD_ROOT="${MOONEP_TD_ROOT:-$HOME/MoonEP-Triton-Distributed}"
export TRITON_DIST_ROOT="${TRITON_DIST_ROOT:-$HOME/Triton-distributed}"

# 修复部分环境下 libstdc++ / NVSHMEM 符号问题（V100 实测需要）
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

# Triton-distributed：NVSHMEM 路径、PYTHONPATH、TRITON_CACHE_DIR 等
cd "$TRITON_DIST_ROOT"
source scripts/setenv.sh
cd "$MOONEP_TD_ROOT"

export PYTHONPATH="$MOONEP_TD_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

> `run_gpu_tests.sh` 仅在 **`/root/Triton-distributed`** 存在时自动 `source setenv.sh`。其他路径下须**先手动** `source "$TRITON_DIST_ROOT/scripts/setenv.sh"`，再跑测试；或临时改写脚本中的路径：

```bash
# 方式 A（推荐）：手动 source 后直接跑
source "$TRITON_DIST_ROOT/scripts/setenv.sh"
cd "$MOONEP_TD_ROOT" && bash scripts/run_gpu_tests.sh

# 方式 B：临时替换脚本中的硬编码路径
sed "s|/root/Triton-distributed|$TRITON_DIST_ROOT|g" scripts/run_gpu_tests.sh | bash
```

多网卡机器若 NVSHMEM 初始化失败，可尝试：

```bash
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0   # 或 ib0 / 实际接口名
```

### 8. 静态检查（无需 GPU 通信）

```bash
cd "$MOONEP_TD_ROOT"
bash scripts/static_check.sh
```

预期：`4 passed`（layout、planning reference、kernel compile），`STATIC CHECK PASSED`。

### 9. GPU 全量回归（与 V100 基线一致）

```bash
cd "$MOONEP_TD_ROOT"
bash scripts/run_gpu_tests.sh
```

**预期输出（2×GPU，Triton kernel 已缓存）：**

```
== GPU tests: NPROC=2 visible_gpus=2 filter='not large_hidden and not i64_offset and not 8rank_smoke' ==
......................................................................................................ss................................
67 passed, 1 skipped, 6 deselected in ~8s
== 8-rank smoke skipped (set RUN_8RANK_TESTS=1 on 8-GPU hosts) ==
```

> **首次运行**（无 Triton cache）需编译全部 kernel，约 **10–15 分钟**；完成后同命令约 **8s**。

单模块调试（须已 `source "$TRITON_DIST_ROOT/scripts/setenv.sh"`，见 §7）：

```bash
torchrun --nproc_per_node=2 -m pytest tests/test_planning.py -v -k "not i64_offset"
torchrun --nproc_per_node=2 -m pytest tests/test_e2e.py -v
```

### 10. 可选：8-GPU / 大显存全矩阵

```bash
# 8 卡 smoke
RUN_8RANK_TESTS=1 NPROC=8 bash scripts/run_gpu_tests.sh

# A100 80GB：跑含 H=7168 / i64 的全部用例
PYTEST_FILTER="" bash scripts/run_gpu_tests.sh
```

---

## 常见问题

| 现象 | 处理 |
|------|------|
| `NVSHMEM_HOME: unbound variable` | 使用最新 `run_gpu_tests.sh`（source setenv 前 `set +u`）；或先 `pip install nvidia-nvshmem-cu12` |
| `import triton_dist` 失败 | 重新按 §5 编译 Triton-distributed；确认 `source setenv.sh` |
| `constexpr_function` / triton 版本混用 | `pip uninstall -y triton` 并删除 `site-packages/triton*`；勿保留 pip triton，使用内置 3.4.0 |
| `pip check` 报 torch requires triton | 预期现象；运行前 `source setenv.sh` 即可，勿 `pip install triton` |
| Triton-distributed 编译 OOM / 磁盘满 | 预留 ≥10 GB；`pip cache purge`；用 `-e python[build]` 即可，勿装 `[tests,tutorials]` |
| submodule 克隆超时 | `git submodule update --init --recursive` 重试；检查网络与 GitHub 连通性 |
| editable 路径失效（移动/删除 clone） | 在新路径重新执行 README §5 的 `pip install -e "python[build]"` |
| `libstdc++.so.6` / NVSHMEM 加载错误 | 设置 `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6` |
| `Address already in use`（torchrun） | 换 `MASTER_PORT=295xx` 或等上次进程退出 |
| 仅 1 张 GPU | 无法跑默认 CI（需 `NPROC=2`）；单卡仅部分静态测试可用 |
| `test_api_signatures` skip | **正常**（未装上游 `moonep`）；可选安装见 §1；装完后 **`pip install cuda-python==12.4.0`** 恢复 nvshmem 栈 |

---

## 调试脚本

迁移/验证用 ad-hoc 脚本在 [`devtools/`](devtools/README.md)（非 MoonEP 上游目录）：

```bash
python devtools/compile_kernels.py
```

正式 CI 入口在 [`scripts/`](scripts/)：

- `run_local_checklist.sh` — **2×V100 本机一键清单**（static + GPU + bench + 刷新 STATUS）
- `run_gpu_tests.sh`、`static_check.sh`、`update_status.sh`

---

## Benchmarks（可选）

```bash
source "$TRITON_DIST_ROOT/scripts/setenv.sh"
export PYTHONPATH="$MOONEP_TD_ROOT:$PYTHONPATH"

torchrun --nproc_per_node=2 benchmarks/bench_comm.py
torchrun --nproc_per_node=2 benchmarks/bench_grad_reduce.py
torchrun --nproc_per_node=2 benchmarks/bench_prefetch.py
```

详见 [`benchmarks/README.md`](benchmarks/README.md)。

---

## 应用示例

Initialize distributed before creating a `Buffer`:

```python
import torch.distributed as dist
from moonep_td.buffer import ensure_nvshmem_initialized
from moonep_td import Buffer

dist.init_process_group("nccl")
ensure_nvshmem_initialized()

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8)
# ... dispatch / prefetch / combine / reduce_grad — API 语义对齐上游 MoonEP 设计
buffer.destroy()
```

`moonep_td.Buffer` 的公开 API 设计对齐 [上游 MoonEP README](https://github.com/MoonshotAI/MoonEP/blob/master/README.md)；实现后端不同（NVSHMEM vs VMM），**无需** import 上游 `moonep`。
