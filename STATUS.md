# MoonEP-Triton-D 迁移状态

**更新**: 2026-07-30  
**本仓库**: `moonep_td`（Triton-distributed + NVSHMEM 实现）  
**对照源项目（可选）**: [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP) 的 `moonep` 包 — 仅 API 签名/语义参考，**非运行依赖**  
**当前里程碑**: 2×V100 默认 CI 通过（`bash scripts/run_gpu_tests.sh`）

> **别混**：下文「上游 MoonEP / `moonep`」指 Moonshot 原版仓库；「本仓库 / `moonep_td`」指当前 TD 实现。GPU 测试不依赖上游安装。

---

## 当前验证基线（已通过）

在 **2× Tesla V100S-32GB**、CUDA **12.4**、PyTorch **2.6.0+cu124**、torchvision **0.21.0+cu124**、Triton-distributed **3.4.0**（内置 triton 3.4.0）源码安装环境下：

```bash
bash scripts/static_check.sh   # 4 passed
bash scripts/run_gpu_tests.sh  # 67 passed, 1 skipped, 6 deselected
# 首次无 cache 约 10–15 min；kernel 已缓存后 ~8s
```

| 类别 | 结果 |
|------|------|
| GPU 回归（filter 后） | **67 passed** |
| 运行时 skip | **1** — `planning[step1_segment_tail_full_tile]`（需 R≥4，2 卡跳过） |
| 收集时 deselect | **6** — 见下文「未在本机执行的用例」 |
| 静态检查 | `bash scripts/static_check.sh` — **4 passed**（必需）+ **5 API 签名**（可选：仅当安装了上游 `moonep` 时） |
| 目录结构 | 已与 MoonEP 基本对齐（见 `STRUCTURE.md`）；调试脚本在 `devtools/` |

依赖栈（与 README §4–§5 一致）：`cuda.core==1.0.1`、`cuda-python==12.4.0`、`nvidia-nvshmem-cu12==3.6.5`、`nvshmem4py-cu12==0.3.0`；Triton-distributed 以 `-e python[build]` 源码安装；**勿**保留 pip 安装的 `triton`（与内置 3.4.0 冲突）。

---

## 综合完成度

| 维度 | 完成度 | 说明 |
|------|--------|------|
| API / 编排 | ~98% | 可选 `test_api_signatures` 5/5（对比上游 `moonep` 与 `moonep_td` 方法签名） |
| 语义正确性（2×V100 默认 filter） | ~95% | 全模块 GPU 回归通过 |
| 语义正确性（全量 MoonEP 测试矩阵） | ~75% | 8-rank / H=7168 / i64_offset 未在本机跑 |
| GPU 内核深度 | ~55% | Triton 简化版；无 MoonEP 级 TMA/warp 分工 |
| 训练路径（grad_reduce） | ~80% | Triton kernel + e2e；大 tensor i64 待 A100 |
| 性能 / Benchmark | ~55% | 2 卡 quick bench 已存档；无 DeepEP 对比、无 8-GPU full |

**整体（相对完整 MoonEP 移植）: ~85–90%**

---

## 已完成（本阶段可提交）

- [x] Phase 9–11：grad_reduce / dedup / planning GPU Triton 路径
- [x] Phase 12：8-rank smoke 测试（代码就绪，2 卡 deselect）、compile-all、`run_gpu_tests.sh`
- [x] Phase 13：benchmarks quick、pipeline mode（warp 扩展首版）
- [x] Bugfix：NVSHMEM 堆复用 padding NaN（e2e）、combine staging 测试、run_gpu_tests `set -u`
- [x] 目录重组：`moonep_td/` 与 MoonEP 同名模块对齐；`devtools/` 归集调试脚本
- [x] 文档：README / STATUS 安装步骤对齐 torch 2.6+cu124、triton-dist `-e python[build]` 与 triton 清理流程
- [x] 本机 2×V100 清单：`test_api_signatures` 5/5、`run_gpu_tests.sh` 支持 `$TRITON_DIST_ROOT`、`update_status.sh` filter 对齐
- [x] Bugfix：`bench_grad_reduce` / `bench_prefetch` 2 卡 `expert_plan` 越界；2 卡 quick benchmark 可跑

---

## 本机 2×V100 可执行清单

一键入口：`bash scripts/run_local_checklist.sh`（static + GPU + bench + 刷新 STATUS）。

| # | 项 | 命令 | 状态 |
|---|-----|------|------|
| 1 | 静态 + kernel compile | `bash scripts/static_check.sh` | ✅ 4 passed |
| 2 | 上游 API 签名（**可选**） | `pytest tests/test_api_signatures.py`（需单独装上游 `moonep`，见下） | ✅ 5 passed（本机已跑；不装则 skip） |
| 3 | GPU 默认回归 | `bash scripts/run_gpu_tests.sh` | ✅ 67 passed, 1 skipped |
| 4 | bench_comm quick | `torchrun --nproc_per_node=2 benchmarks/bench_comm.py --quick` | ✅ 见下表 |
| 5 | bench_comm pipeline | `MOONEP_TD_PIPELINE=1 torchrun … bench_comm.py --quick` | ✅ 见下表 |
| 6 | bench_grad_reduce quick | `torchrun --nproc_per_node=2 benchmarks/bench_grad_reduce.py` | ✅ 见下表 |
| 7 | bench_prefetch quick | `torchrun --nproc_per_node=2 benchmarks/bench_prefetch.py` | ✅ 见下表 |
| 8 | 刷新 STATUS CI 段 | `bash scripts/update_status.sh` | ✅ |
| 9 | `MOONEP_TD_PLANNING_TRITON=0` CPU fallback | 开发路径，按需 | ⬜ 未存档 |
| 10 | 移植 `bench_vs_deepep.py` | 依赖 DeepEP + 8 GPU | ⬜ 待租 8×A800 |
| 11 | `figure/` 插图脚本 | 低优先级 | ⬜ 未做 |

**上游 `moonep` 仅用于 API 签名校验（可选）**

- 安装：`pip install -e /path/to/MoonshotAI/MoonEP`（包名 `moonep`，**不是**本仓库的 `moonep_td`）
- 用途：只跑 `tests/test_api_signatures.py`，对比 `Buffer.dispatch/combine/...` 的参数列表
- 不装：GPU 回归、kernel 测试、benchmark **均不受影响**；`static_check.sh` 会 skip 并继续
- 副作用：会升级 `cuda-python`；签名校验后请 `pip install cuda-python==12.4.0` 恢复 triton-dist / nvshmem 栈

### 2×V100 性能基线（2026-07-30，ep=2 S=256 H=1024 K=4）

| 脚本 | 配置 | 指标 |
|------|------|------|
| `bench_comm.py --quick` | 默认 | planning 6346 µs, dispatch_fwd 448 µs, combine_fwd 185 µs (~4687 GB/s dispatch) |
| `bench_comm.py --quick` | `MOONEP_TD_PIPELINE=1` | planning 6872 µs, dispatch_fwd 395 µs, combine_fwd 219 µs (~5311 GB/s dispatch) |
| `bench_grad_reduce.py` | quick_2gpu | 1471 µs (~1069 GB/s effective) |
| `bench_prefetch.py` | quick_2gpu | 53 µs (~9922 GB/s effective) |

---

## 未做 / 待办（按优先级）

### P0 — 换机或扩硬件后应补跑

| 项 | 说明 | 如何验证 |
|----|------|----------|
| **8-GPU 规模** | `test_8rank_smoke`、8-rank dispatch/combine 路径 | `RUN_8RANK_TESTS=1 NPROC=8 bash scripts/run_gpu_tests.sh`（需 ≥8 GPU） |
| **4-GPU planning case** | `step1_segment_tail_full_tile`（min_R=4） | `NPROC=4 torchrun … pytest tests/test_planning.py -k step1_segment_tail` |
| **大 hidden（H=7168）** | NVSHMEM 显存压力；V100 32GB 默认 filter 排除 | `PYTEST_FILTER="not i64_offset and not 8rank_smoke" bash scripts/run_gpu_tests.sh`（需 A100 80GB 更稳） |
| **i64 字节偏移** | grad_reduce / prefetch 大 H×Hp 寻址 | `PYTEST_FILTER="" bash scripts/run_gpu_tests.sh`（需 A100 80GB） |

### P1 — 功能/parity 缺口（相对 MoonEP 原版）

| 项 | MoonEP | 当前 TD | 备注 |
|----|--------|---------|------|
| **Dispatch/Combine 内核** | CuTe TMA + warp 角色分工 | 简化 Triton tile | 功能对齐，非性能对齐 |
| **Hopper pipeline** | 完整 TMA producer/consumer | `MOONEP_TD_PIPELINE=1` 仅提高 warp 数 | Phase 13 首版，非最终形态 |
| **inter_rank_sync** | meta_buf CAS barrier | NVSHMEM + grid_sync | 语义够用，实现不同 |
| **csrc / VMM** | `bindings.cu` + IPC | 无（NVSHMEM 替代） | 见 `csrc/README.md`，不计划回退 VMM |
| **bench_vs_deepep.py** | 有 | **未移植** | 依赖 DeepEP + 8 GPU；租 8×A800 后移植 |
| **figure/** | `generate_buffer_figures.py` | **未移植** | 文档插图，低优先级 |

### P2 — 工程与 CI

| 项 | 状态 |
|----|------|
| **上游 API 签名对照** | ✅ 可选：`test_api_signatures.py`（需上游 `moonep`；见本机清单）；**非必需** |
| **STATUS 自动刷新** | ✅ `scripts/update_status.sh` filter 已与 `run_gpu_tests.sh` 对齐（`8rank_smoke`） |
| **远程 CI** | 无 GitHub Actions / 定时任务；本机用 `scripts/run_local_checklist.sh` |
| **性能基线存档** | ✅ 2 卡 quick 见上文「性能基线」；MoonEP/DeepEP 对比仍缺 |

### P3 — 已知限制（非 blocker）

- 包名 **`moonep_td`**（本仓库）；上游 pip 包为 **`moonep`**（Moonshot/MoonEP）— 二者并存时 import 名不同，避免冲突
- `run_gpu_tests.sh` 通过 **`$TRITON_DIST_ROOT`**（默认 `/root/Triton-distributed`）加载 `setenv.sh`
- `MOONEP_TD_PLANNING_TRITON=0` CPU fallback 依赖 `tests/planning_reference`（开发路径，非生产默认）
- V100 上 `large_hidden` / `i64_offset` **故意** 默认排除，非实现缺失
- CUDA 12.4 机器上 PyTorch [cu124 index](https://download.pytorch.org/whl/cu124) 最高 **2.6.0**；`moonep_td` 声明 `torch>=2.6,<2.7`
- PyTorch **2.12+** 需 cu126/cu130，与当前 cu124 + `nvidia-nvshmem-cu12==3.6.5` 栈不兼容；V100 亦须 cu126 而非 cu130

---

## 未在本机执行的 7 个用例说明

**1 skipped（会收集，运行时跳过）**

| 用例 | 原因 |
|------|------|
| `test_planning[step1_segment_tail_full_tile]` | case 要求 **R≥4**，当前 CI 用 2 卡 |

**6 deselected（`-k` filter 排除，未收集）**

| 用例 | 原因 |
|------|------|
| `test_8rank_smoke` ×2 | 需 8 GPU；`RUN_8RANK_TESTS=1` 单独跑 |
| `test_combine[large_hidden_stride_identity]` | H=7168，V100 OOM 风险 |
| `test_dispatch[large_hidden_stride_spotcheck]` | 同上 |
| `test_grad_reduce[i64_offset_7168x3072]` | 大 tensor 寻址，需 A100 |
| `test_prefetch[i64_offset_7168x3072]` | 同上 |

---

## 模块状态（2×V100 默认 filter）

| 模块 | 状态 | 测试 |
|------|------|------|
| `planning.py` | ✅ GPU Triton A–D | 18 pass, 1 skip |
| `dispatch*.py` + `_dedup_builder.py` | ✅ | 11 pass |
| `combine*.py` | ✅ | 13 pass |
| `prefetch.py` | ✅ | 9 pass |
| `grad_reduce.py` | ✅ | 11 pass |
| `api.py` / `buffer.py` | ✅ | e2e 1 pass |
| `test_pipeline_mode.py` | ✅ | 1 pass |
| `benchmarks/` | ✅ 2 卡 quick | 8-GPU `--full` 待 A800 |

---

## 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `MOONEP_TD_PLANNING_TRITON` | `1` | `1`=GPU planning；`0`=CPU reference |
| `MOONEP_TD_PIPELINE` | `0` | `1`=dispatch/combine/prefetch 高 warp |
| `MOONEP_NUM_SMS_DEDUP` | all SMs | dedup grid |
| `RUN_8RANK_TESTS` | `0` | `1` 且 ≥8 GPU 时跑 8-rank smoke |
| `NPROC` | `2` | `run_gpu_tests.sh` 的 torchrun 进程数 |
| `PYTEST_FILTER` | 见 `run_gpu_tests.sh` | 默认排除 large_hidden / i64 / 8rank |

---

## 建议下一步（提交后）

1. 在有 **8×GPU** 的机器上：`RUN_8RANK_TESTS=1 bash scripts/run_gpu_tests.sh`
2. 在有 **A100 80GB** 的机器上：`PYTEST_FILTER="" bash scripts/run_gpu_tests.sh`
3. 移植 `bench_vs_deepep.py` 并记录性能基线
4. Hopper 上评估完整 TMA pipeline（替代当前 warp-only `MOONEP_TD_PIPELINE`）

## Last CI run (auto)

- **Time**: 2026-07-30 02:58 UTC
- **GPUs**: 2 (`NPROC`)
- **Filter**: `not large_hidden and not i64_offset and not 8rank_smoke`
- **Result**: PASS (exit 0)
- **Summary**: 67 passed, none failed, 1 skipped
