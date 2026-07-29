# MoonEP → Triton-distributed 迁移计划（修订版）

**目标路径**: `/root/MoonEP-Triton-D`  
**源项目**: `/root/MoonEP` (CuTe DSL + CUDA VMM, ~5,500 LOC 内核)  
**底层栈**: `/root/Triton-distributed` (Triton + NVSHMEM)  
**当前快照**: 见 [`STATUS.md`](STATUS.md)（整体 ~55–60%）

**修订说明（2026-07-29）**: 原 Phase 0–7「骨架优先、GPU 最后」策略已基本完成 API/测试骨架；本版按**实际差距**重排后续工作，区分「语义完整」「GPU 内核化」「性能对等」三层目标。

---

## 0. 目标与约束

### 0.1 三层完成定义

| 层级 | 定义 | 当前 | 剩余 |
|------|------|------|------|
| **L1 语义完整** | API 对齐 + 多卡正确性 + e2e fwd/bwd | ~85% | grad_reduce GPU 化、8-rank、大 hidden |
| **L2 GPU 内核化** | 热路径无 CPU 同步 / Python 循环 | ~25% | planning、dedup、grad_reduce 真 kernel |
| **L3 性能对等** | TMA pipeline、benchmark 接近 MoonEP | ~10% | dispatch/combine/prefetch 重写 + bench |

**首版验收标准**: 达到 **L1**；**L2** 为下一阶段主目标；**L3** 为优化阶段。

### 0.2 功能目标（不变）

| 能力 | MoonEP | Triton-D 现状 |
|------|--------|---------------|
| Online Planning | 单 kernel Phase A–D | ⚠️ PyTorch GPU 循环 + all_gather |
| Dispatch | TMA + GPU dedup warps | ✅ 简化 Triton；⚠️ CPU dedup_builder |
| Combine | 3-stage warp TMA | ✅ 简化单阶段 Triton |
| Prefetch | persistent TMA 2D | ✅ 128×128 block copy |
| Grad Reduce | 539 行 tile kernel | ❌ Python 双重循环 |
| Buffer API | VMM + multicast | ✅ NVSHMEM；无 multicast |

### 0.3 非目标（保持不变）

- 不移植 CuTe DSL / CUTLASS / `moonep._C` VMM
- 不实现 PDL 链式 launch（v1 用 stream 顺序；`enable_pdl` 保留 no-op）
- 不支持 Zhenwu PPU / AMD / Ascend
- L1 阶段不追求逐 cycle 性能对齐

### 0.4 验证策略（修订）

```
已完成: Phase 0–7 骨架 + 2×V100 GPU 正确性（见 STATUS.md）
当前:   Phase 9–11 — GPU 内核化（L2）
后续:   Phase 12–13 — 规模验证 + 性能（L3）
```

不再采用「Phase 8 才跑 GPU」；GPU 测试为**持续回归**，每完成一个内核模块即扩展覆盖。

---

## 1. 已完成工作（Phase 0–7 归档）

> 以下项在原计划中为待办，**现已完成**，仅作归档对照。

| 原 Phase | 内容 | 状态 | 备注 |
|----------|------|------|------|
| 0 | 脚手架、pyproject、static_check | ✅ | 目录已对齐 MoonEP 扁平结构 |
| 1 | constants、layout、buffer | ✅ | layout 合入 `api._create_context` |
| 2 | barriers / grid_sync | ✅ | `_common.py`；无 meta_buf CAS barrier |
| 3 | planning | ⚠️ **过渡** | `planning_triton.py` 为 PyTorch，非 Triton JIT |
| 4 | dispatch + epilogue | ✅ 功能 | 984→138+58 行；dedup 外置 CPU |
| 5 | combine + prologue | ✅ 功能 | 1254→140 行 |
| 6 | prefetch + grad_reduce | ⚠️ 混合 | prefetch ✅；grad_reduce ❌ |
| 7 | API 集成 | ✅ | 351 行；签名对齐 |
| 8 | GPU 验证 | ⚠️ 部分 | 2-rank V100；skip 大 case |

### 1.1 实际目录结构（与计划差异）

```
moonep_td/
├── planning.py              # 入口 + MoonEPCommPlan
├── planning_triton.py       # ⚠️ 名不副实：GPU PyTorch，非 @triton.jit
├── planning_reference.py    # CPU/torch 参考 + src_info publish
├── dedup_builder.py         # ⚠️ CPU dedup（MoonEP 在 dispatch warps 3..）
├── dispatch.py              # Triton dispatch + zero_fill
├── dispatch_epilogue.py
├── combine_prologue.py
├── combine.py
├── prefetch.py
├── grad_reduce.py           # ❌ 待 GPU 化
├── inter_rank_sync.py
├── _common.py
├── _triton_runtime.py
├── buffer.py
├── api.py
└── constants.py

tests/                       # 扁平布局（非 tests/static/ + tests/gpu/）
scripts/run_gpu_tests.sh     # 默认 NPROC=2
```

### 1.2 代码量差距（内核模块）

| 模块 | MoonEP | Triton-D | 比率 |
|------|--------|----------|------|
| planning | 1,316 | 736 (155+252+329) | 56% 行数，~0% GPU kernel |
| dispatch + dedup | 984 | 227 (138+89) | 23% |
| combine 全家 | 1,654 | 197 | 12% |
| prefetch | 385 | 105 | 27% |
| grad_reduce | 539 | 81 | 15% |
| _common | 459 | 80 | 17% |
| **合计** | **~5,500** | **~1,200** | **~22% 深度** |

---

## 2. 架构映射（更新）

| 层次 | MoonEP | Triton-D 现状 | 目标 |
|------|--------|---------------|------|
| Kernel DSL | CuTe DSL | `@triton_dist.jit` + PyTorch 回退 | 热路径全 Triton |
| 对称内存 | CUDA VMM + IPC | NVSHMEM | 保持 NVSHMEM |
| Planning 广播 | NVSwitch multicast | `dist.all_gather` | L2: NVSHMEM put；L3: multimem 可选 |
| Dedup 构建 | dispatch warps 3.. | 独立 CPU `dedup_builder.py` | L2: Triton 或 dispatch 内联 |
| TMA | cp.async.bulk 全链路 | 无 | L3: TD TMA |
| PDL | 全链路 | no-op | 非目标 v1 |
| 跨 rank 同步 | meta_buf atomic CAS | grid_sync + NVSHMEM barrier | L2 可选加强 |

---

## 3. 后续阶段（Phase 9–13）

### Phase 9: Grad Reduce GPU 化（P0，预估 3–5 天）

**理由**: 训练 backward 唯一完全未 GPU 化的模块；MoonEP 539 行 vs TD 81 行 Python。

**源**: `/root/MoonEP/moonep/grad_reduce.py`

| 子任务 | MoonEP 特性 | Triton-D 目标 |
|--------|-------------|---------------|
| 9.1 Prescan expert 列表 | 压缩 `(src,b)` 有效 slot | `@triton.jit` 或 torch prescan + kernel |
| 9.2 128×128 tile reduce | 5 warps/CTA, TMA G2S | 首版: vectorized block reduce |
| 9.3 远程 buffer 读取 | peer NVL 地址 | `dl.symm_at` / `get_peer_tensor` |
| 9.4 清零 consumed slots | barrier 后 zero | Triton zero kernel |

**验收**:
- [ ] `test_grad_reduce.py` 全 case pass（含 `i64_offset`，需 A100 或更大显存）
- [ ] `num_sms` 参数生效
- [ ] 无 `for src in range(R): for b in range(B)` host 循环

**降级**: 若 Triton 远程读复杂，可先做 GPU prescan + 单线程 kernel launch per tile。

---

### Phase 10: GPU Dedup Builder（P0，预估 2–3 天）

**理由**: dispatch 热路径上 `meta_buf.cpu()` 是明确 sync 瓶颈。

**源**: MoonEP `dispatch.py` warps 3.. (`DEDUP_BUILDER_WARPS=4`)

| 子任务 | 说明 |
|--------|------|
| 10.1 读 `src_info` on GPU | 替代 `dedup_builder.py` L34 `meta[...].cpu()` |
| 10.2 primary_packed / kmask | GPU atomic 或 sort-reduce |
| 10.3 写 `dup_groups/loffs/counts` | 与 reference 语义一致 |
| 10.4 集成 | `build_dedup_map=True` 时 launch，可选并入 dispatch grid |

**验收**:
- [ ] `test_dispatch.py` dedup case pass
- [ ] `dedup_plan_semantic_errors` invariant pass
- [ ] profiling 无 `cudaMemcpy DtoH` on meta_buf

**策略**: 优先独立 Triton kernel（与 dispatch 解耦），稳定后再考虑 dispatch 内联（对齐 MoonEP）。

---

### Phase 11: Planning 真 GPU Kernel（P0，预估 5–8 天）

**理由**: 最大单文件差距（1,316 行 CuTe）；当前 `planning_triton.py` 仅为过渡。

**命名修正**: 完成后将 `planning_triton.py` 重命名或拆分为：
- `planning_gpu.py` — 真 `@triton.jit` cooperative kernel
- 保留 `planning_reference.py` — 永久 reference

| Phase | MoonEP | 实现策略 |
|-------|--------|----------|
| A | TPE gather | 首版保留 all_gather；kernel 内读对称 meta |
| B | surplus/deficit | block-0 serial 或 dedicated warp；最复杂 |
| C | layout + expert_off | 按 rank/expert 并行 |
| D | dst + src_info publish | token-parallel；peer write via NVSHMEM |

**分里程碑**:
- [ ] M11.1: Phase B `alloc` 与 reference 一致（GPU）
- [ ] M11.2: Phase C/D layout + dst_pos
- [ ] M11.3: `publish_src_info_to_meta` GPU 化（去 CPU loop）
- [ ] M11.4: `encode_dst_duplicates` GPU 化或 token-parallel Triton
- [ ] M11.5: 单 kernel 或多 phase launch 合并

**验收**: `test_planning.py` 18+ case；`MOONEP_TD_PLANNING_TRITON=1` 走真 kernel。

**风险**: Phase B while 循环 — 备选方案为 2-kernel launch（B 单独 serial kernel）。

---

### Phase 12: 规模与 CI 强化（P1，预估 2–3 天）

| 子任务 | 说明 |
|--------|------|
| 12.1 8-rank CI | `run_gpu_tests.sh` 默认或 matrix 支持 R=8 |
| 12.2 大 hidden | A100 80GB 跑 `large_hidden` / H=7168 |
| 12.3 i64 offset | prefetch + grad_reduce 大 buffer case |
| 12.4 静态工具 | `compile_kernels.py`、`test_kernel_compile.py`（原计划 Phase 3） |
| 12.5 STATUS 自动化 | CI 写入 `STATUS.md` 或 artifact |

**验收**: MoonEP 同 case 在 R=8 全绿；V100 上 heavy case 显式 skip 有文档。

---

### Phase 13: 性能优化与 Benchmark（P2，预估 5–10 天）

**前提**: L2（Phase 9–11）完成。

| 子任务 | MoonEP 源 | 目标 |
|--------|-----------|------|
| 13.1 Dispatch TMA pipeline | dispatch.py 984 行 | warp 0/1 producer/consumer |
| 13.2 Combine 3-stage | combine*.py | fp32 ACC warp 分工 |
| 13.3 Prefetch persistent TMA | prefetch.py | producer/consumer 2D pipeline |
| 13.4 Benchmark 移植 | benchmarks/*.py (~1,850 行) | vs MoonEP + DeepEP |
| 13.5 Figure 复现 | figure/ | 可选 |

**验收**: `bench_comm.py` 达到 MoonEP 同量级（允许 ±30% 首版）；README 补充 perf 数据。

**非目标**: PDL、multicast planning broadcast（除非 profiling 证明 all_gather 为瓶颈）。

---

## 4. 优先级总览（修订）

```
┌─────────────────────────────────────────────────────────────┐
│  P0 — L2 内核化（训练 + 去 CPU sync）                        │
│    9. grad_reduce Triton                                     │
│   10. GPU dedup builder                                      │
│   11. Planning 真 Triton kernel                                │
├─────────────────────────────────────────────────────────────┤
│  P1 — L1 补全 + CI                                           │
│   12. 8-rank / A100 大 case / compile-all CI                 │
├─────────────────────────────────────────────────────────────┤
│  P2 — L3 性能                                                  │
│   13. TMA pipeline + benchmarks                              │
├─────────────────────────────────────────────────────────────┤
│  非目标 v1                                                     │
│    PDL, VMM C++, CuTe, multicast, 多节点                    │
└─────────────────────────────────────────────────────────────┘
```

### 与原「降级路径」对照

| 原降级 | 现状 | 新策略 |
|--------|------|--------|
| P0: planning+dispatch+combine | ✅ 推理链路通 | 转向 L2 内核化 |
| P1: +prefetch | ✅ | 维持 |
| P2: +grad_reduce | ⚠️ ref only | **Phase 9 提升为 P0** |
| P3: TMA+bench | ❌ | Phase 13 |

---

## 5. 静态检查体系（修订）

### 5.1 现有

```bash
bash scripts/static_check.sh          # import + layout + reference
bash scripts/run_gpu_tests.sh         # 2×GPU 全模块
MOONEP_TD_PLANNING_TRITON=0 ...       # planning CPU reference 对照
```

### 5.2 待补

| 工具 | 用途 | Phase |
|------|------|-------|
| `scripts/compile_kernels.py` | 全部 `@triton_dist.jit` compile-only | 12 |
| `tests/test_kernel_compile.py` | CI 集成 | 12 |
| `benchmarks/bench_vs_moonep.py` | TD vs 原版 MoonEP | 13 |

---

## 6. 风险与缓解（更新）

| 风险 | 影响 | 缓解 |
|------|------|------|
| Planning Phase B 控制流 | 高 | 2-kernel 备选；reference 持续对照 |
| grad_reduce 远程 peer 读 | 中 | 先 prescan + 简化 tile；参考 MoonEP prescan |
| dedup GPU 原子竞争 | 中 | sort-based 备选 |
| V100 32GB OOM | 中 | 保持 pytest filter；A100 跑 full matrix |
| `planning_triton` 命名误导 | 低 | Phase 11 重命名 |
| 性能预期管理 | 中 | L1/L2/L3 分层验收，benchmark 仅 L3 |

---

## 7. 文件级对照表（更新状态）

| MoonEP | Triton-D | MoonEP LOC | TD LOC | 状态 |
|--------|----------|------------|--------|------|
| `constants.py` | `constants.py` | 17 | 9 | ✅ |
| `api.py` | `api.py` | 1083 | 351 | ✅ 功能 |
| `buffer.py` + `csrc/*` | `buffer.py` | 765 | 127 | ✅ NVSHMEM |
| `_common.py` | `_common.py` | 459 | 80 | ⚠️ |
| `planning.py` | `planning*.py` | 1316 | 736 | ⚠️ 过渡 |
| `dispatch.py` | `dispatch.py` + `dedup_builder.py` | 984 | ~250 | ⚠️ 功能 ✅；dedup GPU 化 |
| `dispatch_epilogue.py` | `dispatch_epilogue.py` | 416 | 58 | ✅ 简化 |
| `combine_prologue.py` | `combine_prologue.py` | 600 | 54 | ✅ 简化 |
| `combine.py` | `combine.py` | 654 | 84 | ✅ 简化 |
| `prefetch.py` | `prefetch.py` | 385 | 105 | ✅ 简化 |
| `grad_reduce.py` | `grad_reduce.py` | 539 | ~290 | ✅ Phase 9 Triton |
| `inter_rank_sync.py` | `inter_rank_sync.py` | 155 | 11 | ⚠️ |
| `tests/*` | `tests/*` | ~2700 | ~2700 | ✅ 移植 |
| `benchmarks/*` | — | ~1850 | 0 | ❌ |

---

## 8. 时间估算（修订，剩余工作）

| Phase | 内容 | 预估 | 累计 |
|-------|------|------|------|
| ~~0–8~~ | 骨架 + 2-GPU 验证 | ~~已完成~~ | — |
| 9 | grad_reduce GPU | 3–5d | ✅ 完成 |
| 10 | GPU dedup builder | 2–3d | ✅ 完成 |
| 11 | Planning 真 kernel | 5–8d | 16d |
| 12 | 8-rank CI + 大 case | 2–3d | 19d |
| 13 | TMA + benchmark | 5–10d | **29d** |

单人全职：**L2 约 3 周，L3 再 +1.5–2 周**。

---

## 9. 下一步行动

1. **Phase 9 启动** — 阅读 MoonEP `grad_reduce.py`，设计 Triton tile kernel + prescan
2. **并行** — Phase 10 dedup builder 可与 grad_reduce 分工
3. **Phase 11 前** — 将 `planning_triton.py` 文档标注为「过渡实现」；避免新代码依赖其 PyTorch 循环
4. **持续** — 每 phase 合并后跑 `run_gpu_tests.sh` 全模块回归

---

## 附录 A: meta_buf 布局速查

（不变，见原版）

```
meta_buf (per-rank chunk, int32 elements):
  [0, NvS)                  weights (fp32 as int32 alias)
  [NvS, NvS + R*E)          tpe gather
  [..., +planning_scratch)  planning internal
  [..., +N4)                topk0 offload
  [..., +N4)                ORDER
  [..., +N4)                ORDER0
  [..., +3)                 cross-rank barrier
  [..., +NvS)               src_info (dedup builder scratch)
```

## 附录 B: 数据流

```
topk + tpe
    → planning → plan.dst (neg=duplicate)
    → dispatch (copy primary, scatter weights)
    → dedup_builder [TD: 外置 CPU，目标: GPU]
    → dispatch_epilogue (expand duplicates in-place)
    → [prefetch_weight]
    → expert GEMM (framework)
    → combine_prologue (reduce duplicates)
    → combine (K-sum to [S,H])
    → reduce_grad [TD: Python，目标: Triton]
```

## 附录 C: 依赖初始化

```python
import torch
import torch.distributed as dist
from moonep_td.buffer import ensure_nvshmem_initialized
from moonep_td import Buffer

dist.init_process_group("nccl")
ensure_nvshmem_initialized()

buffer = Buffer(S=4096, H=7168, K=8, E=256, num_ep_ranks=8)
```

## 附录 D: 差距分析来源

本修订基于 2026-07-29 对 `/root/MoonEP` 与 `/root/MoonEP-Triton-D` 的逐模块对比：
- 内核 LOC 比 ~22%
- 2×V100 测试已通过模块见 `STATUS.md`
- 主要缺口: planning GPU kernel、dedup CPU sync、grad_reduce 未 GPU 化、无 benchmark
