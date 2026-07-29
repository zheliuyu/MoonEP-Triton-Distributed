# 目录结构对照：MoonEP vs MoonEP-Triton-D

本文档说明 TD 版本与 MoonEP 原版的目录对应关系，便于逐文件 diff。

## 顶层目录

| MoonEP | MoonEP-Triton-D | 说明 |
|--------|-----------------|------|
| `README.md` | `README.md` | 使用说明 |
| `LICENSE` | `LICENSE` | 许可证 |
| `setup.py` | `setup.py` + `pyproject.toml` | 构建入口 |
| — | `MIGRATION_PLAN.md` / `STATUS.md` | 迁移文档（TD 特有） |
| `moonep/` | `moonep_td/` | Python 包（`_td` 后缀区分 NVSHMEM 后端；避免与上游 `pip install moonep` 冲突） |
| `csrc/` | `csrc/README.md` | 原版 VMM C++；TD 用 NVSHMEM，无 `.cu` |
| `tests/` | `tests/` | pytest 回归（结构对齐 MoonEP） |
| `benchmarks/` | `benchmarks/` | 性能脚本 |
| — | `devtools/` | 迁移/调试脚本（TD 特有，见 `devtools/README.md`） |
| `scripts/` | `scripts/` | TD：仅 CI 入口（`run_gpu_tests.sh` 等） |
| `figure/` | — | 文档插图，暂未移植 |

## `moonep/` ↔ `moonep_td/` 模块对照

与 MoonEP **同名**的模块职责一致；Triton/NVSHMEM 实现细节用 `_` 前缀内部模块存放：

| MoonEP (`moonep/`) | MoonEP-Triton-D (`moonep_td/`) | 说明 |
|--------------------|--------------------------------|------|
| `__init__.py` | `__init__.py` | 导出 `Buffer`, `MoonEPCommPlan` |
| `api.py` | `api.py` | `_create_context` + `Buffer` 编排 |
| `buffer.py` | `buffer.py` | 对称内存（NVSHMEM 替代 VMM） |
| `constants.py` | `constants.py` | 常量 |
| `_common.py` | `_common.py` | grid sync / barrier 原语 |
| `planning.py` | `planning.py` | `MoonEPCommPlan` + `launch_planning` |
| `dispatch.py` | `dispatch.py` | dispatch kernel + 调用 dedup |
| `dispatch_epilogue.py` | `dispatch_epilogue.py` | dispatch epilogue |
| `combine_prologue.py` | `combine_prologue.py` | duplicate row 累加 |
| `combine.py` | `combine.py` | combine gather |
| `prefetch.py` | `prefetch.py` | expert weight prefetch |
| `grad_reduce.py` | `grad_reduce.py` | grad reduce |
| `inter_rank_sync.py` | `inter_rank_sync.py` | 跨 rank barrier |
| — | `_planning_gpu.py` | Triton planning Phase A–D（MoonEP 在 CuTe `planning.py` 内） |
| — | `_dedup_builder.py` | Triton dedup pass1/2（MoonEP 在 dispatch warps 3..） |
| — | `_pipeline.py` | `MOONEP_TD_PIPELINE` warp 扩展 |
| — | `_triton_runtime.py` | Triton-distributed 懒加载 |

**已移除的非 MoonEP 顶层文件**（重组后）：

- ~~`planning_triton.py`~~ — 别名，已删除
- ~~`planning_reference.py`~~ — 已移至 `tests/planning_reference.py`（与 MoonEP 一致）

## `tests/` 对照

| MoonEP | MoonEP-Triton-D | 说明 |
|--------|-----------------|------|
| `planning_reference.py` | `planning_reference.py` | CPU torch 参考实现 |
| `test_planning.py` … `test_e2e.py` | 同名 | 内核/API 回归 |
| — | `test_8rank_smoke.py` | 8-GPU smoke（2-GPU 跳过） |
| — | `test_kernel_compile.py` | 调用 `devtools/compile_kernels.py` |
| — | `test_pipeline_mode.py` | pipeline 模式 smoke |
| — | `test_api_signatures.py` | 与上游 `moonep` API 签名对照 |

## 脚本目录

```
MoonEP/          （无 scripts/）
MoonEP-Triton-D/
├── scripts/           # CI：run_gpu_tests.sh, static_check.sh, update_status.sh
└── devtools/          # 调试：compile_kernels
```

## 快速 diff

```bash
diff -u /root/MoonEP/moonep/dispatch.py /root/MoonEP-Triton-D/moonep_td/dispatch.py | less
diff -u /root/MoonEP/tests/planning_reference.py /root/MoonEP-Triton-D/tests/planning_reference.py | less
```
