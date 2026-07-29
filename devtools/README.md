# Devtools — 迁移/调试脚本

本目录存放 **MoonEP-Triton-D 迁移过程中形成的验证与调试脚本**，与 MoonEP 上游仓库的 `scripts/` 目录不同（上游无此目录）。正式 CI 入口保留在 [`scripts/`](../scripts/)。

## 脚本说明

| 文件 | 用途 | 典型用法 |
|------|------|----------|
| `compile_kernels.py` | 预编译全部 Triton `@td.jit` kernel（R∈{1,2,4,8}） | `python devtools/compile_kernels.py` |

## 与 `scripts/` 的分工

| 目录 | 内容 |
|------|------|
| [`scripts/`](../scripts/) | **CI / 发布**：`run_gpu_tests.sh`、`static_check.sh`、`update_status.sh` |
| `devtools/` | **开发调试**：kernel 预编译等 ad-hoc 工具 |

## 与 `tests/` 的分工

| 目录 | 内容 |
|------|------|
| [`tests/`](../tests/) | pytest 回归（与 MoonEP `tests/` 对齐） |
| `devtools/` | 非 pytest 的 ad-hoc 脚本；`tests/test_kernel_compile.py` 会调用 `compile_kernels.py` |
