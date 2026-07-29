#!/usr/bin/env bash
# 2×V100 本机可执行项一键回归（不含 P0 大显存/8-rank 用例）。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MOONEP_TD_ROOT="${MOONEP_TD_ROOT:-$ROOT}"
export TRITON_DIST_ROOT="${TRITON_DIST_ROOT:-/root/Triton-distributed}"
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

if [[ -f "${TRITON_DIST_ROOT}/scripts/setenv.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${TRITON_DIST_ROOT}/scripts/setenv.sh"
  set -u
fi
export PYTHONPATH="${MOONEP_TD_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "== [1/4] static_check (layout + kernel compile; optional upstream moonep API check) =="
bash scripts/static_check.sh

echo "== [2/4] GPU regression (67 passed baseline) =="
bash scripts/run_gpu_tests.sh

echo "== [3/4] 2-GPU quick benchmarks =="
torchrun --nproc_per_node=2 benchmarks/bench_comm.py --quick
MOONEP_TD_PIPELINE=1 torchrun --nproc_per_node=2 benchmarks/bench_comm.py --quick
torchrun --nproc_per_node=2 benchmarks/bench_grad_reduce.py
torchrun --nproc_per_node=2 benchmarks/bench_prefetch.py

echo "== [4/4] refresh STATUS Last CI run =="
bash scripts/update_status.sh

echo "LOCAL CHECKLIST PASSED"
