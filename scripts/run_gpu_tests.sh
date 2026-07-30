#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

NPROC="${NPROC:-2}"
MASTER_PORT="${MASTER_PORT:-$((29600 + RANDOM % 1000))}"
# V100 32GB: skip H=7168 NVSHMEM OOM + 8-rank-only smoke on 2-GPU machines.
PYTEST_FILTER="${PYTEST_FILTER:-not large_hidden and not i64_offset and not 8rank_smoke}"

# large_hidden / i64 (H=7168) need large symmetric heap. setenv.sh omits this.
# i64 prefetch test allocates ~18 GiB once (not 4×); leave headroom for prior tests.
if [[ "${PYTEST_FILTER}" != *"not large_hidden"* ]] \
   || [[ "${PYTEST_FILTER}" != *"not i64_offset"* ]]; then
  export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-34359738368}"
fi

TRITON_DIST_ROOT="${TRITON_DIST_ROOT:-/root/Triton-distributed}"
if [[ -f "${TRITON_DIST_ROOT}/scripts/setenv.sh" ]]; then
  # setenv.sh reads $NVSHMEM_HOME before assigning it; disable nounset while sourcing.
  set +u
  # shellcheck disable=SC1091
  source "${TRITON_DIST_ROOT}/scripts/setenv.sh"
  set -u
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"

run_tests() {
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" -m pytest tests/ -q \
    --ignore=tests/test_api_signatures.py \
    --ignore=tests/test_kernel_compile.py \
    -k "${PYTEST_FILTER}" \
    "$@"
}

if [[ -n "${NVSHMEM_SYMMETRIC_SIZE:-}" ]]; then
  echo "== GPU tests: NPROC=${NPROC} visible_gpus=${GPU_COUNT} NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE} filter='${PYTEST_FILTER}' =="
else
  echo "== GPU tests: NPROC=${NPROC} visible_gpus=${GPU_COUNT} filter='${PYTEST_FILTER}' =="
fi
run_tests "$@"

if [[ "${RUN_8RANK_TESTS:-0}" == "1" ]]; then
  if [[ "${GPU_COUNT}" -ge 8 ]]; then
    echo "== 8-rank smoke (RUN_8RANK_TESTS=1, GPUs=${GPU_COUNT}) =="
    EIGHT_PORT=$((MASTER_PORT + 1))
    torchrun --nproc_per_node=8 --master_port="${EIGHT_PORT}" -m pytest tests/test_8rank_smoke.py -v
  else
    echo "== skip 8-rank smoke: need >=8 GPUs, have ${GPU_COUNT} =="
  fi
else
  echo "== 8-rank smoke skipped (set RUN_8RANK_TESTS=1 on 8-GPU hosts) =="
fi
