#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libstdc++.so.6}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"

export NVSHMEM_SYMMETRIC_SIZE="${NVSHMEM_SYMMETRIC_SIZE:-34359738368}"

TRITON_DIST_ROOT="${TRITON_DIST_ROOT:-/root/Triton-distributed}"
if [[ -f "${TRITON_DIST_ROOT}/scripts/setenv.sh" ]]; then
  set +u
  # shellcheck disable=SC1091
  source "${TRITON_DIST_ROOT}/scripts/setenv.sh"
  set -u
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"

NPROC="${NPROC:-}"
if [[ -z "${NPROC}" ]]; then
  if [[ "${GPU_COUNT}" -ge 4 ]]; then
    NPROC=4
  elif [[ "${GPU_COUNT}" -ge 2 ]]; then
    NPROC=2
  else
    NPROC="${GPU_COUNT:-1}"
  fi
fi
MASTER_PORT="${MASTER_PORT:-$((29600 + RANDOM % 1000))}"
# Full suite (large_hidden + i64_offset included). 8-rank: test_8rank_smoke.py + RUN_8RANK_TESTS.
PYTEST_FILTER="${PYTEST_FILTER:-}"
# Per-test skip via tests/conftest.py (SIGALRM). Set 0 for full MoonEP-parity runs on 4 GPUs.
GPU_TEST_TIMEOUT_SEC="${GPU_TEST_TIMEOUT_SEC:-0}"
GPU_TEST_TIMEOUT_SKIP="${GPU_TEST_TIMEOUT_SKIP:-1}"
export GPU_TEST_TIMEOUT_SEC GPU_TEST_TIMEOUT_SKIP

run_tests() {
  local -a args=(
    tests/ -q
    --ignore=tests/test_api_signatures.py
    --ignore=tests/test_kernel_compile.py
    --ignore=tests/test_8rank_smoke.py
  )
  if [[ -n "${PYTEST_FILTER}" ]]; then
    args+=(-k "${PYTEST_FILTER}")
  fi
  # Per-test skip via tests/conftest.py (SIGALRM → pytest.skip). Do not use
  # pytest-timeout here: its thread mode calls os._exit and dumps stacks under torchrun.
  if [[ -n "${GPU_TEST_TIMEOUT_SEC}" ]] && [[ "${GPU_TEST_TIMEOUT_SEC}" != "0" ]]; then
    args+=(-p no:timeout)
  fi
  torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" -m pytest \
    "${args[@]}" \
    "$@"
}

echo "== GPU tests: NPROC=${NPROC} visible_gpus=${GPU_COUNT} NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE} GPU_TEST_TIMEOUT_SEC=${GPU_TEST_TIMEOUT_SEC} RUN_SLOW_GPU_TESTS=${RUN_SLOW_GPU_TESTS:-0} ${PYTEST_FILTER:+(filter=${PYTEST_FILTER})} =="
if [[ "${RUN_SLOW_GPU_TESTS:-0}" != "1" ]]; then
  echo "== slow dispatch large_hidden test skipped (RUN_SLOW_GPU_TESTS=1 to enable) =="
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
