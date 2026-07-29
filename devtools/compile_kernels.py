#!/usr/bin/env python3
"""Compile all MoonEP-Triton-D Triton kernels (no multi-GPU required)."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Minimal env for triton_dist import on CPU-only compile hosts.
os.environ.setdefault("CUDA_HOME", os.environ.get("CUDA_HOME", "/usr/local/cuda-12.4"))


def _compile_planning(R: int) -> None:
    from moonep_td._planning_gpu import (
        _balance_distribute_kernel,
        _compute_dst_kernel,
        _publish_dst_scratch_kernel,
        _publish_src_info_kernel,
        _publish_tpe_kernel,
    )

    _publish_src_info_kernel()
    _publish_tpe_kernel()
    _publish_dst_scratch_kernel()
    _compute_dst_kernel(R)
    _balance_distribute_kernel(R)


def _compile_dispatch() -> None:
    from moonep_td.dispatch import _kernels

    _kernels()


def _compile_rest() -> None:
    from moonep_td.combine import _kernel
    from moonep_td.combine_prologue import _kernel as _cp
    from moonep_td._dedup_builder import _kernels as _dedup
    from moonep_td.dispatch_epilogue import _kernel as _de
    from moonep_td.grad_reduce import _kernels as _gr
    from moonep_td.prefetch import _kernel as _prefetch_k
    from moonep_td._common import _grid_sync_kernel_fn

    _dedup()
    _de()
    _cp()
    _kernel()
    _gr()
    _prefetch_k()
    _grid_sync_kernel_fn()


def main() -> int:
    if not torch_available():
        print("skip compile_kernels: no CUDA device")
        return 0

    import torch

    if not torch.cuda.is_available():
        print("skip compile_kernels: torch.cuda unavailable")
        return 0

    print("Compiling MoonEP-Triton-D Triton kernels...")
    for R in (1, 2, 4, 8):
        _compile_planning(R)
    _compile_dispatch()
    _compile_rest()
    print("All kernels compiled OK")
    return 0


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
