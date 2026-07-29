"""Lazy grid-sync kernel (imports triton_dist on first use)."""

from __future__ import annotations

import functools

from moonep_td.constants import GRID_SYNC_TAG, WARP_SYNC_TAG

__all__ = [
    "GRID_SYNC_TAG",
    "WARP_SYNC_TAG",
    "block_h",
    "cached_block_h",
    "launch_cooperative_opts",
    "launch_cross_rank_barrier",
    "launch_grid_sync",
]


def block_h(H: int) -> int:
    for b in (128, 64, 32, 16, 8):
        if H % b == 0:
            return b
    return 8


@functools.lru_cache(maxsize=None)
def cached_block_h(H: int) -> int:
    return block_h(H)


def launch_cooperative_opts() -> dict:
    try:
        from triton_dist.utils import launch_cooperative_grid_options
        return launch_cooperative_grid_options()
    except Exception:
        return {}


@functools.lru_cache(maxsize=1)
def _grid_sync_kernel_fn():
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def grid_sync_kernel(bar_ptr):
        pid = tl.program_id(0)
        num_pid = tl.num_programs(0)
        if pid == 0:
            nb = tl.cast(0x80000000, tl.uint32, bitcast=True) - tl.cast(num_pid - 1, tl.uint32)
            tl.atomic_add(bar_ptr, nb, sem="release", scope="gpu")
        else:
            tl.atomic_add(bar_ptr, 1, sem="release", scope="gpu")
        expected = tl.cast(0x80000000, tl.uint32, bitcast=True)
        while tl.atomic_cas(bar_ptr, expected, expected, sem="acquire", scope="gpu") != expected:
            pass
        tl.atomic_xchg(bar_ptr, 0)

    return grid_sync_kernel


def launch_grid_sync(ctx: dict) -> None:
    try:
        kernel = _grid_sync_kernel_fn()
    except ImportError:
        return
    num_sms = max(1, int(ctx.get("num_sms", 1)))
    kernel[(num_sms,)](ctx["grid_sync_bar"], **launch_cooperative_opts())


def launch_cross_rank_barrier(ctx: dict) -> None:
    """Publish NVSHMEM/NVL writes across ranks (MoonEP cross_rank_barrier)."""
    launch_grid_sync(ctx)
    from triton_dist.utils import nvshmem_barrier_all_on_stream
    import torch

    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    launch_grid_sync(ctx)
