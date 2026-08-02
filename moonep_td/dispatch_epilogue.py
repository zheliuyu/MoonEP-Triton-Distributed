"""In-place duplicate expansion on the local NVL shard."""

from __future__ import annotations

import functools

import torch

from moonep_td._common import cached_block_h
from moonep_td.planning import MoonEPCommPlan


@functools.lru_cache(maxsize=None)
def _kernel():
    import triton.language as tl

    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def epilogue_kernel(
        hidden_local_ptr, dup_groups_ptr, dup_loffs_ptr, dup_counts_ptr, stride_h, H, BLOCK_H: tl.constexpr
    ):
        g = tl.program_id(0)
        n_groups = tl.load(dup_counts_ptr)
        if g >= n_groups:
            return
        primary_loff = tl.load(dup_groups_ptr + g * 3)
        dup_start = tl.load(dup_groups_ptr + g * 3 + 1)
        dup_n = tl.load(dup_groups_ptr + g * 3 + 2)
        stride_h64 = stride_h.to(tl.int64)
        src_row = hidden_local_ptr + primary_loff.to(tl.int64) * stride_h64
        for d in range(dup_n):
            dup_loff = tl.load(dup_loffs_ptr + dup_start + d)
            dst_row = hidden_local_ptr + dup_loff.to(tl.int64) * stride_h64
            for h_off in range(0, H, BLOCK_H):
                cols = h_off + tl.arange(0, BLOCK_H)
                mask = cols < H
                tl.store(dst_row + cols, tl.load(src_row + cols, mask=mask, other=0), mask=mask)

    return epilogue_kernel


def launch_dispatch_epilogue(ctx: dict, plan: MoonEPCommPlan, *, pdl_launch: bool = False) -> None:
    del pdl_launch
    H, NvS = int(ctx["H"]), int(ctx["NvS"])
    dev = ctx["hidden_buf_local"].device
    assert isinstance(plan, MoonEPCommPlan)
    assert plan.dup_loffs.dtype == torch.int32 and plan.dup_loffs.is_contiguous()
    assert tuple(plan.dup_loffs.shape) == (NvS,), "dup_loffs shape mismatch"
    assert plan.dup_loffs.device == dev, "dup_loffs device mismatch"
    n_groups = int(plan.dup_counts[0].item()) if plan.dup_counts.numel() else 0
    if n_groups <= 0:
        return
    H = int(ctx["H"])
    epilogue_kernel = _kernel()
    epilogue_kernel[(n_groups,)](
        ctx["hidden_buf_local"],
        plan.dup_groups,
        plan.dup_loffs,
        plan.dup_counts,
        ctx["hidden_buf_local"].stride(0),
        H,
        BLOCK_H=cached_block_h(H),
        num_warps=4,
    )
