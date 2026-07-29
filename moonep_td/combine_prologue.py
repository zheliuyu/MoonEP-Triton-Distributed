"""FP32 accumulation of duplicate rows into primary slots."""

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
    def prologue_kernel(hidden_local_ptr, dup_groups_ptr, dup_loffs_ptr, dup_counts_ptr,
                        stride_h, H, BLOCK_H: tl.constexpr):
        g = tl.program_id(0)
        n_groups = tl.load(dup_counts_ptr)
        if g >= n_groups:
            return
        primary_loff = tl.load(dup_groups_ptr + g * 3)
        dup_start = tl.load(dup_groups_ptr + g * 3 + 1)
        dup_n = tl.load(dup_groups_ptr + g * 3 + 2)
        dst_row = hidden_local_ptr + primary_loff * stride_h
        for d in range(dup_n):
            dup_loff = tl.load(dup_loffs_ptr + dup_start + d)
            src_row = hidden_local_ptr + dup_loff * stride_h
            for h_off in range(0, H, BLOCK_H):
                cols = h_off + tl.arange(0, BLOCK_H)
                mask = cols < H
                acc = tl.load(dst_row + cols, mask=mask, other=0).to(tl.float32)
                acc += tl.load(src_row + cols, mask=mask, other=0).to(tl.float32)
                tl.store(dst_row + cols, acc.to(tl.bfloat16), mask=mask)

    return prologue_kernel


def launch_combine_prologue(ctx: dict, plan: MoonEPCommPlan, *, pdl_trigger: bool = False) -> None:
    del pdl_trigger
    n_groups = int(plan.dup_counts[0].item()) if plan.dup_counts.numel() else 0
    if n_groups <= 0:
        return
    H = int(ctx["H"])
    prologue_kernel = _kernel()
    prologue_kernel[(n_groups,)](
        ctx["hidden_buf_local"], plan.dup_groups, plan.dup_loffs, plan.dup_counts,
        ctx["hidden_buf_local"].stride(0), H, BLOCK_H=cached_block_h(H), num_warps=4,
    )
