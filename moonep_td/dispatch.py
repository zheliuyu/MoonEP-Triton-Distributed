"""MoonEP Dispatch — Triton-distributed implementation."""

from __future__ import annotations

import functools

import torch

from moonep_td._common import cached_block_h, launch_cross_rank_barrier
from moonep_td._dedup_builder import launch_dedup_builder
from moonep_td._pipeline import pipeline_enabled
from moonep_td.planning import MoonEPCommPlan


def _check_dispatch_plan(ctx: dict, hidden_sh: torch.Tensor, plan: MoonEPCommPlan) -> None:
    _, K, N = int(ctx["S"]), int(ctx["K"]), int(ctx["S"]) * int(ctx["K"])
    R, E, B, NvS = int(ctx["R"]), int(ctx["E"]), int(ctx.get("B", 0)), int(ctx["NvS"])
    dev = hidden_sh.device
    assert plan.N == N and plan.R == R and plan.K == K and plan.NvS == NvS
    for name, shape in (
        ("dst", (N,)),
        ("zero_fill_ranges", (E + B, 2)),
        ("dup_groups", (NvS, 3)),
        ("dup_loffs", (NvS,)),
        ("dup_counts", (2,)),
    ):
        t = getattr(plan, name)
        assert t.dtype == torch.int32 and t.is_contiguous(), f"{name} must be contiguous int32"
        assert tuple(t.shape) == shape, f"{name} shape {tuple(t.shape)} != {shape}"
        assert t.device == dev, f"{name} device mismatch"


@functools.lru_cache(maxsize=None)
def _kernels():
    import triton.language as tl
    import triton_dist.language as dl

    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def dispatch_kernel(
        hidden_sh_ptr,
        hidden_buf_ptr,
        weights_meta_ptr,
        route_w_ptr,
        dst_ptr,
        stride_sh_h,
        stride_buf_row,
        K,
        H,
        N,
        NvS,
        NvS_padded,
        meta_stride,
        weights_off,
        WITH_WEIGHTS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        slot = tl.program_id(0)
        if slot >= N:
            return
        dst_val = tl.load(dst_ptr + slot)
        token = slot // K
        src_row = hidden_sh_ptr + token * stride_sh_h
        raw = dst_val if dst_val >= 0 else -dst_val - 1
        dest_rank = raw // NvS
        loff = raw - dest_rank * NvS
        if dst_val >= 0:
            remote_buf = dl.symm_at(hidden_buf_ptr, dest_rank)
            row_idx = (dest_rank * NvS_padded + loff).to(tl.int64)
            stride_buf_row64 = stride_buf_row.to(tl.int64)
            dst_row = remote_buf + row_idx * stride_buf_row64
            for h_off in range(0, H, BLOCK_H):
                cols = h_off + tl.arange(0, BLOCK_H)
                mask = cols < H
                vals = tl.load(src_row + cols, mask=mask, other=0).to(tl.bfloat16)
                tl.store(dst_row + cols, vals, mask=mask)
        if WITH_WEIGHTS:
            remote_meta = dl.symm_at(weights_meta_ptr, dest_rank)
            w_val = tl.load(route_w_ptr + slot)
            tl.store(remote_meta + dest_rank * meta_stride + weights_off + loff, w_val)

    @td.jit
    def zero_fill_kernel(
        hidden_buf_ptr,
        weights_meta_ptr,
        zero_fill_ptr,
        stride_buf_row,
        rank,
        NvS_padded,
        H,
        num_groups,
        meta_stride,
        weights_off,
        WITH_WEIGHTS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        g = tl.program_id(0)
        if g >= num_groups:
            return
        pad_start = tl.load(zero_fill_ptr + g * 2)
        n_pad = tl.load(zero_fill_ptr + g * 2 + 1)
        if n_pad <= 0:
            return
        stride_buf_row64 = stride_buf_row.to(tl.int64)
        row_base = (rank * NvS_padded + pad_start).to(tl.int64) * stride_buf_row64
        for row in range(n_pad):
            dst_row = hidden_buf_ptr + row_base + row.to(tl.int64) * stride_buf_row64
            for h_off in range(0, H, BLOCK_H):
                cols = h_off + tl.arange(0, BLOCK_H)
                mask = cols < H
                tl.store(dst_row + cols, tl.zeros([BLOCK_H], dtype=tl.bfloat16), mask=mask)
            if WITH_WEIGHTS:
                loff = pad_start + row
                tl.store(weights_meta_ptr + rank * meta_stride + weights_off + loff, 0.0)

    return dispatch_kernel, zero_fill_kernel


def _pipeline_num_warps(default: int) -> int:
    return 8 if pipeline_enabled() else default


def launch_dispatch(
    ctx: dict,
    hidden_sh: torch.Tensor,
    route_weights_sk: torch.Tensor | None,
    plan: MoonEPCommPlan,
    *,
    build_dedup_map: bool = False,
    pdl_trigger: bool = False,
) -> None:
    del pdl_trigger
    assert hidden_sh.dtype == torch.bfloat16 and hidden_sh.is_contiguous(), "hidden_sh must be contiguous bf16"
    assert tuple(hidden_sh.shape) == (int(ctx["S"]), int(ctx["H"])), "hidden_sh shape mismatch"
    if route_weights_sk is not None:
        assert route_weights_sk.dtype == torch.float32 and route_weights_sk.is_contiguous(), (
            "route_weights_sk must be contiguous fp32"
        )
        assert tuple(route_weights_sk.shape) == (int(ctx["S"]), int(ctx["K"])), "route_weights_sk shape mismatch"
    assert plan.dst.dtype == torch.int32, "dst must be int32"
    _check_dispatch_plan(ctx, hidden_sh, plan)
    dispatch_kernel, zero_fill_kernel = _kernels()
    H, K, N = int(ctx["H"]), int(ctx["K"]), int(ctx["N"])
    NvS, NvS_padded = int(ctx["NvS"]), int(ctx["NvS_padded"])
    meta_stride = int(ctx["meta_chunk_padded"])
    weights_off = int(ctx["WEIGHTS_OFF"])
    with_weights = route_weights_sk is not None
    if not with_weights:
        route_weights_sk = plan.dst
    w_meta = ctx["meta_buf"].view(torch.float32)
    BLOCK_H = cached_block_h(H)
    nw = _pipeline_num_warps(4)
    dispatch_kernel[(N,)](
        hidden_sh,
        ctx["hidden_buf"],
        w_meta,
        route_weights_sk,
        plan.dst,
        hidden_sh.stride(0),
        ctx["hidden_buf"].stride(0),
        K,
        H,
        N,
        NvS,
        NvS_padded,
        meta_stride,
        weights_off,
        WITH_WEIGHTS=with_weights,
        BLOCK_H=BLOCK_H,
        num_warps=nw,
    )
    num_groups = int(plan.zero_fill_ranges.shape[0])
    zero_fill_kernel[(num_groups,)](
        ctx["hidden_buf"],
        w_meta,
        plan.zero_fill_ranges,
        ctx["hidden_buf"].stride(0),
        int(ctx["rank"]),
        NvS_padded,
        H,
        num_groups,
        meta_stride,
        weights_off,
        WITH_WEIGHTS=with_weights,
        BLOCK_H=BLOCK_H,
        num_warps=1,
    )
    if build_dedup_map:
        launch_dedup_builder(ctx, plan)
    launch_cross_rank_barrier(ctx)
