"""Gather + K-sum back to token-major layout."""

from __future__ import annotations

import functools

import torch

from moonep_td._common import cached_block_h, launch_cross_rank_barrier
from moonep_td._pipeline import pipeline_enabled


@functools.lru_cache(maxsize=None)
def _kernel():
    import triton.language as tl
    import triton_dist.language as dl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def combine_kernel(
        output_ptr, output_w_ptr, hidden_buf_ptr, weights_meta_ptr, dst_ptr,
        stride_out_h, stride_buf_row, S, K, H, NvS, NvS_padded,
        meta_stride, weights_off, WITH_WEIGHTS: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        token = tl.program_id(0)
        if token >= S:
            return
        dst_row = output_ptr + token * stride_out_h
        for h_off in range(0, H, BLOCK_H):
            cols = h_off + tl.arange(0, BLOCK_H)
            mask = cols < H
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for k in range(K):
                slot = token * K + k
                dst_val = tl.load(dst_ptr + slot)
                raw = -dst_val - 1 if dst_val < 0 else dst_val
                dest_rank = raw // NvS
                loff = raw - dest_rank * NvS
                if WITH_WEIGHTS:
                    remote_meta = dl.symm_at(weights_meta_ptr, dest_rank)
                    tl.store(
                        output_w_ptr + token * K + k,
                        tl.load(remote_meta + dest_rank * meta_stride + weights_off + loff),
                    )
                if dst_val >= 0:
                    remote_buf = dl.symm_at(hidden_buf_ptr, dest_rank)
                    src_row = remote_buf + (dest_rank * NvS_padded + loff) * stride_buf_row
                    acc += tl.load(src_row + cols, mask=mask, other=0).to(tl.float32)
            tl.store(dst_row + cols, acc.to(tl.bfloat16), mask=mask)

    return combine_kernel


def _pipeline_num_warps(default: int) -> int:
    return 8 if pipeline_enabled() else default


def launch_combine(
    ctx: dict,
    output_sh: torch.Tensor,
    dst: torch.Tensor,
    output_sk: torch.Tensor | None = None,
    *,
    pdl_launch: bool = False,
) -> None:
    del pdl_launch
    launch_cross_rank_barrier(ctx)
    assert output_sh.dtype == torch.bfloat16 and output_sh.is_contiguous(), "output_sh must be contiguous bf16"
    assert dst.dtype == torch.int32 and dst.is_contiguous(), "dst must be contiguous int32"
    if output_sk is not None:
        assert output_sk.dtype == torch.float32 and output_sk.is_contiguous(), \
            "output_sk must be contiguous fp32"
        assert tuple(output_sk.shape) == (int(ctx["S"]), int(ctx["K"])), "output_sk shape mismatch"
    combine_kernel = _kernel()
    S, H, K = int(ctx["S"]), int(ctx["H"]), int(ctx["K"])
    NvS, NvS_padded = int(ctx["NvS"]), int(ctx["NvS_padded"])
    meta_stride, weights_off = int(ctx["meta_chunk_padded"]), int(ctx["WEIGHTS_OFF"])
    with_weights = output_sk is not None
    if not with_weights:
        output_sk = dst
    combine_kernel[(S,)](
        output_sh, output_sk, ctx["hidden_buf"], ctx["meta_buf"].view(torch.float32), dst,
        output_sh.stride(0), ctx["hidden_buf"].stride(0),
        S, K, H, NvS, NvS_padded,
        meta_stride, weights_off, WITH_WEIGHTS=with_weights,
        BLOCK_H=cached_block_h(H), num_warps=_pipeline_num_warps(4),
    )
