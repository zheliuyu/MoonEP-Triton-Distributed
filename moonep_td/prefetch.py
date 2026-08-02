"""Remote expert weight prefetch into local slots [E, E+B)."""

from __future__ import annotations

import functools

import torch

from moonep_td._pipeline import pipeline_enabled

_TILE = 128


@functools.lru_cache(maxsize=None)
def _kernel():
    import triton.language as tl

    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def prefetch_kernel(
        remote_ptr,
        prefetch_ptr,
        experts_ptr,
        stride_e,
        stride_h,
        stride_b,
        H,
        Hp,
        B,
        tiles_m,
        tiles_n,
        NUM_SMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        total = B * tiles_m * tiles_n
        for work in range(pid, total, NUM_SMS):
            b = work // (tiles_m * tiles_n)
            rem = work % (tiles_m * tiles_n)
            tm = rem // tiles_n
            tn = rem % tiles_n
            expert = tl.load(experts_ptr + b).to(tl.int32)
            if expert >= 0:
                m0 = tm * BLOCK_M
                n0 = tn * BLOCK_N
                cols = n0 + tl.arange(0, BLOCK_N)
                col_mask = cols < Hp
                # int64 offsets: expert * H * Hp crosses 2^31 at production shapes.
                expert64 = expert.to(tl.int64)
                stride_e64 = stride_e.to(tl.int64)
                stride_h64 = stride_h.to(tl.int64)
                stride_b64 = stride_b.to(tl.int64)
                b64 = b.to(tl.int64)
                for mi in tl.static_range(BLOCK_M):
                    row = (m0 + mi).to(tl.int64)
                    if row < H:
                        src_row = remote_ptr + expert64 * stride_e64 + row * stride_h64
                        dst_row = prefetch_ptr + b64 * stride_b64 + row * stride_h64
                        vals = tl.load(src_row + cols, mask=col_mask, other=0)
                        tl.store(dst_row + cols, vals, mask=col_mask)

    return prefetch_kernel


def _pipeline_num_warps(default: int) -> int:
    return 8 if pipeline_enabled() else default


def launch_prefetch(
    remote_expert: torch.Tensor,
    prefetch_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    num_sms: int,
) -> None:
    if prefetch_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return
    assert remote_expert.dtype == torch.bfloat16 and remote_expert.is_contiguous()
    assert prefetch_buffers.dtype == torch.bfloat16 and prefetch_buffers.is_contiguous()
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous()

    E, H, Hp = (int(x) for x in remote_expert.shape)
    B, out_h, out_hp = (int(x) for x in prefetch_buffers.shape)
    assert out_h == H and out_hp == Hp
    assert experts_to_copy.numel() == B
    assert H % _TILE == 0 and Hp % _TILE == 0, f"H and H' must be multiples of {_TILE}, got ({H}, {Hp})"
    assert isinstance(num_sms, int) and num_sms > 0

    tiles_m = H // _TILE
    tiles_n = Hp // _TILE
    stride_e = H * Hp
    stride_h = Hp
    stride_b = H * Hp

    kernel = _kernel()
    kernel[(num_sms,)](
        remote_expert,
        prefetch_buffers,
        experts_to_copy,
        stride_e,
        stride_h,
        stride_b,
        H,
        Hp,
        B,
        tiles_m,
        tiles_n,
        NUM_SMS=num_sms,
        BLOCK_M=_TILE,
        BLOCK_N=_TILE,
        num_warps=_pipeline_num_warps(4),
    )
