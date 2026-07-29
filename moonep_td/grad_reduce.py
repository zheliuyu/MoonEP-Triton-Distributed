"""Remote expert grad reduce — Triton tile accumulator (Phase 9)."""

from __future__ import annotations

import functools

import torch

from moonep_td._common import launch_cross_rank_barrier

_TILE = 128


def _build_grad_reduce_work(
    experts_to_copy: torch.Tensor,
    rank: int,
    E: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """GPU prescan: group remote slots by local expert in rb-ascending order."""
    dev = experts_to_copy.device
    R, B = int(experts_to_copy.shape[0]), int(experts_to_copy.shape[1])
    epn = E // R
    owner_start = rank * epn
    owner_end = owner_start + epn

    experts_flat = experts_to_copy.reshape(-1)
    rb = torch.arange(R * B, device=dev, dtype=torch.int64)

    valid = (
        (experts_flat >= 0)
        & (experts_flat >= owner_start)
        & (experts_flat < owner_end)
    )
    if not bool(valid.any().item()):
        empty_i32 = torch.empty(0, dtype=torch.int32, device=dev)
        offsets = torch.zeros(epn + 1, dtype=torch.int32, device=dev)
        clear_b = torch.nonzero(experts_to_copy[rank] >= 0, as_tuple=False).view(-1).to(torch.int32)
        return empty_i32, offsets, empty_i32, clear_b, empty_i32, 0, int(clear_b.numel())

    local_expert = (experts_flat[valid] - owner_start).to(torch.int32)
    slot_rb = rb[valid].to(torch.int32)

    sort_key = local_expert.to(torch.int64) * (R * B + 1) + slot_rb.to(torch.int64)
    order = sort_key.argsort(stable=True)
    local_expert = local_expert[order]
    slot_rb = slot_rb[order]

    counts = torch.bincount(local_expert.to(torch.int64), minlength=epn).to(torch.int32)
    offsets = torch.zeros(epn + 1, dtype=torch.int32, device=dev)
    offsets[1:] = counts.cumsum(0)

    active = torch.nonzero(counts > 0, as_tuple=False).view(-1).to(torch.int32)
    clear_b = torch.nonzero(experts_to_copy[rank] >= 0, as_tuple=False).view(-1).to(torch.int32)
    return (
        active,
        offsets,
        slot_rb,
        clear_b,
        counts,
        int(active.numel()),
        int(clear_b.numel()),
    )


@functools.lru_cache(maxsize=None)
def _kernels():
    import triton.language as tl
    import triton_dist.language as dl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def grad_reduce_accum_kernel(
        grad_ptr,
        reduce_ptr,
        active_ptr,
        offsets_ptr,
        slot_rb_ptr,
        stride_e,
        stride_h,
        stride_slot,
        H,
        Hp,
        B,
        owner_start,
        num_active,
        tiles_m,
        tiles_n,
        NUM_SMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_per_expert = tiles_m * tiles_n
        total = num_active * tiles_per_expert
        cols = tl.arange(0, BLOCK_N)

        for work in range(pid, total, NUM_SMS):
            ae_idx = work // tiles_per_expert
            rem = work % tiles_per_expert
            tm = rem // tiles_n
            tn = rem % tiles_n
            m0 = tm * BLOCK_M
            n0 = tn * BLOCK_N
            col_mask = (n0 + cols) < Hp

            local_expert = tl.load(active_ptr + ae_idx).to(tl.int32)
            beg = tl.load(offsets_ptr + local_expert)
            end = tl.load(offsets_ptr + local_expert + 1)
            nslot = end - beg

            expert_id = owner_start + local_expert
            expert64 = expert_id.to(tl.int64)

            for mi in tl.static_range(BLOCK_M):
                row = m0 + mi
                row_ok = row < H
                grad_row = grad_ptr + expert64 * stride_e + row.to(tl.int64) * stride_h
                dst_cols = n0 + cols
                acc = tl.load(grad_row + dst_cols, mask=row_ok & col_mask, other=0.0)

                for s in range(nslot):
                    rb = tl.load(slot_rb_ptr + beg + s).to(tl.int64)
                    src_rank = (rb // B).to(tl.int32)
                    remote = dl.symm_at(reduce_ptr, src_rank)
                    slot_base = rb * stride_slot + row.to(tl.int64) * stride_h
                    src_cols = n0 + cols
                    remote_vals = tl.load(
                        remote + slot_base + src_cols,
                        mask=row_ok & col_mask,
                        other=0.0,
                    )
                    acc = acc + remote_vals

                tl.store(grad_row + dst_cols, acc, mask=row_ok & col_mask)

    @td.jit
    def grad_reduce_clear_kernel(
        reduce_ptr,
        clear_b_ptr,
        rank,
        B,
        stride_slot,
        stride_h,
        H,
        Hp,
        num_clear,
        tiles_m,
        tiles_n,
        NUM_SMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        tiles_per_slot = tiles_m * tiles_n
        total = num_clear * tiles_per_slot
        cols = tl.arange(0, BLOCK_N)

        for work in range(pid, total, NUM_SMS):
            slot_idx = work // tiles_per_slot
            rem = work % tiles_per_slot
            tm = rem // tiles_n
            tn = rem % tiles_n
            m0 = tm * BLOCK_M
            n0 = tn * BLOCK_N
            col_mask = (n0 + cols) < Hp

            b = tl.load(clear_b_ptr + slot_idx).to(tl.int64)
            rb = tl.full([], rank, tl.int64) * tl.full([], B, tl.int64) + b
            slot_base = rb * stride_slot

            for mi in tl.static_range(BLOCK_M):
                row = m0 + mi
                row_ok = row < H
                dst_row = reduce_ptr + slot_base + row.to(tl.int64) * stride_h
                tl.store(
                    dst_row + n0 + cols,
                    tl.zeros([BLOCK_N], dtype=tl.float32),
                    mask=row_ok & col_mask,
                )

    return grad_reduce_accum_kernel, grad_reduce_clear_kernel


def launch_grad_reduce(
    remote_expert_grads: torch.Tensor,
    remote_reduce_buffers: torch.Tensor,
    experts_to_copy: torch.Tensor,
    rank: int,
    num_sms: int,
    meta_buf: torch.Tensor,
    meta_stride: int,
    barrier_off: int,
    grid_sync_bar: torch.Tensor,
) -> None:
    """Launch remote expert grad reduction."""
    if remote_reduce_buffers.numel() == 0 or experts_to_copy.numel() == 0:
        return

    assert remote_expert_grads.dtype == torch.float32 and remote_expert_grads.is_contiguous()
    assert remote_reduce_buffers.dtype == torch.float32 and remote_reduce_buffers.is_contiguous()
    assert experts_to_copy.dtype == torch.int32 and experts_to_copy.is_contiguous()
    assert remote_expert_grads.ndim == 3
    assert remote_reduce_buffers.ndim == 4
    assert experts_to_copy.ndim == 2

    E, H, Hp = (int(x) for x in remote_expert_grads.shape)
    R, B, buf_H, buf_Hp = (int(x) for x in remote_reduce_buffers.shape)
    assert tuple(experts_to_copy.shape) == (R, B)
    assert buf_H == H and buf_Hp == Hp
    assert E % R == 0
    assert 0 <= int(rank) < R
    assert H % _TILE == 0 and Hp % _TILE == 0, (
        f"H and H' must be multiples of {_TILE}, got ({H}, {Hp})"
    )
    assert isinstance(num_sms, int) and num_sms > 0

    ctx = {
        "R": R,
        "rank": rank,
        "meta_buf": meta_buf,
        "meta_chunk_padded": meta_stride,
        "BARRIER_OFF": barrier_off,
        "grid_sync_bar": grid_sync_bar,
        "num_sms": num_sms,
    }
    launch_cross_rank_barrier(ctx)

    active, offsets, slot_rb, clear_b, _counts, num_active, num_clear = _build_grad_reduce_work(
        experts_to_copy, rank, E,
    )

    if num_active > 0:
        accum_kernel, clear_kernel = _kernels()
        tiles_m = H // _TILE
        tiles_n = Hp // _TILE
        stride_e = H * Hp
        stride_h = Hp
        stride_slot = H * Hp
        owner_start = rank * (E // R)

        accum_kernel[(num_sms,)](
            remote_expert_grads,
            remote_reduce_buffers,
            active,
            offsets,
            slot_rb,
            stride_e,
            stride_h,
            stride_slot,
            H,
            Hp,
            B,
            owner_start,
            num_active,
            tiles_m,
            tiles_n,
            NUM_SMS=num_sms,
            BLOCK_M=_TILE,
            BLOCK_N=_TILE,
            num_warps=4,
        )

    launch_cross_rank_barrier(ctx)

    if num_clear > 0:
        _, clear_kernel = _kernels()
        stride_slot = H * Hp
        stride_h = Hp
        tiles_m = H // _TILE
        tiles_n = Hp // _TILE
        clear_kernel[(num_sms,)](
            remote_reduce_buffers,
            clear_b,
            rank,
            B,
            stride_slot,
            stride_h,
            H,
            Hp,
            num_clear,
            tiles_m,
            tiles_n,
            NUM_SMS=num_sms,
            BLOCK_M=_TILE,
            BLOCK_N=_TILE,
            num_warps=4,
        )
