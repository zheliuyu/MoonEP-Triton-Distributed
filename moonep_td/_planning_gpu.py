"""GPU planning kernels and unified launch (Phase 11)."""

from __future__ import annotations

import functools
import os

import torch

from moonep_td._common import launch_cross_rank_barrier
from moonep_td.planning import MoonEPCommPlan


def use_gpu_planning() -> bool:
    raw = os.environ.get("MOONEP_TD_PLANNING_TRITON", "1")
    return raw not in ("0", "false", "False", "")


def _peer_meta_slice(meta: torch.Tensor, peer_rank: int, local_rank: int) -> torch.Tensor:
    if peer_rank == local_rank:
        return meta
    import nvshmem.core as nvs
    return nvs.get_peer_tensor(meta, peer_rank)


def _gather_rows_from_meta(
    meta: torch.Tensor,
    local_rank: int,
    ms: int,
    off: int,
    row_len: int,
    n_rows: int,
    *,
    row_stride: int | None = None,
) -> torch.Tensor:
    """Read ``n_rows`` rows published by each rank into symmetric meta."""
    if row_stride is None:
        row_stride = row_len
    out = torch.empty(n_rows, row_len, dtype=torch.int32, device=meta.device)
    for r in range(n_rows):
        peer = _peer_meta_slice(meta, r, local_rank)
        base = r * ms + off + r * row_stride
        out[r].copy_(peer[base : base + row_len])
    return out


@functools.lru_cache(maxsize=None)
def _publish_tpe_kernel():
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def publish_tpe_kernel(meta_ptr, tpe_ptr, rank, ms, tpe_off, E):
        e = tl.program_id(0)
        if e >= E:
            return
        val = tl.load(tpe_ptr + e)
        tl.store(meta_ptr + rank * ms + tpe_off + rank * E + e, val)

    return publish_tpe_kernel


@functools.lru_cache(maxsize=None)
def _publish_dst_scratch_kernel():
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def publish_dst_scratch_kernel(meta_ptr, dst_ptr, rank, ms, scratch_off, N):
        slot = tl.program_id(0)
        if slot >= N:
            return
        val = tl.load(dst_ptr + slot)
        tl.store(meta_ptr + rank * ms + scratch_off + slot, val)

    return publish_dst_scratch_kernel


def gather_tpe_gpu(ctx: dict, tokens_per_expert: torch.Tensor, R: int, E: int) -> torch.Tensor:
    """Phase A: publish local TPE to symmetric meta and gather ``[R, E]`` (no all_gather)."""
    if tokens_per_expert.dim() == 2:
        return tokens_per_expert.contiguous()
    rank = int(ctx["rank"])
    ms = int(ctx["meta_chunk_padded"])
    tpe_off = int(ctx["TPE_OFF"])
    meta = ctx["meta_buf"]
    tpe_local = tokens_per_expert.reshape(-1)[:E].contiguous().to(torch.int32)
    if R == 1:
        return tpe_local.view(1, E)

    kernel = _publish_tpe_kernel()
    kernel[(E,)](meta, tpe_local, rank, ms, tpe_off, E, num_warps=1)
    launch_cross_rank_barrier(ctx)
    return _gather_rows_from_meta(meta, rank, ms, tpe_off, E, R, row_stride=E)


def _per_expert_local_rank(experts: torch.Tensor) -> torch.Tensor:
    n = int(experts.numel())
    device = experts.device
    if n == 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    perm = experts.argsort(stable=True)
    sorted_e = experts[perm]
    change = torch.ones(n, dtype=torch.int32, device=device)
    change[1:] = (sorted_e[1:] != sorted_e[:-1]).to(torch.int32)
    seg_starts = torch.nonzero(change, as_tuple=False).squeeze(1)
    seg_ids = torch.cumsum(change, dim=0) - 1
    local_sorted = torch.arange(n, device=device, dtype=torch.int32) - seg_starts[seg_ids].to(torch.int32)
    local = torch.empty(n, dtype=torch.int32, device=device)
    local[perm] = local_sorted
    return local


@functools.lru_cache(maxsize=None)
def _publish_src_info_kernel():
    import triton.language as tl
    import triton_dist.language as dl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def publish_src_info_kernel(
        meta_ptr,
        dst_ptr,
        rank,
        NvS,
        ms,
        src_off,
        N,
    ):
        slot = tl.program_id(0)
        if slot >= N:
            return
        v = tl.load(dst_ptr + slot)
        dest = v // NvS
        loff = v - dest * NvS
        src_val = rank * NvS + slot
        if dest == rank:
            tl.store(meta_ptr + rank * ms + src_off + loff, src_val)
        else:
            remote = dl.symm_at(meta_ptr, dest)
            tl.store(remote + dest * ms + src_off + loff, src_val)

    return publish_src_info_kernel


@functools.lru_cache(maxsize=None)
def _compute_dst_kernel(R: int):
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def compute_dst_kernel(
        topk_ptr,
        local_cnt_ptr,
        tpe_prev_ptr,
        alloc_cumsum_ptr,
        expert_off_ptr,
        dst_ptr,
        rank,
        nvs,
        E,
        MAX_R: tl.constexpr,
    ):
        slot = tl.program_id(0)
        expert = tl.load(topk_ptr + slot).to(tl.int64)
        local_cnt = tl.load(local_cnt_ptr + slot)
        if rank == 0:
            prev = tl.full([], 0, tl.int32)
        else:
            prev = tl.load(tpe_prev_ptr + expert)
        global_rank = prev + local_cnt

        dest = tl.full([], 0, tl.int32)
        for d in tl.static_range(MAX_R):
            cum = tl.load(alloc_cumsum_ptr + expert * MAX_R + d)
            if global_rank >= cum:
                dest = d + 1

        prev_alloc = tl.full([], 0, tl.int32)
        if dest > 0:
            prev_alloc = tl.load(alloc_cumsum_ptr + expert * MAX_R + (dest - 1))
        seg_pos = global_rank - prev_alloc
        base_off = tl.load(expert_off_ptr + dest * E + expert)
        out = dest * nvs + base_off + seg_pos
        tl.store(dst_ptr + slot, out)

    return compute_dst_kernel


def publish_src_info_to_meta_gpu(ctx: dict, dst_positive: torch.Tensor) -> None:
    """GPU publish of per-slot src_info into ``meta_buf`` (no CPU loop)."""
    rank = int(ctx["rank"])
    NvS = int(ctx["NvS"])
    ms = int(ctx["meta_chunk_padded"])
    src_off = int(ctx["SRC_INFO_OFF"])
    meta = ctx["meta_buf"]
    N = int(dst_positive.numel())

    meta[rank * ms + src_off : rank * ms + src_off + NvS].fill_(-1)
    torch.cuda.synchronize()
    launch_cross_rank_barrier(ctx)

    if int((dst_positive < 0).any().item()):
        raise AssertionError("publish_src_info_to_meta expects canonical positive dst")

    kernel = _publish_src_info_kernel()
    kernel[(N,)](meta, dst_positive, rank, NvS, ms, src_off, N, num_warps=1)

    torch.cuda.synchronize()
    launch_cross_rank_barrier(ctx)


def compute_dst_positive_gpu(
    topk_flat: torch.Tensor,
    local_cnt: torch.Tensor,
    tpe_cumsum: torch.Tensor,
    alloc_cumsum: torch.Tensor,
    expert_off: torch.Tensor,
    rank: int,
    nvs: int,
) -> torch.Tensor:
    """Token-parallel dst computation via Triton."""
    R, E = int(alloc_cumsum.shape[1]), int(alloc_cumsum.shape[0])
    n = int(topk_flat.numel())
    dev = topk_flat.device
    dst = torch.empty(n, dtype=torch.int32, device=dev)

    if rank == 0:
        tpe_prev = torch.zeros(E, dtype=torch.int32, device=dev)
    else:
        tpe_prev = tpe_cumsum[rank - 1].to(torch.int32)

    alloc_pad = torch.zeros(E, R, dtype=torch.int32, device=dev)
    alloc_pad[:, :R] = alloc_cumsum.to(torch.int32)

    kernel = _compute_dst_kernel(R)
    kernel[(n,)](
        topk_flat,
        local_cnt,
        tpe_prev,
        alloc_pad,
        expert_off.to(torch.int32),
        dst,
        rank,
        nvs,
        E,
        MAX_R=R,
        num_warps=1,
    )
    return dst


def encode_dst_duplicates_gpu(
    dst: torch.Tensor,
    ctx: dict,
    *,
    rank: int,
    R: int,
    S: int,
    K: int,
    NvS: int,
) -> torch.Tensor:
    """GPU duplicate encoding via symmetric meta gather (no all_gather)."""
    dst = dst.clone()
    N = int(dst.numel())
    ms = int(ctx["meta_chunk_padded"])
    scratch_off = int(ctx["TOPK0_OFF"])
    meta = ctx["meta_buf"]

    if R > 1:
        kernel = _publish_dst_scratch_kernel()
        kernel[(N,)](meta, dst, rank, ms, scratch_off, N, num_warps=1)
        launch_cross_rank_barrier(ctx)
        dst_by_rank = _gather_rows_from_meta(meta, rank, ms, scratch_off, N, R, row_stride=0)
    else:
        dst_by_rank = dst.unsqueeze(0)

    dev = dst.device
    k_idx = torch.arange(K, device=dev, dtype=torch.int64)

    for src_rank in range(R):
        vals = dst_by_rank[src_rank].view(S, K)
        dests = vals // NvS
        dest_eq = dests.unsqueeze(2).eq(dests.unsqueeze(1))
        k_i = k_idx.view(1, K, 1).expand(S, K, K)
        k_j = k_idx.view(1, 1, K).expand(S, K, K)
        is_dup = (dest_eq & (k_i < k_j)).any(dim=1)
        if src_rank == rank and bool(is_dup.any().item()):
            flat_idx = torch.arange(S, device=dev, dtype=torch.int64).unsqueeze(1) * K + k_idx.view(1, K)
            dup_flat = flat_idx[is_dup]
            dst[dup_flat] = -dst[dup_flat] - 1
    return dst


@functools.lru_cache(maxsize=None)
def _balance_distribute_kernel(R: int):
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def balance_distribute_kernel(
        expert_count_ptr,
        z_ptr,
        alloc_ptr,
        remaining_ptr,
        quotas_ptr,
        epn,
        n_ranks,
        E,
        MAX_R: tl.constexpr,
    ):
        h = tl.program_id(0)
        if h >= n_ranks:
            return

        expert_start = h * epn
        row_rem = remaining_ptr + h * epn
        row_q = quotas_ptr + h * MAX_R

        for le in range(epn):
            e = expert_start + le
            cnt = tl.load(expert_count_ptr + e)
            tl.store(row_rem + le, cnt)
            tl.store(alloc_ptr + e * MAX_R + h, cnt)

        for d in range(n_ranks):
            q = tl.load(z_ptr + h * MAX_R + d)
            tl.store(row_q + d, q)

        max_iter = epn * n_ranks
        for _ in range(max_iter):
            best_d = 0
            best_q = tl.load(row_q + 0)
            for d in range(1, n_ranks):
                q = tl.load(row_q + d)
                if q > best_q:
                    best_q = q
                    best_d = d

            if best_q > 0:
                best_le = 0
                best_rem = tl.load(row_rem + 0)
                for le in range(1, epn):
                    rem = tl.load(row_rem + le)
                    if rem > best_rem:
                        best_rem = rem
                        best_le = le

                take = tl.minimum(best_rem, best_q)
                e = expert_start + best_le
                dst_val = tl.load(alloc_ptr + e * MAX_R + best_d) + take
                home_val = tl.load(alloc_ptr + e * MAX_R + h) - take
                tl.store(alloc_ptr + e * MAX_R + best_d, dst_val)
                tl.store(alloc_ptr + e * MAX_R + h, home_val)
                tl.store(row_rem + best_le, best_rem - take)
                tl.store(row_q + best_d, best_q - take)

    return balance_distribute_kernel


def _compute_transfer_quotas(group_tokens: torch.Tensor, cap: int, R: int) -> torch.Tensor:
    """Greedy rank-balance transfer matrix ``z[h, u]`` (at most R pairing steps)."""
    device = group_tokens.device
    balance = group_tokens - cap
    z = torch.zeros(R, R, dtype=torch.int32, device=device)
    for _ in range(R):
        h = int(balance.argmax().item())
        u = int(balance.argmin().item())
        if balance[h] <= 0:
            break
        move = -balance[u]
        z[h, u] = move.to(torch.int32)
        balance[h] -= move
        balance[u] = 0
    return z


def balance_alloc_gpu(
    tpe: torch.Tensor,
    R: int,
    E: int,
    epn: int,
    cap: int,
) -> torch.Tensor:
    """Phase B token allocation ``alloc[e, d]`` via Triton distribute kernel."""
    device = tpe.device
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1].to(torch.int32)

    group_tokens = expert_count.view(R, epn).sum(dim=1)
    z = _compute_transfer_quotas(group_tokens, cap, R)

    alloc = torch.zeros(E, R, dtype=torch.int32, device=device)
    remaining = torch.zeros(R, epn, dtype=torch.int32, device=device)
    quotas = torch.zeros(R, R, dtype=torch.int32, device=device)

    kernel = _balance_distribute_kernel(R)
    kernel[(R,)](
        expert_count,
        z,
        alloc,
        remaining,
        quotas,
        epn,
        R,
        E,
        MAX_R=R,
        num_warps=1,
    )

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("planning_gpu: per-expert token conservation failed")
    if bool((alloc.sum(dim=0) > cap).any().item()):
        raise AssertionError("planning_gpu: rank capacity exceeded")
    return alloc


def build_per_rank_layout_gpu(
    alloc: torch.Tensor,
    R: int,
    E: int,
    B: int,
    epn: int,
    nvs: int,
    token_padding: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized Phase C layout (no per-group Python loop)."""
    device = alloc.device
    expert_ids = torch.arange(E, device=device, dtype=torch.int32)
    experts_to_copy = torch.full((R, B), -1, dtype=torch.int32, device=device)
    remote_stats_all = torch.zeros(R, 2, dtype=torch.int32, device=device)
    expert_off = torch.zeros(R, E, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(E + B, dtype=torch.int32, device=device)
    zero_fill = torch.zeros(E + B, 2, dtype=torch.int32, device=device)
    g = E + B
    z0 = torch.zeros((), dtype=torch.int32, device=device)

    for d in range(R):
        local_start = d * epn
        cnt_d = alloc[:, d]
        is_local = (expert_ids >= local_start) & (expert_ids < local_start + epn)
        remote_mask = (cnt_d > 0) & (~is_local)
        n_remote = int(remote_mask.sum().item())
        remote_stats_all[d, 0] = n_remote

        if n_remote > 0:
            sort_key = torch.where(
                remote_mask,
                cnt_d.to(torch.int64) * E + expert_ids.to(torch.int64),
                torch.full((E,), -1, device=device, dtype=torch.int64),
            )
            order = sort_key.argsort(descending=True)
            pick = order[: min(B, n_remote)]
            experts_to_copy[d, : pick.numel()] = expert_ids[pick]
            owners = (expert_ids[pick] // epn).to(torch.int64)
            remote_stats_all[:, 1] += torch.bincount(owners, minlength=R).to(torch.int32)

        prefetch_mask = torch.zeros(E, dtype=torch.bool, device=device)
        n_prefetch = min(B, n_remote)
        if n_prefetch > 0:
            prefetch_mask[experts_to_copy[d, :n_prefetch]] = True

        counts = torch.zeros(g, dtype=torch.int32, device=device)
        experts_g = torch.full((g,), -1, dtype=torch.int32, device=device)
        non_pf = ~prefetch_mask
        counts[:E] = torch.where(non_pf, cnt_d, z0.expand(E))
        experts_g[:E] = torch.where(non_pf, expert_ids, experts_g[:E])

        eid = experts_to_copy[d]
        valid_b = eid >= 0
        pref_counts = torch.zeros(B, dtype=torch.int32, device=device)
        if bool(valid_b.any().item()):
            pref_counts[valid_b] = cnt_d[eid[valid_b].to(torch.int64)]
        counts[E:E + B] = pref_counts
        experts_g[E:E + B] = eid

        padded = torch.zeros(g, dtype=torch.int32, device=device)
        pos = counts > 0
        if bool(pos.any().item()):
            c = counts[pos]
            padded[pos] = ((c + token_padding - 1) // token_padding) * token_padding

        cu = padded.cumsum(0, dtype=torch.int32)
        starts = cu - padded
        active = counts > 0
        if bool(active.any().item()):
            expert_off[d].scatter_(
                0, experts_g[active].to(torch.int64), starts[active],
            )

        if d == rank:
            cu_seqlens.copy_(cu)
            n_pad = padded - counts
            zf = n_pad > 0
            if bool(zf.any().item()):
                zero_fill[zf, 0] = starts[zf] + counts[zf]
                zero_fill[zf, 1] = n_pad[zf]
            if int(cu.max().item()) > nvs:
                raise AssertionError("planning_gpu: padded layout exceeds NvS")

    return cu_seqlens, experts_to_copy, remote_stats_all[rank], zero_fill, expert_off


def launch_planning_gpu(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
) -> None:
    """Unified GPU planning launch: Phase A–D via Triton + symmetric meta."""
    rank = int(ctx["rank"])
    R = int(ctx["R"])
    E = int(ctx["E"])
    B = int(ctx["B"])
    S = int(ctx["S"])
    K = int(ctx["K"])
    NvS = int(ctx["NvS"])
    epn = E // R
    cap = int(ctx["NvS_capacity"])
    token_padding = int(ctx.get("token_padding", 1))
    dev = topk_experts_flat.device

    topk = topk_experts_flat.reshape(-1).contiguous()
    tpe = gather_tpe_gpu(ctx, tokens_per_expert.contiguous(), R, E).to(dev)
    alloc = balance_alloc_gpu(tpe, R, E, epn, cap)
    ctx["alloc"].copy_(alloc.t().reshape(-1))

    cu, etc, rs, zfr, expert_off = build_per_rank_layout_gpu(
        alloc, R, E, B, epn, NvS, token_padding, rank,
    )

    tpe_cumsum = tpe.cumsum(dim=0)
    alloc_cumsum = alloc.cumsum(dim=1)
    local_cnt = _per_expert_local_rank(topk.to(torch.int32))

    dst_pos = compute_dst_positive_gpu(
        topk, local_cnt, tpe_cumsum, alloc_cumsum, expert_off, rank, NvS,
    )

    publish_src_info_to_meta_gpu(ctx, dst_pos)
    dst = encode_dst_duplicates_gpu(
        dst_pos, ctx, rank=rank, R=R, S=S, K=K, NvS=NvS,
    )

    plan.dst.copy_(dst)
    cu_seqlens.copy_(cu)
    plan.experts_to_copy.copy_(etc)
    plan.remote_stats.copy_(rs)
    plan.zero_fill_ranges.copy_(zfr)
    plan.dup_groups.zero_()
    plan.dup_loffs.zero_()
    plan.dup_counts.zero_()
