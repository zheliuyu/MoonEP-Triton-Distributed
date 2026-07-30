"""Torch reference for the MoonEP planning kernel (mirrors MoonEP tests/planning_reference.py)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ReferenceDedupPlan:
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor


def publish_src_info_to_meta(ctx: dict, dst_positive: torch.Tensor) -> None:
    """Publish per-slot source provenance into ``meta_buf`` (MoonEP planning)."""
    import nvshmem.core

    from moonep_td._common import launch_cross_rank_barrier

    rank = int(ctx["rank"])
    R = int(ctx["R"])
    NvS = int(ctx["NvS"])
    ms = int(ctx["meta_chunk_padded"])
    src_off = int(ctx["SRC_INFO_OFF"])
    meta = ctx["meta_buf"]

    meta[rank * ms + src_off : rank * ms + src_off + NvS].fill_(-1)
    torch.cuda.synchronize()
    launch_cross_rank_barrier(ctx)

    dst_cpu = dst_positive.detach().cpu()
    for i in range(int(dst_cpu.numel())):
        v = int(dst_cpu[i].item())
        if v < 0:
            raise AssertionError("publish_src_info_to_meta expects canonical positive dst")
        dest = v // NvS
        loff = v % NvS
        src_val = rank * NvS + i
        if dest == rank:
            meta[rank * ms + src_off + loff] = src_val
        else:
            peer = nvshmem.core.get_peer_tensor(meta, dest)
            peer[dest * ms + src_off + loff] = src_val

    torch.cuda.synchronize()
    launch_cross_rank_barrier(ctx)


def _gather_dst_by_rank(dst: torch.Tensor, rank: int, R: int, group) -> torch.Tensor:
    if R > 1:
        gathered = [torch.zeros_like(dst) for _ in range(R)]
        dist.all_gather(gathered, dst.contiguous(), group=group)
        return torch.stack(gathered).cpu()
    return dst.unsqueeze(0).cpu()


def encode_dst_duplicates(
    dst: torch.Tensor,
    *,
    rank: int,
    R: int,
    S: int,
    K: int,
    NvS: int,
    group,
) -> torch.Tensor:
    """Canonicalize duplicate top-k entries to ``-raw_dst - 1`` on this rank."""
    dst = dst.clone()
    dst_by_rank = _gather_dst_by_rank(dst, rank, R, group)
    for src_rank in range(R):
        for s_tok in range(S):
            base = s_tok * K
            dst_vals = dst_by_rank[src_rank, base : base + K]
            dests = torch.div(dst_vals, NvS, rounding_mode="floor")
            loffs = dst_vals % NvS
            groups: dict[int, list[int]] = {}
            indices: dict[int, list[int]] = {}
            for k in range(K):
                dest = int(dests[k].item())
                groups.setdefault(dest, []).append(int(loffs[k].item()))
                indices.setdefault(dest, []).append(k)
            for dest, group_loffs in groups.items():
                for i in range(1, len(group_loffs)):
                    if src_rank == rank:
                        idx = base + indices[dest][i]
                        dst[idx] = -dst[idx] - 1
    return dst


def build_reference_dedup_plan(
    dst_positive: torch.Tensor,
    *,
    rank: int,
    R: int,
    S: int,
    K: int,
    NvS: int,
    group,
    dev: torch.device,
) -> ReferenceDedupPlan:
    """Deterministic reference dup structures for semantic dispatch tests."""
    dst_by_rank = _gather_dst_by_rank(dst_positive, rank, R, group)
    dup_groups_by_rank = torch.zeros((R, NvS, 3), dtype=torch.int32)
    dup_loffs_by_rank = torch.zeros((R, NvS), dtype=torch.int32)
    dup_counts_by_rank = torch.zeros((R, 2), dtype=torch.int32)

    for src_rank in range(R):
        for s_tok in range(S):
            base = s_tok * K
            dst_vals = dst_by_rank[src_rank, base : base + K]
            dests = torch.div(dst_vals, NvS, rounding_mode="floor")
            loffs = dst_vals % NvS
            groups: dict[int, list[int]] = {}
            for k in range(K):
                dest = int(dests[k].item())
                groups.setdefault(dest, []).append(int(loffs[k].item()))
            for dest, group_loffs in groups.items():
                dup_count = len(group_loffs) - 1
                if dup_count <= 0:
                    continue
                primary_loff = group_loffs[0]
                group_idx = int(dup_counts_by_rank[dest, 0].item())
                dup_start = int(dup_counts_by_rank[dest, 1].item())
                dup_groups_by_rank[dest, group_idx, 0] = primary_loff
                dup_groups_by_rank[dest, group_idx, 1] = dup_start
                dup_groups_by_rank[dest, group_idx, 2] = dup_count
                dup_loffs_by_rank[dest, dup_start : dup_start + dup_count] = (
                    dup_loffs_by_rank.new_tensor(group_loffs[1:])
                )
                dup_counts_by_rank[dest, 0] += 1
                dup_counts_by_rank[dest, 1] += dup_count

    return ReferenceDedupPlan(
        dup_groups=dup_groups_by_rank[rank].to(dev),
        dup_loffs=dup_loffs_by_rank[rank].to(dev),
        dup_counts=dup_counts_by_rank[rank].to(dev),
    )


def launch_planning_torch_reference(
    ctx,
    topk_experts,
    tokens_per_expert,
    *,
    return_alloc=False,
    materialize_dedup: bool = True,
):
    rank = ctx["rank"]
    R = ctx["R"]
    E = ctx["E"]
    B = int(ctx["B"])
    S = ctx["S"]
    K = ctx["K"]
    N = S * K
    epn = E // R
    CAP = ctx["NvS_capacity"]
    NvS = ctx.get("NvS", CAP)
    token_padding = ctx.get("token_padding", 1)
    group = ctx.get("group")

    flat_topk_experts = topk_experts[rank].reshape(-1) if topk_experts.dim() == 3 else topk_experts.reshape(-1)
    if tokens_per_expert.dim() == 2:
        tpe = tokens_per_expert
    else:
        tpe = tokens_per_expert.reshape(1, E)
        if R > 1:
            gathered = [torch.zeros_like(tokens_per_expert) for _ in range(R)]
            dist.all_gather(gathered, tokens_per_expert, group=group)
            tpe = torch.stack(gathered)
    flat_topk_experts = flat_topk_experts.cpu()
    tpe = tpe.cpu()

    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[R - 1]

    group_tokens = torch.zeros(R, dtype=expert_count.dtype)
    for h in range(R):
        group_tokens[h] = expert_count[h * epn : (h + 1) * epn].sum()

    balance = group_tokens - CAP
    alloc = torch.zeros(E, R, dtype=torch.int32)
    for e in range(E):
        alloc[e, e // epn] = expert_count[e]

    z = torch.zeros(R, R, dtype=torch.int32)
    while True:
        h = balance.argmax()
        u = balance.argmin()
        if balance[h] <= 0:
            break
        move = -balance[u]
        z[h, u] = move
        balance[h] -= move
        balance[u] = 0

    for h in range(R):
        expert_start = h * epn
        remaining = expert_count[expert_start : expert_start + epn].clone()
        quotas = z[h].clone()
        while True:
            d = quotas.argmax()
            quota = quotas[d]
            if quota <= 0:
                break
            local_e = remaining.argmax()
            e = expert_start + local_e
            rem = remaining[local_e]
            take = torch.minimum(rem, quota)
            alloc[e, d] += take
            alloc[e, h] -= take
            remaining[local_e] -= take
            quotas[d] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("torch planning reference: per-expert token conservation failed")
    if bool((alloc.sum(dim=0) > CAP).any().item()):
        raise AssertionError("torch planning reference: rank capacity exceeded")

    if return_alloc:
        return alloc.t().contiguous()

    alloc_cumsum = alloc.cumsum(dim=1)
    expert_off = torch.zeros(R, E, dtype=torch.int32)
    cu_seqlens = torch.zeros(R, E + B, dtype=torch.int32)
    experts_to_copy = torch.full((R, B), -1, dtype=torch.int32)
    remote_stats_all = torch.zeros(R, 2, dtype=torch.int32)
    zero_fill_by_rank = torch.zeros(R, E + B, 2, dtype=torch.int32)

    for d in range(R):
        local_start = d * epn
        remote_experts = [
            e for e in range(E)
            if alloc[e, d].item() > 0 and not (local_start <= e < local_start + epn)
        ]
        remote_experts.sort(key=lambda e: (alloc[e, d].item(), e), reverse=True)
        remote_stats_all[d, 0] = len(remote_experts)
        for b, e in enumerate(remote_experts[:B]):
            experts_to_copy[d, b] = e
            remote_stats_all[e // epn, 1] += 1

        experts_to_prefetch = set(remote_experts[:B])
        start = 0
        for g in range(E + B):
            cnt = 0
            expert_id = -1
            if g < E:
                if g not in experts_to_prefetch:
                    cnt = alloc[g, d].item()
                    expert_id = g
            else:
                b = g - E
                if experts_to_copy[d, b].item() >= 0:
                    expert_id = experts_to_copy[d, b].item()
                    cnt = alloc[expert_id, d].item()
            end = start + cnt
            if cnt > 0:
                padded = ((cnt + token_padding - 1) // token_padding) * token_padding
            else:
                padded = 0
            aligned_end = start + padded
            cu_seqlens[d, g] = aligned_end
            if cnt > 0:
                expert_off[d, expert_id] = start
                n_pad = padded - cnt
                if n_pad > 0:
                    zero_fill_by_rank[d, g, 0] = end
                    zero_fill_by_rank[d, g, 1] = n_pad
            start = aligned_end
        if cu_seqlens[d].max().item() > NvS:
            raise AssertionError("torch planning reference: padded layout exceeds NvS")

    local_cnt = torch.zeros(N, dtype=torch.int32)
    counter = torch.zeros(E, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        local_cnt[i] = counter[e]
        counter[e] += 1

    global_rank = torch.zeros(N, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        prev = 0 if rank == 0 else tpe_cumsum[rank - 1, e].item()
        global_rank[i] = prev + local_cnt[i]

    dst = torch.zeros(N, dtype=torch.int32)
    for i in range(N):
        e = flat_topk_experts[i].item()
        g = global_rank[i].item()
        dest = int(torch.searchsorted(alloc_cumsum[e], g, right=True).item())
        if dest >= R:
            raise AssertionError("torch planning reference: no destination rank found")
        prev_alloc = 0 if dest == 0 else alloc_cumsum[e, dest - 1].item()
        seg_pos = g - prev_alloc
        base_off = expert_off[dest, e].item()
        dst[i] = dest * NvS + base_off + seg_pos

    dev = tokens_per_expert.device
    if not materialize_dedup:
        return (
            dst.to(dev),
            cu_seqlens[rank].to(dev),
            experts_to_copy.to(dev),
            remote_stats_all[rank].to(dev),
            zero_fill_by_rank[rank].to(dev),
            ReferenceDedupPlan(
                dup_groups=torch.zeros(NvS, 3, dtype=torch.int32, device=dev),
                dup_loffs=torch.zeros(NvS, dtype=torch.int32, device=dev),
                dup_counts=torch.zeros(2, dtype=torch.int32, device=dev),
            ),
        )

    dup_groups_by_rank = torch.zeros((R, NvS, 3), dtype=torch.int32)
    dup_loffs_by_rank = torch.zeros((R, NvS), dtype=torch.int32)
    dup_counts_by_rank = torch.zeros((R, 2), dtype=torch.int32)

    dst_by_rank = dst.unsqueeze(dim=0).contiguous()
    if R > 1:
        gathered = [torch.zeros_like(dst, device=dev) for _ in range(R)]
        dist.all_gather(gathered, dst.to(device=dev), group=group)
        dst_by_rank = torch.stack(gathered).cpu()

    for src_rank in range(R):
        for s_tok in range(S):
            base = s_tok * K
            dst_vals = dst_by_rank[src_rank, base : base + K]
            dests = torch.div(dst_vals, NvS, rounding_mode="floor")
            loffs = dst_vals % NvS

            groups: dict[int, list[int]] = {}
            indices: dict[int, list[int]] = {}
            for k in range(K):
                dest = int(dests[k].item())
                groups.setdefault(dest, []).append(int(loffs[k].item()))
                indices.setdefault(dest, []).append(k)

            for dest, group_loffs in groups.items():
                dup_count = len(group_loffs) - 1
                primary_loff = group_loffs[0]
                if dup_count > 0:
                    group_idx = int(dup_counts_by_rank[dest, 0].item())
                    dup_start = int(dup_counts_by_rank[dest, 1].item())
                    dup_groups_by_rank[dest, group_idx, 0] = primary_loff
                    dup_groups_by_rank[dest, group_idx, 1] = dup_start
                    dup_groups_by_rank[dest, group_idx, 2] = dup_count
                    dup_loffs_by_rank[dest, dup_start : dup_start + dup_count] = (
                        dup_loffs_by_rank.new_tensor(group_loffs[1:])
                    )
                    dup_counts_by_rank[dest, 0] += 1
                    dup_counts_by_rank[dest, 1] += dup_count

                for i in range(1, len(group_loffs)):
                    if src_rank == rank:
                        idx = base + indices[dest][i]
                        dst[idx] = -dst[idx] - 1

    ref_dedup = ReferenceDedupPlan(
        dup_groups=dup_groups_by_rank[rank].to(dev),
        dup_loffs=dup_loffs_by_rank[rank].to(dev),
        dup_counts=dup_counts_by_rank[rank].to(dev),
    )
    return (
        dst.to(dev),
        cu_seqlens[rank].to(dev),
        experts_to_copy.to(dev),
        remote_stats_all[rank].to(dev),
        zero_fill_by_rank[rank].to(dev),
        ref_dedup,
    )
