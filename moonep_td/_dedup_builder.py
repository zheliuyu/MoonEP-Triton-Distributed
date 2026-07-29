"""GPU dedup builder (MoonEP dispatch warps 3..) — Phase 10."""

from __future__ import annotations

import functools

import torch

from moonep_td.constants import KIDX_BITS
from moonep_td.planning import MoonEPCommPlan

_INT32_MAX = 0x7FFFFFFF
_NVS_BITS = 32 - 1 - KIDX_BITS
_NVS_MASK = (1 << _NVS_BITS) - 1


@functools.lru_cache(maxsize=None)
def _kernels():
    import triton.language as tl
    from moonep_td._triton_runtime import triton_dist

    td = triton_dist()

    @td.jit
    def dedup_init_kernel(
        primary_packed_ptr,
        kmask_ptr,
        n_keys,
        NUM_SMS: tl.constexpr,
        INT32_MAX: tl.constexpr,
    ):
        pid = tl.program_id(0)
        for i in range(pid, n_keys, NUM_SMS):
            tl.store(primary_packed_ptr + i, INT32_MAX)
            tl.store(kmask_ptr + i, 0)

    @td.jit
    def dedup_pass1_kernel(
        src_info_ptr,
        primary_packed_ptr,
        kmask_ptr,
        kidx_to_loff_ptr,
        NvS,
        K,
        S,
        NVS_BITS: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        for loff in range(pid, NvS, NUM_SMS):
            info = tl.load(src_info_ptr + loff)
            if info >= 0:
                src_rank = info // NvS
                offv = info - src_rank * NvS
                token = offv // K
                kidx = offv - token * K
                key = src_rank * S + token
                packed = (kidx << NVS_BITS) | loff
                tl.atomic_min(primary_packed_ptr + key, packed)
                bit = 1 << kidx
                tl.atomic_or(kmask_ptr + key, bit)
                tl.store(kidx_to_loff_ptr + key * K + kidx, loff)

    @td.jit
    def dedup_pass2_kernel(
        src_info_ptr,
        primary_packed_ptr,
        kmask_ptr,
        kidx_to_loff_ptr,
        dup_groups_ptr,
        dup_loffs_ptr,
        dup_counts_ptr,
        NvS,
        K,
        S,
        NVS_BITS: tl.constexpr,
        NVS_MASK: tl.constexpr,
        MAX_K: tl.constexpr,
        NUM_SMS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        for loff in range(pid, NvS, NUM_SMS):
            info = tl.load(src_info_ptr + loff)
            if info >= 0:
                src_rank = info // NvS
                offv = info - src_rank * NvS
                token = offv // K
                key = src_rank * S + token
                packed = tl.load(primary_packed_ptr + key)
                mask = tl.load(kmask_ptr + key).to(tl.uint32)
                primary_loff = packed & NVS_MASK
                primary_kidx = packed >> NVS_BITS
                popc = tl.full([], 0, tl.int32)
                for kk in tl.static_range(MAX_K):
                    popc += ((mask >> kk) & 1).to(tl.int32)
                dup_count = popc - 1
                if (loff == primary_loff) & (dup_count > 0):
                    grp_idx = tl.atomic_add(dup_counts_ptr + 0, 1)
                    dup_start = tl.atomic_add(dup_counts_ptr + 1, dup_count)
                    tl.store(dup_groups_ptr + grp_idx * 3 + 0, primary_loff)
                    tl.store(dup_groups_ptr + grp_idx * 3 + 1, dup_start)
                    tl.store(dup_groups_ptr + grp_idx * 3 + 2, dup_count)
                    pos = 0
                    key_row = key * K
                    for kk in tl.static_range(MAX_K):
                        if (kk != primary_kidx) & ((mask >> kk) & 1):
                            dlo = tl.load(kidx_to_loff_ptr + key_row + kk)
                            tl.store(dup_loffs_ptr + dup_start + pos, dlo)
                            pos += 1

    return dedup_init_kernel, dedup_pass1_kernel, dedup_pass2_kernel


def launch_dedup_builder(ctx: dict, plan: MoonEPCommPlan) -> None:
    """Materialize ``dup_*`` plan fields from ``meta_buf`` src_info scratch (GPU)."""
    rank = int(ctx["rank"])
    R, S, K, NvS = int(ctx["R"]), int(ctx["S"]), int(ctx["K"]), int(ctx["NvS"])
    ms = int(ctx["meta_chunk_padded"])
    src_off = int(ctx["SRC_INFO_OFF"])
    meta = ctx["meta_buf"]
    num_sms = max(1, int(ctx.get("num_sms", 1)))

    primary_packed = ctx["primary_packed"]
    kmask = ctx["kmask"]
    kidx_to_loff = ctx["kidx_to_loff"]

    plan.dup_groups.zero_()
    plan.dup_loffs.zero_()
    plan.dup_counts.zero_()
    kidx_to_loff.zero_()

    src_info = meta[rank * ms + src_off : rank * ms + src_off + NvS]
    n_keys = R * S

    init_k, pass1_k, pass2_k = _kernels()
    init_k[(num_sms,)](primary_packed, kmask, n_keys, NUM_SMS=num_sms, INT32_MAX=_INT32_MAX, num_warps=1)
    pass1_k[(num_sms,)](
        src_info,
        primary_packed,
        kmask,
        kidx_to_loff,
        NvS,
        K,
        S,
        NVS_BITS=_NVS_BITS,
        NUM_SMS=num_sms,
        num_warps=4,
    )
    pass2_k[(num_sms,)](
        src_info,
        primary_packed,
        kmask,
        kidx_to_loff,
        plan.dup_groups,
        plan.dup_loffs,
        plan.dup_counts,
        NvS,
        K,
        S,
        NVS_BITS=_NVS_BITS,
        NVS_MASK=_NVS_MASK,
        MAX_K=32,
        NUM_SMS=num_sms,
        num_warps=4,
    )
