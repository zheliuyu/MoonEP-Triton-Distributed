"""CuTe DSL / Triton planning kernel and MoonEPCommPlan."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moonep_td.constants import KIDX_BITS


@dataclass(frozen=True, slots=True)
class MoonEPCommPlan:
    dst: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    N: int
    R: int
    E: int
    B: int
    NvS: int
    K: int
    dup_groups: torch.Tensor
    dup_loffs: torch.Tensor
    dup_counts: torch.Tensor

    def __post_init__(self) -> None:
        N, R, E, B, NvS = int(self.N), int(self.R), int(self.E), int(self.B), int(self.NvS)
        assert self.dst.dtype == torch.int32 and self.dst.is_contiguous()
        assert self.dst.numel() == N
        assert tuple(self.experts_to_copy.shape) == (R, B)
        assert tuple(self.zero_fill_ranges.shape) == (E + B, 2)
        assert tuple(self.remote_stats.shape) == (2,)
        assert tuple(self.dup_groups.shape) == (NvS, 3)
        assert tuple(self.dup_loffs.shape) == (NvS,)
        assert tuple(self.dup_counts.shape) == (2,)

    def clone(self) -> MoonEPCommPlan:
        return type(self)(
            dst=self.dst.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            dup_groups=self.dup_groups.clone(),
            dup_loffs=self.dup_loffs.clone(),
            dup_counts=self.dup_counts.clone(),
            N=self.N, R=self.R, E=self.E, B=self.B, NvS=self.NvS, K=self.K,
        )


def _round4(n: int) -> int:
    return (n + 3) & ~3


def _check_planning_outputs(ctx: dict, cu_seqlens, plan: MoonEPCommPlan) -> None:
    E = ctx["E"]
    B = ctx.get("B", 0)
    assert plan.N == ctx["S"] * ctx["K"]
    assert plan.R == ctx["R"]
    assert plan.E == E
    assert plan.B == B
    assert plan.NvS == ctx["NvS"]
    assert plan.K == ctx["K"]
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()
    assert tuple(cu_seqlens.shape) == (E + B,)


def _check_dedup_encoding_bounds(ctx: dict) -> None:
    R, S, K, NvS = int(ctx["R"]), int(ctx["S"]), int(ctx["K"]), int(ctx["NvS"])
    NvS_BITS = 32 - 1 - KIDX_BITS
    N = S * K
    int32_max = 2**31 - 1
    assert N <= NvS
    assert R * NvS <= int32_max
    assert R <= 128
    assert K <= (1 << KIDX_BITS) - 1
    assert NvS <= (1 << NvS_BITS) - 1
    assert K <= 32


def allocate_planning_outputs(ctx: dict) -> tuple[MoonEPCommPlan, torch.Tensor]:
    E, B, S, K = ctx["E"], ctx["B"], ctx["S"], ctx["K"]
    N = S * K
    NvS, R = ctx["NvS"], ctx["R"]
    dev = ctx["meta_buf"].device
    dst = torch.empty(_round4(N), dtype=torch.int32, device=dev)[:N]
    cu_seqlens = torch.empty(_round4(E + B), dtype=torch.int32, device=dev)[: E + B]
    experts_to_copy = torch.empty(_round4(R * B), dtype=torch.int32, device=dev)[: R * B].view(R, B)
    zero_fill_ranges = torch.empty(_round4((E + B) * 2), dtype=torch.int32, device=dev)[:(E + B) * 2].view(E + B, 2)
    remote_stats = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]
    dup_groups = torch.empty(_round4(NvS * 3), dtype=torch.int32, device=dev)[: NvS * 3].view(NvS, 3)
    dup_loffs = torch.empty(_round4(NvS), dtype=torch.int32, device=dev)[:NvS]
    dup_counts = torch.empty(_round4(2), dtype=torch.int32, device=dev)[:2]
    plan = MoonEPCommPlan(
        dst=dst, experts_to_copy=experts_to_copy, zero_fill_ranges=zero_fill_ranges,
        remote_stats=remote_stats, dup_groups=dup_groups, dup_loffs=dup_loffs,
        dup_counts=dup_counts, N=N, R=R, E=E, B=B, NvS=NvS, K=K,
    )
    return plan, cu_seqlens


def _launch_planning_kernel(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
) -> None:
    """GPU planning path (default) or host reference fallback."""
    from moonep_td._planning_gpu import launch_planning_gpu, use_gpu_planning

    if use_gpu_planning():
        launch_planning_gpu(ctx, topk_experts_flat, tokens_per_expert, cu_seqlens, plan)
        return

    from tests.planning_reference import (
        encode_dst_duplicates,
        launch_planning_torch_reference,
        publish_src_info_to_meta,
    )

    dst_pos, cu, etc, rs, zfr, _ = launch_planning_torch_reference(
        ctx, topk_experts_flat, tokens_per_expert, materialize_dedup=False,
    )
    publish_src_info_to_meta(ctx, dst_pos)
    dst = encode_dst_duplicates(
        dst_pos,
        rank=int(ctx["rank"]),
        R=int(ctx["R"]),
        S=int(ctx["S"]),
        K=int(ctx["K"]),
        NvS=int(ctx["NvS"]),
        group=ctx.get("group"),
    )
    plan.dst.copy_(dst)
    cu_seqlens.copy_(cu)
    plan.experts_to_copy.copy_(etc)
    plan.remote_stats.copy_(rs)
    plan.zero_fill_ranges.copy_(zfr)
    plan.dup_groups.zero_()
    plan.dup_loffs.zero_()
    plan.dup_counts.zero_()


def launch_planning(
    ctx: dict,
    topk_experts_flat: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    cu_seqlens: torch.Tensor,
    plan: MoonEPCommPlan,
) -> None:
    _check_planning_outputs(ctx, cu_seqlens, plan)
    _check_dedup_encoding_bounds(ctx)
    _launch_planning_kernel(ctx, topk_experts_flat, tokens_per_expert, cu_seqlens, plan)
