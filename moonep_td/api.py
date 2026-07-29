"""
MoonEP-TD top-level API (parallel to MoonEP moonep.api).

``_create_context`` lives here, matching MoonEP where buffer layout and Buffer
orchestration share the same module.
"""

from __future__ import annotations

import logging
import os
import warnings

import torch
import torch.distributed as dist

from moonep_td.buffer import (
    create_nvl_dist_tensor,
    ensure_nvshmem_initialized,
    pad_dim0_for_alignment,
    release_nvl_dist_tensor,
)
from moonep_td.combine import launch_combine
from moonep_td.combine_prologue import launch_combine_prologue
from moonep_td.constants import BARRIER_SLOTS, DEDUP_BUILDER_WARPS
from moonep_td.dispatch import launch_dispatch
from moonep_td.dispatch_epilogue import launch_dispatch_epilogue
from moonep_td.grad_reduce import launch_grad_reduce
from moonep_td.inter_rank_sync import launch_inter_rank_sync
from moonep_td.planning import MoonEPCommPlan, allocate_planning_outputs, launch_planning
from moonep_td.prefetch import launch_prefetch

logger = logging.getLogger(__name__)


def _align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def _num_sms_dedup_from_env(max_sms: int) -> int:
    raw = os.environ.get("MOONEP_NUM_SMS_DEDUP")
    if raw is None or raw == "":
        return max_sms
    value = int(raw)
    if not (1 <= value <= max_sms):
        raise ValueError(f"MOONEP_NUM_SMS_DEDUP must be in [1, {max_sms}], got {value}")
    return value


def _create_context(
    S: int,
    H: int,
    K: int,
    E: int,
    num_ep_ranks: int,
    num_sms: int | None = None,
    token_padding: int = 128,
    B: int | None = None,
    group: dist.ProcessGroup | None = None,
) -> dict:
    """Pre-allocate NVSHMEM symmetric buffers (MoonEP: api._create_context + VMM)."""
    ensure_nvshmem_initialized(group=group)
    if num_sms is None:
        num_sms = 32
    rank = dist.get_rank(group=group)
    R = num_ep_ranks
    assert R == dist.get_world_size(group=group)
    device = torch.cuda.current_device()
    dev = f"cuda:{device}"
    epn = E // R
    assert E % R == 0
    if B is None:
        B = epn
    N = S * K
    NvS_capacity = S * K
    token_padding_extra = (token_padding - 1) * 2 * epn
    NvS = NvS_capacity + token_padding_extra
    num_vblocks = (N + 2048 - 1) // 2048
    NvS_padded = pad_dim0_for_alignment([NvS, H], torch.bfloat16)

    WEIGHTS_OFF = 0
    TPE_OFF = _align_up(NvS, 4)
    PLAN_OFF = _align_up(TPE_OFF + R * E, 4)
    broadcast_elems = 3 * E * R
    planning_out_elems = broadcast_elems + R * (E + B) + 2 * R * (E + B) + B * R + 2 * R
    N4 = _align_up(N, 4)
    TOPK0_OFF = _align_up(PLAN_OFF + planning_out_elems, 4)
    ORDER_OFF = TOPK0_OFF + N4
    ORDER0_OFF = ORDER_OFF + N4
    BARRIER_OFF = ORDER0_OFF + N4
    SRC_INFO_OFF = BARRIER_OFF + BARRIER_SLOTS
    meta_chunk_logical = SRC_INFO_OFF + NvS
    meta_chunk_padded = _align_up(meta_chunk_logical * 4, 128) // 4

    hidden_buf = create_nvl_dist_tensor([R * NvS_padded, H], torch.bfloat16, rank, R, group=group)
    meta_buf = create_nvl_dist_tensor([R * meta_chunk_padded], torch.int32, rank, R, group=group)
    # NVSHMEM heap blocks are reused without zeroing; clear local shards so
    # padding slots and weight scratch do not inherit stale/NaN patterns.
    hidden_buf[rank * NvS_padded:(rank + 1) * NvS_padded].zero_()
    meta_buf[rank * meta_chunk_padded:(rank + 1) * meta_chunk_padded].zero_()
    for r in range(R):
        meta_buf[r * meta_chunk_padded + BARRIER_OFF: r * meta_chunk_padded + BARRIER_OFF + BARRIER_SLOTS].zero_()
    dist.barrier(group=group)

    ctx = {
        "rank": rank, "group": group, "R": R, "E": E, "S": S, "K": K, "H": H, "B": B,
        "N": N, "NvS": NvS, "NvS_capacity": NvS_capacity, "NvS_padded": NvS_padded,
        "num_sms": num_sms, "num_sms_dedup": num_sms, "token_padding": token_padding,
        "device": device, "token_padding_extra": token_padding_extra,
        "hidden_buf": hidden_buf, "meta_buf": meta_buf,
        "meta_chunk_padded": meta_chunk_padded,
        "WEIGHTS_OFF": WEIGHTS_OFF, "TPE_OFF": TPE_OFF, "PLAN_OFF": PLAN_OFF,
        "TOPK0_OFF": TOPK0_OFF, "ORDER_OFF": ORDER_OFF, "ORDER0_OFF": ORDER0_OFF,
        "BARRIER_OFF": BARRIER_OFF, "SRC_INFO_OFF": SRC_INFO_OFF,
        "num_vblocks": num_vblocks,
        "alloc": torch.empty(E * R, dtype=torch.int32, device=dev),
        "group_tokens": torch.empty(R, dtype=torch.int32, device=dev),
        "z": torch.empty(R * R, dtype=torch.int32, device=dev),
        "local_hist": torch.empty(E * num_vblocks, dtype=torch.int32, device=dev),
        "grid_sync_bar": torch.zeros(1, dtype=torch.int32, device=dev),
        "primary_packed": torch.empty(R * S, dtype=torch.int32, device=dev),
        "kmask": torch.empty(R * S, dtype=torch.int32, device=dev),
        "kidx_to_loff": torch.empty(R * S * K, dtype=torch.int32, device=dev),
        "builder_bar": torch.zeros(1, dtype=torch.int32, device=dev),
    }
    ctx["hidden_buf_local"] = hidden_buf[rank * NvS_padded: rank * NvS_padded + NvS]
    ctx["weights_buf_local"] = meta_buf[rank * meta_chunk_padded + WEIGHTS_OFF:
                                         rank * meta_chunk_padded + WEIGHTS_OFF + NvS]
    return ctx


def _launch_full_weight_prefetches(ctx, full_gate_weight, full_up_weight, full_down_weight, experts_to_copy):
    E, num_sms, rank = int(ctx["E"]), int(ctx["num_sms"]), int(ctx["rank"])
    for w in (full_gate_weight, full_up_weight, full_down_weight):
        launch_prefetch(w[:E], w[E:], experts_to_copy[rank], num_sms=num_sms)


def _launch_full_grad_reduces(ctx, experts_to_copy, full_gate_grad, full_up_grad, full_down_grad,
                              gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer):
    E, rank, num_sms = int(ctx["E"]), int(ctx["rank"]), int(ctx["num_sms"])
    for full_grad, reduce_buffer in (
        (full_gate_grad, gate_reduce_buffer),
        (full_up_grad, up_reduce_buffer),
        (full_down_grad, down_reduce_buffer),
    ):
        launch_grad_reduce(
            full_grad[:E], reduce_buffer, experts_to_copy, rank=rank, num_sms=num_sms,
            meta_buf=ctx["meta_buf"], meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]), grid_sync_bar=ctx["grid_sync_bar"],
        )


class Buffer:
    """MoonEP communication buffer (Triton-distributed backend)."""

    def __init__(
        self,
        S: int,
        H: int,
        K: int,
        E: int,
        num_ep_ranks: int,
        num_sms: int | None = None,
        token_padding: int = 128,
        B: int | None = None,
        group: dist.ProcessGroup | None = None,
        comm_stream_priority: int = -1,
        enable_pdl: bool = True,
        explicitly_destroy: bool = False,
    ):
        self.explicitly_destroy = explicitly_destroy
        self.comm_stream_priority = comm_stream_priority
        self.enable_pdl = enable_pdl
        self._comm_stream: torch.cuda.Stream | None = None
        self._destroyed = False
        self._ctx = _create_context(S, H, K, E, num_ep_ranks, num_sms=num_sms,
                                    token_padding=token_padding, B=B, group=group)
        max_sms = torch.cuda.get_device_properties(int(self._ctx["device"])).multi_processor_count
        self._ctx["num_sms_dedup"] = _num_sms_dedup_from_env(max_sms)
        self._comm_stream = torch.cuda.Stream(device=int(self._ctx["device"]), priority=comm_stream_priority)

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    def _require_ctx(self) -> dict:
        assert not self._destroyed, "Buffer has been destroyed"
        return self._ctx

    def destroy(self) -> None:
        if self._destroyed:
            return
        ctx = self._ctx
        if ctx is None:
            self._destroyed = True
            return
        if self._comm_stream is not None:
            self._comm_stream.synchronize()
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier(group=ctx.get("group"))
        for key in ("hidden_buf", "meta_buf"):
            t = ctx.pop(key, None)
            if t is not None:
                release_nvl_dist_tensor(t)
        ctx.clear()
        self._ctx = None
        self._comm_stream = None
        self._destroyed = True

    def __del__(self):
        if getattr(self, "_destroyed", True):
            return
        if getattr(self, "explicitly_destroy", False):
            warnings.warn("Buffer was not destroyed explicitly.", ResourceWarning)
            return
        try:
            self.destroy()
        except Exception as exc:
            warnings.warn(f"Buffer destructor failed: {exc!r}", ResourceWarning)

    @staticmethod
    def _record_streams(tensors, stream: torch.cuda.Stream) -> None:
        for t in tensors:
            if t is not None:
                t.record_stream(stream)

    @staticmethod
    def _plan_runtime_tensors(plan: MoonEPCommPlan):
        return (plan.dst, plan.experts_to_copy, plan.zero_fill_ranges, plan.remote_stats,
                plan.dup_groups, plan.dup_loffs, plan.dup_counts)

    def _run_dispatch_on_current_stream(self, ctx, hidden_sh, route_weights_sk, planning_args, plan,
                                        hidden_nvsh, route_weights_nvs, *, inter_rank_sync, zero_copy):
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)
        if planning_args is not None:
            topk_flat, tokens_per_expert, cu_seqlens = planning_args
            launch_planning(ctx, topk_flat, tokens_per_expert, cu_seqlens, plan)
        launch_dispatch(ctx, hidden_sh, route_weights_sk, plan, build_dedup_map=planning_args is not None, pdl_trigger=self.enable_pdl)
        launch_dispatch_epilogue(ctx, plan, pdl_launch=self.enable_pdl)
        if not zero_copy:
            hidden_nvsh.copy_(ctx["hidden_buf_local"])
            if route_weights_nvs is not None:
                route_weights_nvs.copy_(ctx["weights_buf_local"].view(torch.float32))

    def _run_combine_on_current_stream(self, ctx, hidden_sh, plan, hidden_nvsh, route_weights_nvs,
                                       route_weights_sk, *, inter_rank_sync, zero_copy):
        if inter_rank_sync:
            launch_inter_rank_sync(ctx)
        if not zero_copy:
            ctx["hidden_buf_local"].copy_(hidden_nvsh)
            if route_weights_nvs is not None:
                ctx["weights_buf_local"].copy_(route_weights_nvs.view(torch.int32))
        launch_combine_prologue(ctx, plan, pdl_trigger=self.enable_pdl)
        launch_combine(ctx, hidden_sh, plan.dst, output_sk=route_weights_sk, pdl_launch=self.enable_pdl)

    def dispatch(self, hidden_sh, route_weights_sk=None, topk_experts_sk=None, tokens_per_expert=None,
                 plan=None, async_finish=False, *, inter_rank_sync=True, zero_copy=False):
        ctx = self._require_ctx()
        if plan is None:
            assert topk_experts_sk is not None and tokens_per_expert is not None
            plan, cu_seqlens = allocate_planning_outputs(ctx)
            planning_args = (topk_experts_sk.reshape(-1), tokens_per_expert, cu_seqlens)
        else:
            cu_seqlens, planning_args = None, None
        if zero_copy:
            hidden_nvsh = ctx["hidden_buf_local"]
            route_weights_nvs = ctx["weights_buf_local"].view(torch.float32) if route_weights_sk is not None else None
        else:
            hidden_nvsh = torch.empty_like(ctx["hidden_buf_local"])
            route_weights_nvs = torch.empty(ctx["NvS"], dtype=torch.float32, device=ctx["meta_buf"].device) if route_weights_sk is not None else None
        if not async_finish:
            self._run_dispatch_on_current_stream(ctx, hidden_sh, route_weights_sk, planning_args, plan,
                                                 hidden_nvsh, route_weights_nvs, inter_rank_sync=inter_rank_sync, zero_copy=zero_copy)
            return hidden_nvsh, route_weights_nvs, cu_seqlens, plan
        comm = self._comm_stream
        tensors = [hidden_sh, route_weights_sk, hidden_nvsh, route_weights_nvs, *self._plan_runtime_tensors(plan)]
        if planning_args is not None:
            tensors.extend(planning_args)
        self._record_streams(tensors, comm)
        comm.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(comm):
            self._run_dispatch_on_current_stream(ctx, hidden_sh, route_weights_sk, planning_args, plan,
                                                 hidden_nvsh, route_weights_nvs, inter_rank_sync=inter_rank_sync, zero_copy=zero_copy)
            done = comm.record_event()
        return hidden_nvsh, route_weights_nvs, cu_seqlens, plan, done

    def prefetch_weight(self, plan=None, async_finish=False, *, full_gate_weight=None, full_up_weight=None, full_down_weight=None):
        ctx = self._require_ctx()
        assert isinstance(plan, MoonEPCommPlan)
        args = (full_gate_weight, full_up_weight, full_down_weight)
        assert all(w is not None for w in args)
        if not async_finish:
            _launch_full_weight_prefetches(ctx, *args, plan.experts_to_copy)
            return None
        comm = self._comm_stream
        self._record_streams((plan.experts_to_copy, *args), comm)
        comm.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(comm):
            _launch_full_weight_prefetches(ctx, *args, plan.experts_to_copy)
            return comm.record_event()

    def combine(self, plan=None, hidden_nvsh=None, route_weights_nvs=None, async_finish=False,
                inter_rank_sync=True, *, zero_copy=False):
        ctx = self._require_ctx()

        assert isinstance(plan, MoonEPCommPlan), "Buffer.combine: plan is required"
        assert hidden_nvsh is not None
        assert hidden_nvsh.dtype == torch.bfloat16
        assert hidden_nvsh.is_contiguous()
        assert tuple(hidden_nvsh.shape) == (int(ctx["NvS"]), int(ctx["H"]))
        if route_weights_nvs is not None:
            assert route_weights_nvs.dtype == torch.float32
            assert route_weights_nvs.is_contiguous()
            assert tuple(route_weights_nvs.shape) == (int(ctx["NvS"]),)
        if zero_copy:
            assert hidden_nvsh.data_ptr() == ctx["hidden_buf_local"].data_ptr(), (
                "combine(zero_copy=True): hidden_nvsh must alias the NVL shard "
                "view returned by dispatch(zero_copy=True)"
            )
            if route_weights_nvs is not None:
                assert route_weights_nvs.data_ptr() == ctx["weights_buf_local"].view(torch.float32).data_ptr(), (
                    "combine(zero_copy=True): route_weights_nvs must alias "
                    "the NVL weights view returned by dispatch(zero_copy=True)"
                )
        hidden_sh = torch.empty(int(ctx["S"]), int(ctx["H"]), dtype=hidden_nvsh.dtype, device=hidden_nvsh.device)
        route_weights_sk = torch.empty(int(ctx["S"]), int(ctx["K"]), dtype=torch.float32, device=hidden_nvsh.device) if route_weights_nvs is not None else None
        if not async_finish:
            self._run_combine_on_current_stream(ctx, hidden_sh, plan, hidden_nvsh, route_weights_nvs, route_weights_sk,
                                                inter_rank_sync=inter_rank_sync, zero_copy=zero_copy)
            return hidden_sh, route_weights_sk, None
        comm = self._comm_stream
        self._record_streams((hidden_sh, *self._plan_runtime_tensors(plan), hidden_nvsh, route_weights_nvs, route_weights_sk), comm)
        comm.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(comm):
            self._run_combine_on_current_stream(ctx, hidden_sh, plan, hidden_nvsh, route_weights_nvs, route_weights_sk,
                                                inter_rank_sync=inter_rank_sync, zero_copy=zero_copy)
            return hidden_sh, route_weights_sk, comm.record_event()

    def reduce_grad(self, plan=None, async_finish=False, full_gate_grad=None, full_up_grad=None, full_down_grad=None,
                    gate_reduce_buffer=None, up_reduce_buffer=None, down_reduce_buffer=None):
        ctx = self._require_ctx()
        assert isinstance(plan, MoonEPCommPlan)
        args = (full_gate_grad, full_up_grad, full_down_grad, gate_reduce_buffer, up_reduce_buffer, down_reduce_buffer)
        assert all(t is not None for t in args)
        if not async_finish:
            _launch_full_grad_reduces(ctx, plan.experts_to_copy, *args)
            return None
        comm = self._comm_stream
        self._record_streams((plan.experts_to_copy, *args), comm)
        comm.wait_event(torch.cuda.current_stream().record_event())
        with torch.cuda.stream(comm):
            _launch_full_grad_reduces(ctx, plan.experts_to_copy, *args)
            return comm.record_event()
