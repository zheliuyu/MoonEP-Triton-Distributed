#!/usr/bin/env python3
"""
MoonEP-Triton-D dispatch/combine operator benchmark.

2-GPU default: ep=2, H=1024 mini sweep.
8-GPU full:    ep=4/8, H=3584/7168 (requires >= 8 GPUs).

Run:
  torchrun --nproc_per_node=2 benchmarks/bench_comm.py --quick
  torchrun --nproc_per_node=8 benchmarks/bench_comm.py --full   # skipped if <8 GPUs
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks._bench_common import setup_dist, time_gpu_op, skip_if_insufficient_gpus
from tests.generate_topk_routing import generate_topk_routing

from moonep_td import Buffer
from moonep_td.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment, view_nvl_dist_rows
from moonep_td.combine import launch_combine
from moonep_td.combine_prologue import launch_combine_prologue
from moonep_td.dispatch import launch_dispatch
from moonep_td.dispatch_epilogue import launch_dispatch_epilogue
from moonep_td.grad_reduce import launch_grad_reduce
from moonep_td.inter_rank_sync import launch_inter_rank_sync
from moonep_td.planning import allocate_planning_outputs, launch_planning
from moonep_td.prefetch import launch_prefetch


def configs(*, quick: bool, full: bool, world_size: int):
    if full:
        if world_size < 8:
            return []
        return [
            {"ep": 4, "S": 8192, "K": 8, "E": 896, "H": 3584, "unbalance_ratio": 1.0},
            {"ep": 8, "S": 8192, "K": 8, "E": 896, "H": 3584, "unbalance_ratio": 1.0},
        ]
    # quick path: always runnable on 2+ GPUs
    ep = min(2, world_size)
    return [
        {"ep": ep, "S": 256, "K": 4, "E": ep * 8, "H": 1024, "unbalance_ratio": 1.0},
    ]


def bench_one(group, group_rank, R, S, K, E, H, bias_ratio, Hp, num_sms, warmup, iters):
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    buffer = Buffer(S, H, K, E, R, num_sms=num_sms, group=group, explicitly_destroy=True)
    ctx = buffer._require_ctx()
    topk, tpe = generate_topk_routing(S, K, E, R, bias_ratio, dev, 1234, rank=group_rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev)
    output = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
    topk_flat = topk.reshape(-1).contiguous()
    plan, cu = allocate_planning_outputs(ctx)

    def _plan():
        launch_planning(ctx, topk_flat, tpe, cu, plan)

    planning_us = time_gpu_op(_plan, warmup, iters, group)
    launch_planning(ctx, topk_flat, tpe, cu, plan)
    dst = plan.dst
    NvS = int(ctx["NvS"])

    dispatch_fwd_us = time_gpu_op(
        lambda: launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True),
        warmup, iters, group,
    )
    epilogue_fwd_us = time_gpu_op(
        lambda: launch_dispatch_epilogue(ctx, plan), warmup, iters, group,
    )
    grad_output = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    dispatch_bwd_us = time_gpu_op(
        lambda: launch_dispatch(ctx, grad_output, None, plan, build_dedup_map=False),
        warmup, iters, group,
    )

    hidden_nvsh = torch.randn(NvS, H, dtype=torch.bfloat16, device=dev)
    ctx["hidden_buf_local"].copy_(hidden_nvsh)
    torch.cuda.synchronize()
    dist.barrier(group=group)

    combine_prologue_us = time_gpu_op(
        lambda: launch_combine_prologue(ctx, plan), warmup, iters, group,
    )
    combine_fwd_us = time_gpu_op(
        lambda: launch_combine(ctx, output, dst), warmup, iters, group,
    )

    B = int(ctx["B"])
    epn = E // R
    padded_B = pad_dim0_for_alignment([B, H, Hp], torch.float32)
    remote = create_nvl_dist_tensor([epn, H, Hp], torch.bfloat16, group_rank, R, group=group)
    prefetch_buf = torch.empty(B, H, Hp, dtype=torch.bfloat16, device=dev)
    etc = plan.experts_to_copy[group_rank]

    def _prefetch():
        launch_prefetch(remote, prefetch_buf, etc, num_sms=num_sms)
        launch_inter_rank_sync(ctx)

    prefetch_us = time_gpu_op(_prefetch, warmup, iters, group)

    reduce_full = create_nvl_dist_tensor([padded_B, H, Hp], torch.float32, group_rank, R, group=group)
    reduce_bufs = view_nvl_dist_rows(reduce_full, R, padded_B)
    full_grad = torch.randn(epn + B, H, Hp, dtype=torch.float32, device=dev)

    grad_us = time_gpu_op(
        lambda: launch_grad_reduce(
            full_grad[:epn], reduce_bufs, plan.experts_to_copy,
            rank=group_rank, num_sms=num_sms,
            meta_buf=ctx["meta_buf"], meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]), grid_sync_bar=ctx["grid_sync_bar"],
        ),
        warmup, iters, group,
    )

    bytes_dispatch = float(S * K * H * 2)
    buffer.destroy()
    return {
        "planning_us": planning_us,
        "dispatch_fwd_us": dispatch_fwd_us,
        "epilogue_fwd_us": epilogue_fwd_us,
        "dispatch_bwd_us": dispatch_bwd_us,
        "combine_prologue_us": combine_prologue_us,
        "combine_fwd_us": combine_fwd_us,
        "prefetch_us": prefetch_us,
        "grad_reduce_us": grad_us,
        "dispatch_fwd_GBps": bytes_dispatch / dispatch_fwd_us if dispatch_fwd_us else 0.0,
        "combine_fwd_GBps": bytes_dispatch / combine_fwd_us if combine_fwd_us else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--hp", type=int, default=128)
    parser.add_argument("--num-sms", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    rank, world_size, _ = setup_dist()
    cfgs = configs(quick=not args.full, full=args.full, world_size=world_size)
    if args.full:
        skip_if_insufficient_gpus(8, "bench_comm --full")

    if rank == 0:
        mode = "full" if args.full else "quick"
        print(f"bench_comm mode={mode} world_size={world_size} configs={len(cfgs)}")

    group = dist.group.WORLD
    for cfg in cfgs:
        ep = cfg["ep"]
        if ep > world_size:
            continue
        if rank == 0:
            print(f"  ep={ep} S={cfg['S']} H={cfg['H']} K={cfg['K']}", flush=True)
        res = bench_one(
            group, rank, ep, cfg["S"], cfg["K"], cfg["E"], cfg["H"],
            cfg["unbalance_ratio"], args.hp, args.num_sms, args.warmup, args.iters,
        )
        dist.barrier()
        if rank == 0:
            print(
                f"    planning={res['planning_us']:.1f}us "
                f"dispatch_fwd={res['dispatch_fwd_us']:.1f}us "
                f"combine_fwd={res['combine_fwd_us']:.1f}us "
                f"({res['dispatch_fwd_GBps']:.2f} GB/s dispatch)",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
