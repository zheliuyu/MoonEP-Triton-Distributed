#!/usr/bin/env python3
"""Grad-reduce bandwidth benchmark (2-GPU quick + 8-GPU full cases)."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks._bench_common import setup_dist, time_gpu_op, skip_if_insufficient_gpus
from moonep_td.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment, view_nvl_dist_rows
from moonep_td.grad_reduce import launch_grad_reduce


QUICK_CASE = {"label": "quick_2gpu", "epn": 4, "H": 1024, "Hp": 128, "base_B": 8, "counts": [2, 1, 0, 0]}
FULL_CASE = {"label": "full_8gpu", "epn": 8, "H": 3584, "Hp": 3072, "base_B": 14, "counts": [3, 3, 2, 3, 1, 0, 2, 3]}


def expert_plan(R, B, epn, counts, dev):
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for i in range(counts[e]):
            r = i % R
            plan[r, e] = e
    return plan


def bench_case(case, rank, R, num_sms, warmup, iters, ctx):
    dev = f"cuda:{rank}"
    epn = int(case["epn"])
    E = R * epn
    H, Hp = int(case["H"]), int(case["Hp"])
    B = pad_dim0_for_alignment([int(case["base_B"]), H, Hp], torch.float32)
    counts = case["counts"]
    plan = expert_plan(R, B, epn, counts, dev)
    reduce_full = create_nvl_dist_tensor([B, H, Hp], torch.float32, rank, R)
    reduce_bufs = view_nvl_dist_rows(reduce_full, R, B)
    reduce_bufs[rank].zero_()
    full_grad = torch.randn(E, H, Hp, dtype=torch.float32, device=dev)

    def _run():
        launch_grad_reduce(
            full_grad, reduce_bufs, plan, rank=rank, num_sms=num_sms,
            meta_buf=ctx["meta_buf"], meta_stride=int(ctx["meta_chunk_padded"]),
            barrier_off=int(ctx["BARRIER_OFF"]), grid_sync_bar=ctx["grid_sync_bar"],
        )

    us = time_gpu_op(_run, warmup, iters, dist.group.WORLD)
    n_slots = sum(counts)
    bytes_moved = n_slots * H * Hp * 4
    return us, bytes_moved / us if us else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-sms", type=int, default=32)
    args = parser.parse_args()

    rank, R, _ = setup_dist()
    from moonep_td import Buffer

    if args.full:
        skip_if_insufficient_gpus(8, "bench_grad_reduce --full")
        case = FULL_CASE
    else:
        case = QUICK_CASE

    buffer = Buffer(128, case["H"], 1, R * case["epn"], R, num_sms=args.num_sms, explicitly_destroy=True)
    ctx = buffer._require_ctx()
    us, gbps = bench_case(case, rank, R, args.num_sms, args.warmup, args.iters, ctx)
    buffer.destroy()
    if rank == 0:
        print(f"bench_grad_reduce {case['label']}: {us:.1f} us  ({gbps:.3f} GB/s effective)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
