#!/usr/bin/env python3
"""Prefetch bandwidth benchmark (2-GPU quick + 8-GPU full cases)."""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.distributed as dist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from benchmarks._bench_common import setup_dist, time_gpu_op, skip_if_insufficient_gpus
from moonep_td.buffer import create_nvl_single_owner_tensor, pad_dim0_for_alignment
from moonep_td.prefetch import launch_prefetch


QUICK = {"label": "quick_2gpu", "epn": 4, "H": 1024, "Hp": 128, "base_B": 8, "counts": [2, 1, 0, 0]}
FULL = {"label": "full_8gpu", "epn": 8, "H": 3584, "Hp": 3072, "base_B": 14, "counts": [3, 0, 2, 1, 3, 0, 2, 1]}


def expert_plan(R, B, epn, counts, dev):
    plan = torch.full((R, B), -1, dtype=torch.int32, device=dev)
    for e in range(epn):
        for i in range(counts[e]):
            r = i % R
            plan[r, e] = r * epn + e
    return plan


def bench_case(case, rank, R, num_sms, warmup, iters):
    dev = f"cuda:{rank}"
    epn = int(case["epn"])
    H, Hp = int(case["H"]), int(case["Hp"])
    B = pad_dim0_for_alignment([int(case["base_B"]), H, Hp], torch.bfloat16)
    padded_E = pad_dim0_for_alignment([epn, H, Hp], torch.bfloat16)
    remote = create_nvl_single_owner_tensor([padded_E, H, Hp], torch.bfloat16, 0, rank, R)
    prefetch_buf = torch.empty(B, H, Hp, dtype=torch.bfloat16, device=dev)
    etc = expert_plan(R, B, epn, case["counts"], dev)[rank]

    def _run():
        launch_prefetch(remote[:epn], prefetch_buf, etc, num_sms=num_sms)

    us = time_gpu_op(_run, warmup, iters, dist.group.WORLD)
    n_prefetch = int((etc >= 0).sum().item())
    bytes_moved = n_prefetch * H * Hp * 2
    return us, bytes_moved / us if us else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--num-sms", type=int, default=32)
    args = parser.parse_args()

    rank, R, _ = setup_dist()
    if args.full:
        skip_if_insufficient_gpus(8, "bench_prefetch --full")
        case = FULL
    else:
        case = QUICK

    us, gbps = bench_case(case, rank, R, args.num_sms, args.warmup, args.iters)
    if rank == 0:
        print(f"bench_prefetch {case['label']}: {us:.1f} us  ({gbps:.3f} GB/s effective)")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
