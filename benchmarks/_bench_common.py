"""Shared helpers for MoonEP-Triton-D benchmarks."""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist


def available_gpus() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def require_torchrun() -> None:
    if "RANK" not in os.environ:
        print("benchmarks must be launched with torchrun", file=sys.stderr)
        sys.exit(1)


def setup_dist():
    require_torchrun()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    from moonep_td.buffer import ensure_nvshmem_initialized

    ensure_nvshmem_initialized()
    return dist.get_rank(), dist.get_world_size(), local_rank


def skip_if_insufficient_gpus(min_gpus: int, label: str = "") -> None:
    n = available_gpus()
    if n < min_gpus:
        msg = f"skip {label}: need >={min_gpus} GPUs, have {n}"
        if dist.is_initialized() and dist.get_rank() == 0:
            print(msg)
        sys.exit(0)


def time_gpu_op(launch_fn, warmup, iters, group, *, cudagraph: bool = False):
    """Cross-rank mean per-iteration latency in microseconds."""
    for _ in range(warmup):
        launch_fn()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if cudagraph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                launch_fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        start.record()
        g.replay()
        end.record()
    else:
        start.record()
        for _ in range(iters):
            launch_fn()
        end.record()
    end.synchronize()
    local_us = start.elapsed_time(end) / iters * 1e3

    world_size = dist.get_world_size(group=group)
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    t = torch.tensor([local_us], dtype=torch.float64, device=dev)
    outs = [torch.empty(1, dtype=torch.float64, device=dev) for _ in range(world_size)]
    dist.all_gather(outs, t, group=group)
    return torch.cat(outs).mean().item()
