"""Smoke test for MOONEP_TD_PIPELINE=1 path."""

import os

import pytest
import torch

from tests.kernel_test_utils import KernelCase, init_case, make_topk


@pytest.fixture(scope="module")
def dist_env():
    import torch.distributed as dist

    if "RANK" not in os.environ:
        pytest.skip("pipeline smoke requires torchrun")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    from moonep_td.buffer import ensure_nvshmem_initialized

    ensure_nvshmem_initialized()
    yield rank, dist.get_world_size()


def test_pipeline_dispatch_combine_smoke(dist_env, monkeypatch):
    monkeypatch.setenv("MOONEP_TD_PIPELINE", "1")
    rank, R = dist_env
    case = KernelCase("pipeline_smoke", S=64, K=4, epn=4, H=128, num_sms=8, B=2)
    ctx = init_case(case, R)
    from moonep_td.planning import allocate_planning_outputs, launch_planning
    from moonep_td.dispatch import launch_dispatch
    from moonep_td.combine_prologue import launch_combine_prologue
    from moonep_td.combine import launch_combine

    dev = f"cuda:{rank}"
    hidden = torch.randn(case.S, case.H, dtype=torch.bfloat16, device=dev)
    weights = torch.rand(case.S, case.K, dtype=torch.float32, device=dev)
    topk, tpe = make_topk(case, rank, R)
    plan, cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1), tpe, cu, plan)
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    ctx["hidden_buf_local"].copy_(ctx["hidden_buf_local"])
    launch_combine_prologue(ctx, plan)
    out = torch.empty(case.S, case.H, dtype=torch.bfloat16, device=dev)
    launch_combine(ctx, out, plan.dst)
    torch.cuda.synchronize()
    assert out.shape == (case.S, case.H)
