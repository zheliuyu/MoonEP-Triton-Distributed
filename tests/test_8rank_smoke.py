"""8-rank smoke tests — written for R=8 CI, skipped when fewer GPUs are available."""

import os

import pytest
import torch
import torch.distributed as dist

from tests.kernel_test_utils import KernelCase, init_case, skip_if_unsupported_world_size


def _gpu_count() -> int:
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


requires_8_gpus = pytest.mark.skipif(
    _gpu_count() < 8,
    reason="8-rank smoke requires >= 8 visible GPUs (skipped on 2-GPU dev machines)",
)

pytestmark = [pytest.mark.eight_rank, requires_8_gpus]


@pytest.fixture(scope="module")
def dist_env_8():
    if "RANK" not in os.environ:
        pytest.skip("8-rank tests must be launched with torchrun")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    from moonep_td.buffer import ensure_nvshmem_initialized

    ensure_nvshmem_initialized()
    R = dist.get_world_size()
    if R != 8:
        pytest.skip(f"8-rank smoke requires torchrun --nproc_per_node=8, got R={R}")
    yield rank, R
    dist.barrier(device_ids=[rank])


@requires_8_gpus
def test_8rank_planning_smoke(dist_env_8):
    rank, R = dist_env_8
    case = KernelCase("8rank_planning", S=64, K=4, epn=8, H=128, num_sms=8, B=2, min_R=8)
    skip_if_unsupported_world_size(case, R)
    ctx = init_case(case, R)
    from moonep_td.planning import allocate_planning_outputs, launch_planning

    dev = f"cuda:{rank}"
    topk = torch.randint(0, case.E(R), (case.S, case.K), device=dev, dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=case.E(R)).to(torch.int32)
    plan, cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1), tpe, cu, plan)
    torch.cuda.synchronize()
    assert plan.dst.numel() == case.S * case.K


@requires_8_gpus
def test_8rank_dispatch_combine_smoke(dist_env_8):
    rank, R = dist_env_8
    case = KernelCase("8rank_dispatch_combine", S=64, K=4, epn=8, H=128, num_sms=8, B=2, min_R=8)
    skip_if_unsupported_world_size(case, R)
    ctx = init_case(case, R)
    from moonep_td.planning import allocate_planning_outputs, launch_planning
    from moonep_td.dispatch import launch_dispatch
    from moonep_td.dispatch_epilogue import launch_dispatch_epilogue
    from moonep_td.combine_prologue import launch_combine_prologue
    from moonep_td.combine import launch_combine

    dev = f"cuda:{rank}"
    S, H, K, E = case.S, case.H, case.K, case.E(R)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev)
    topk = torch.randint(0, E, (S, K), device=dev, dtype=torch.int32)
    tpe = torch.bincount(topk.flatten(), minlength=E).to(torch.int32)
    plan, cu = allocate_planning_outputs(ctx)
    launch_planning(ctx, topk.reshape(-1), tpe, cu, plan)
    launch_dispatch(ctx, hidden, weights, plan, build_dedup_map=True)
    launch_dispatch_epilogue(ctx, plan)
    ctx["hidden_buf_local"].copy_(ctx["hidden_buf_local"])
    launch_combine_prologue(ctx, plan)
    out = torch.empty(S, H, dtype=torch.bfloat16, device=dev)
    launch_combine(ctx, out, plan.dst)
    torch.cuda.synchronize()
    assert out.shape == (S, H)
