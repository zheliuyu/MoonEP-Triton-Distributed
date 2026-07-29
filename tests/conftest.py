import os

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    config.addinivalue_line("markers", "eight_rank: requires 8 visible GPUs")
    config.addinivalue_line("markers", "large_hidden: H=7168 NVSHMEM stress cases")
    config.addinivalue_line("markers", "i64_offset: int64 byte-offset stress cases")
    config.addinivalue_line("markers", "kernel_compile: Triton compile-only tests")


@pytest.fixture(scope="session")
def gpu_count():
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    yield
    import torch

    from tests.kernel_test_utils import destroy_active_buffers

    destroy_active_buffers()
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@pytest.fixture(scope="session")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("distributed kernel tests must be launched with torchrun")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    from moonep_td.buffer import ensure_nvshmem_initialized

    ensure_nvshmem_initialized()

    yield rank, dist.get_world_size()

    dist.barrier(device_ids=[rank])
