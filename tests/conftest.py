import os

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    config.addinivalue_line("markers", "eight_rank: requires 8 visible GPUs")
    config.addinivalue_line("markers", "large_hidden: H=7168 NVSHMEM stress cases")
    config.addinivalue_line("markers", "i64_offset: int64 byte-offset stress cases")
    config.addinivalue_line("markers", "kernel_compile: Triton compile-only tests")


def pytest_collection_modifyitems(items):
    """Run i64 grad_reduce before large_hidden tests.

    After large_hidden dispatch/combine, the 32 GiB NVSHMEM heap is heavily
    used; the i64 grad_reduce case then needs another ~11 GiB CUDA allocation
    and can OOM or wedge NCCL. Running it first keeps the combined P0 suite
    reliable on 4×80GB hosts.
    """
    i64_grad = []
    rest = []
    for item in items:
        nodeid = item.nodeid
        if "test_grad_reduce.py" in nodeid and "i64_offset_7168x3072" in nodeid:
            i64_grad.append(item)
        else:
            rest.append(item)
    items[:] = i64_grad + rest


@pytest.fixture(scope="session")
def gpu_count():
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


@pytest.fixture(autouse=True)
def cleanup_moonep_buffers():
    yield
    import gc

    from tests.kernel_test_utils import destroy_active_buffers

    destroy_active_buffers()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


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
