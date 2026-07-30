import os
import signal

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    config.addinivalue_line("markers", "eight_rank: requires 8 visible GPUs")
    config.addinivalue_line("markers", "large_hidden: H=7168 NVSHMEM stress cases")
    config.addinivalue_line("markers", "i64_offset: int64 byte-offset stress cases")
    config.addinivalue_line("markers", "kernel_compile: Triton compile-only tests")

    sec = os.environ.get("GPU_TEST_TIMEOUT_SEC", "").strip()
    if sec:
        try:
            val = float(sec)
        except ValueError:
            pytest.exit(f"invalid GPU_TEST_TIMEOUT_SEC={sec!r}", returncode=2)
        if val > 0 and not hasattr(signal, "SIGALRM"):
            pytest.exit(
                "GPU_TEST_TIMEOUT_SEC requires SIGALRM (not available on this platform)",
                returncode=2,
            )


def _gpu_test_timeout_sec():
    if os.environ.get("GPU_TEST_TIMEOUT_SKIP", "1") != "1":
        return None
    raw = os.environ.get("GPU_TEST_TIMEOUT_SEC", "").strip()
    if not raw:
        return None
    try:
        sec = float(raw)
    except ValueError:
        return None
    return sec if sec > 0 else None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Skip (no failure, no stack dump) when a single test body exceeds the budget."""
    sec = _gpu_test_timeout_sec()
    if sec is None:
        yield
        return

    def _on_alarm(signum, frame):
        pytest.skip(f"exceeded GPU_TEST_TIMEOUT_SEC={sec}s")

    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, sec, 0.0)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0, 0.0)
        signal.signal(signal.SIGALRM, previous)


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
