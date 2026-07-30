"""
Remote expert prefetch correctness test.

Run with:
    torchrun --nproc_per_node=8 -m pytest -s tests/test_prefetch.py

Coverage axes:
  - shapes: single tile, multi tile, rectangular tiles, production
    7168x3072 (>2^31-element expert offsets at expert ids >= 98).
  - plans: hand-picked (holes, duplicates, extremes of the expert range),
    empty (all -1, total_tiles == 0), random multi-seed.
  - pipeline: tiles-per-CTA both below and far above the smem stage count,
    so the producer/consumer state and the deferred consumer_release wrap
    many times within one launch.

grad_reduce-style barrier/repeated-launch cases don't apply here: prefetch
is a one-sided local-plan copy with no cross-rank sync and no state that
survives a launch.
"""

import os

import pytest
import torch
import torch.distributed as dist

from moonep_td.buffer import create_nvl_dist_tensor, pad_dim0_for_alignment, release_nvl_dist_tensor
from moonep_td.prefetch import launch_prefetch

# NVSHMEM symmetric tensors created per prefetch case (freed in run_case).
_PREFETCH_NVL_TENSORS: list[torch.Tensor] = []


def _random_experts(E, B, seed):
    """Random plan with ~40% holes and possible duplicates. Local to each
    rank's launch, so it doesn't need to agree across ranks; seeding keeps
    failures reproducible."""
    g = torch.Generator()
    g.manual_seed(seed)
    live = torch.rand(B, generator=g) < 0.6
    ids = torch.randint(0, E, (B,), generator=g, dtype=torch.int32)
    return torch.where(live, ids, torch.full_like(ids, -1)).tolist()


CASES = [
    {
        "name": "single_tile_one_buffer",
        "E": 4,
        "H": 128,
        "Hp": 128,
        "B": 1,
        "num_sms": 1,
        "owner_offset": 1,
        "experts": lambda E: [E - 1],
    },
    {
        "name": "multi_tile_with_unused_slot",
        "E": 8,
        "H": 256,
        "Hp": 256,
        "B": 4,
        "num_sms": 8,
        "owner_offset": 1,
        "experts": lambda E: [0, E - 1, E // 2, -1],
    },
    {
        "name": "rectangular_wide_tiles_duplicates",
        "E": 8,
        "H": 256,
        "Hp": 384,
        "B": 5,
        "num_sms": 16,
        "owner_offset": 2,
        "experts": lambda E: [E - 1, 0, E // 2, E // 2, -1],
    },
    {
        "name": "more_sms_than_tiles",
        "E": 16,
        "H": 512,
        "Hp": 128,
        "B": 3,
        "num_sms": 32,
        "owner_offset": 3,
        "experts": lambda E: [E - 1, 1, -1],
    },
    # All -1: total_tiles == 0. The kernel must pass the warp split and the
    # final cp_async_bulk_wait_group(0) without issuing a single load/store,
    # leaving every sentinel intact.
    {
        "name": "empty_plan",
        "E": 8,
        "H": 128,
        "Hp": 128,
        "B": 4,
        "num_sms": 8,
        "owner_offset": 1,
        "experts": lambda E: [-1, -1, -1, -1],
    },
    # 96 tiles on 2 CTAs = 48 tiles per CTA >> stages (<= 6): the load
    # pipeline and the deferred consumer_release wrap many times within one
    # launch. The other cases give each CTA at most ~2 tiles and never wrap.
    {
        "name": "pipeline_wraparound",
        "E": 16,
        "H": 512,
        "Hp": 384,
        "B": 8,
        "num_sms": 2,
        "owner_offset": 1,
        "experts": lambda E: [3, E - 1, 3, 0, E // 2, E - 1, 7, 1],
    },
    {
        "name": "i64_offset_7168x3072",
        "E": 104,
        "H": 7168,
        "Hp": 3072,
        "B": 3,
        "num_sms": 32,
        "owner_offset": 1,
        "experts": lambda E: [E - 1, 98, -1],
    },
    # Random plans (holes, duplicates, uneven coverage) at B=16.
    {
        "name": "random_plan_s1",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "owner_offset": 2,
        "experts": lambda E: _random_experts(E, 16, seed=1),
    },
    {
        "name": "random_plan_s2",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "owner_offset": 1,
        "experts": lambda E: _random_experts(E, 16, seed=2),
    },
    {
        "name": "random_plan_s3",
        "E": 32,
        "H": 256,
        "Hp": 128,
        "B": 16,
        "num_sms": 16,
        "owner_offset": 3,
        "experts": lambda E: _random_experts(E, 16, seed=3),
    },
]


@pytest.fixture(scope="module")
def dist_env():
    if "RANK" not in os.environ:
        pytest.skip("test_prefetch requires torchrun with at least 2 GPUs")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    from moonep_td.buffer import ensure_nvshmem_initialized

    ensure_nvshmem_initialized()
    R = dist.get_world_size()
    if R < 2:
        if owns_process_group:
            dist.destroy_process_group()
        pytest.skip("test_prefetch requires at least 2 GPUs")

    yield rank, R

    dist.barrier(device_ids=[rank])


def _fill_owner_experts(dst, E, H, Hp, seed, device):
    """Fill owner expert rows without a single E×H×Hp temporary."""
    gen = torch.Generator(device=device).manual_seed(seed)
    dst.zero_()
    elems_per_expert = H * Hp
    chunk = max(1, min(E, (256 * 1024 * 1024) // max(1, elems_per_expert * 2)))
    for e0 in range(0, E, chunk):
        e1 = min(E, e0 + chunk)
        dst[e0:e1].copy_(
            torch.randn(
                e1 - e0, H, Hp, dtype=torch.bfloat16, device=device, generator=gen
            )
        )


def make_single_owner_experts(rank, R, E, H, Hp):
    """One symmetric [R, E_pad, H, Hp] tensor; row ``owner`` is filled on that rank."""
    padded_E = pad_dim0_for_alignment([E, H, Hp], torch.bfloat16)
    full = create_nvl_dist_tensor([R, padded_E, H, Hp], torch.bfloat16, rank, R)
    _PREFETCH_NVL_TENSORS.append(full)
    for owner in range(R):
        if rank == owner:
            seed = 2026 + owner + E * 13 + H * 17 + Hp * 19
            _fill_owner_experts(
                full[owner, :E], E, H, Hp, seed, f"cuda:{rank}"
            )
            if padded_E > E:
                full[owner, E:].zero_()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[rank])

    import nvshmem.core as nvs

    owners = []
    for owner in range(R):
        view = full[owner, :E] if rank == owner else nvs.get_peer_tensor(full, owner)[owner, :E]
        owners.append(view)
    return owners


def remote_owner_for(rank, R, owner_offset):
    owner_offset %= R
    if owner_offset == 0:
        owner_offset = 1
    return (rank + owner_offset) % R


def _release_prefetch_nvl_tensors():
    while _PREFETCH_NVL_TENSORS:
        release_nvl_dist_tensor(_PREFETCH_NVL_TENSORS.pop())


def run_case(rank, R, case):
    E = case["E"]
    H = case["H"]
    Hp = case["Hp"]
    B = case["B"]
    num_sms = case["num_sms"]
    dev = f"cuda:{rank}"

    assert H % 128 == 0 and Hp % 128 == 0, \
        f"{case['name']}: H/Hp must be multiples of 128"

    try:
        all_remote = make_single_owner_experts(rank, R, E, H, Hp)
        remote_owner = remote_owner_for(rank, R, case["owner_offset"])
        remote_expert = all_remote[remote_owner]
        assert remote_owner != rank, f"{case['name']}: remote owner must not be local"
        assert remote_expert.is_contiguous(), "leading-dim slice should stay contiguous"

        expert_ids = case["experts"](E)
        assert len(expert_ids) == B
        experts_to_copy = torch.tensor(expert_ids, dtype=torch.int32, device=dev)
        sentinel = -123.0
        prefetch_buffers = torch.full(
            (B, H, Hp), sentinel, dtype=torch.bfloat16, device=dev
        )

        launch_prefetch(remote_expert, prefetch_buffers, experts_to_copy, num_sms=num_sms)
        torch.cuda.synchronize()

        ok = True
        for b, e in enumerate(expert_ids):
            if e < 0:
                if not torch.equal(
                    prefetch_buffers[b],
                    torch.full_like(prefetch_buffers[b], sentinel),
                ):
                    ok = False
                    print(
                        f"[rank {rank}] {case['name']} mismatch: unused "
                        f"buffer={b} was modified"
                    )
                continue
            if not torch.equal(prefetch_buffers[b], remote_expert[e]):
                ok = False
                diff = (prefetch_buffers[b].float() - remote_expert[e].float()).abs().max().item()
                print(
                    f"[rank {rank}] {case['name']} mismatch: buffer={b}, "
                    f"expert={e}, remote_owner={remote_owner}, max_abs_diff={diff}"
                )

        ok_tensor = torch.tensor([int(ok)], dtype=torch.int32, device=dev)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        assert ok_tensor.item() == 1, f"{case['name']} failed"

        if rank == 0:
            print(
                f"  [PASS] {case['name']}: E={E}, B={B}, "
                f"H={H}, Hp={Hp}, num_sms={num_sms}"
            )

        dist.barrier(device_ids=[rank])
    finally:
        _release_prefetch_nvl_tensors()
        torch.cuda.empty_cache()


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_prefetch_case(dist_env, case):
    rank, R = dist_env
    if rank == 0:
        print(f"[test_prefetch] Running case={case['name']} with R={R}")
    run_case(rank, R, case)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
