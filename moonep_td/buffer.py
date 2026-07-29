"""
Symmetric memory helpers (parallel to MoonEP moonep.buffer + csrc/).

MoonEP uses CUDA VMM + IPC via ``moonep._C``; this module uses NVSHMEM via
Triton-distributed instead.  Function names mirror the original where possible.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

_ELEM_SIZE = {
    torch.float32: 4,
    torch.bfloat16: 2,
    torch.int32: 4,
}

_NVSHMEM_READY = False


def pad_to_granularity(nbytes: int, gran_bytes: int = 128) -> int:
    return ((nbytes + gran_bytes - 1) // gran_bytes) * gran_bytes


def pad_dim0_for_alignment(chunk_shape: list[int], dtype: torch.dtype, gran_bytes: int = 128) -> int:
    """Compute padded dim0 for row alignment (MoonEP buffer.pad_dim0_for_alignment)."""
    elem_size = _ELEM_SIZE[dtype]
    inner_size = elem_size
    for d in chunk_shape[1:]:
        inner_size *= d
    nbytes = chunk_shape[0] * inner_size
    padded_bytes = pad_to_granularity(nbytes, gran_bytes)
    padded_dim0 = padded_bytes // inner_size
    while padded_dim0 * inner_size % gran_bytes != 0:
        padded_dim0 += 1
    return padded_dim0


def ensure_nvshmem_initialized(group: dist.ProcessGroup | None = None) -> None:
    global _NVSHMEM_READY
    if _NVSHMEM_READY:
        return
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before moonep_td.Buffer. "
            "Call dist.init_process_group then ensure_nvshmem_initialized()."
        )
    from triton_dist.utils import init_nvshmem_by_torch_process_group, is_shmem_initialized, nvshmem_barrier_all_on_stream

    if not is_shmem_initialized():
        pg = group if group is not None else dist.group.WORLD
        init_nvshmem_by_torch_process_group(pg)
    nvshmem_barrier_all_on_stream(torch.cuda.current_stream())
    _NVSHMEM_READY = True


def view_nvl_dist_rows(full: torch.Tensor, world_size: int, B: int) -> torch.Tensor:
    """View a MoonEP-style ``[R * B, ...]`` NVL buffer as ``[R, B, ...]``."""
    assert full.shape[0] == world_size * B, (
        f"expected dim0={world_size * B}, got {full.shape[0]}"
    )
    tail = full.shape[1:]
    return full.view(world_size, B, *tail)


def nvl_dist_peer_row(full: torch.Tensor, row: int, rank: int, world_size: int, B: int) -> torch.Tensor:
    """Return EP rank ``row``'s ``[B, ...]`` slice from an NVL-distributed buffer.

    MoonEP maps every rank's chunk into one VA; NVSHMEM keeps one symmetric
    heap per PE, so remote rows must be read through ``get_peer_tensor``.
    """
    if full.ndim >= 2 and full.shape[0] == world_size and full.shape[1] == B:
        buf = full
    else:
        buf = view_nvl_dist_rows(full, world_size, B)
    if row == rank:
        return buf[row]
    import nvshmem.core
    return nvshmem.core.get_peer_tensor(buf, row)[row]


def create_nvl_dist_tensor(
    shape: list[int],
    dtype: torch.dtype,
    local_rank: int,
    world_size: int,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """NVSHMEM symmetric tensor (MoonEP: create_nvl_dist_tensor via VMM).

    A 3D chunk ``[B, H, H']`` is stored MoonEP-style as ``[R * B, H, H']``.
    All other shapes are allocated verbatim (expert stacks, meta, hidden_buf, …).
    """
    del local_rank, group
    from triton_dist.utils import nvshmem_create_tensor

    if len(shape) == 3:
        shape = [world_size * shape[0], *shape[1:]]
    return nvshmem_create_tensor(tuple(shape), dtype=dtype)


def release_nvl_dist_tensor(tensor: torch.Tensor) -> None:
    from triton_dist.utils import nvshmem_free_tensor_sync
    nvshmem_free_tensor_sync(tensor)


def create_nvl_single_owner_tensor(
    shape: list[int],
    dtype: torch.dtype,
    owner_rank: int,
    local_rank: int,
    world_size: int | None = None,
    group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Test helper mirroring MoonEP single-owner expert mapping via NVSHMEM.

    Allocates symmetric ``[world_size, E_padded, H, H']`` storage; every rank
    can read ``full[owner_rank]`` (equivalent to NVLink IPC mapping).
    """
    if world_size is None:
        world_size = dist.get_world_size(group=group)
    padded_E, H, Hp = shape[0], shape[1], shape[2]
    full = create_nvl_dist_tensor(
        [world_size, padded_E, H, Hp], dtype, local_rank, world_size, group=group,
    )
    return full[owner_rank]
