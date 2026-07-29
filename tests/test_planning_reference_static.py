"""Planning reference invariants (CPU, mirrors MoonEP tests/planning_reference.py)."""

import torch

from tests.planning_reference import launch_planning_torch_reference


def test_reference_conservation():
    ctx = {
        "rank": 0, "R": 1, "E": 4, "B": 2, "S": 4, "K": 2,
        "NvS_capacity": 8, "NvS": 8, "token_padding": 1, "group": None,
    }
    topk = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32)
    tpe = torch.bincount(topk, minlength=4).to(torch.int32)
    dst, cu, etc, rs, zfr, dedup = launch_planning_torch_reference(ctx, topk, tpe)
    assert dst.shape == (8,)
    assert cu.shape == (6,)
    assert etc.shape == (1, 2)
    assert int(cu.max()) <= ctx["NvS"]
