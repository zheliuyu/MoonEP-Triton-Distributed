"""Pre-planning inter-rank sync (parallel to MoonEP moonep.inter_rank_sync)."""

from __future__ import annotations

from moonep_td._common import launch_cross_rank_barrier


def launch_inter_rank_sync(ctx: dict) -> None:
    launch_cross_rank_barrier(ctx)
