"""Pipeline mode toggle (Phase 13 software pipelining)."""

from __future__ import annotations

import os


def pipeline_enabled() -> bool:
    raw = os.environ.get("MOONEP_TD_PIPELINE", "0")
    return raw not in ("0", "false", "False", "")


def pipeline_stages() -> int:
    return max(2, int(os.environ.get("MOONEP_TD_PIPELINE_STAGES", "2")))
