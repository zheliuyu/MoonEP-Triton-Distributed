"""Lazy import of triton_dist (TD-only; no MoonEP equivalent)."""

from __future__ import annotations

import functools


@functools.lru_cache(maxsize=1)
def triton_dist():
    import triton_dist as td
    return td
