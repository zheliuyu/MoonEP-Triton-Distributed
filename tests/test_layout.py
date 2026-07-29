"""Layout offset tests (logic mirrors MoonEP api._create_context)."""

from moonep_td.buffer import pad_dim0_for_alignment


def _align_up(x, a):
    return ((x + a - 1) // a) * a


def compute_meta_chunk_padded(S, H, K, E, R, B, token_padding=128):
    epn = E // R
    N = S * K
    NvS = S * K + (token_padding - 1) * 2 * epn
    NvS_padded = pad_dim0_for_alignment([NvS, H], __import__("torch").bfloat16)
    TPE_OFF = _align_up(NvS, 4)
    PLAN_OFF = _align_up(TPE_OFF + R * E, 4)
    broadcast_elems = 3 * E * R
    planning_out = broadcast_elems + R * (E + B) + 2 * R * (E + B) + B * R + 2 * R
    N4 = _align_up(N, 4)
    SRC_INFO_OFF = _align_up(PLAN_OFF + planning_out, 4) + N4 * 3 + 3
    meta_chunk_logical = SRC_INFO_OFF + NvS
    return _align_up(meta_chunk_logical * 4, 128) // 4, NvS, NvS_padded


def test_default_config_layout():
    meta_padded, NvS, NvS_padded = compute_meta_chunk_padded(4096, 7168, 8, 256, 8, 32)
    assert NvS >= 4096 * 8
    assert meta_padded > NvS
    assert NvS_padded >= NvS


def test_tiny_layout():
    meta_padded, NvS, _ = compute_meta_chunk_padded(4, 16, 2, 4, 2, 2, token_padding=8)
    assert NvS > 4 * 2
    assert meta_padded > 0
