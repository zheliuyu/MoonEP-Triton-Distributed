"""Static compile-all for every @td.jit kernel (no torchrun)."""

import pytest

pytestmark = pytest.mark.kernel_compile


def test_all_kernels_compile():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "devtools" / "compile_kernels.py"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and "skip compile_kernels" not in proc.stdout:
        raise AssertionError(
            f"compile_kernels failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
