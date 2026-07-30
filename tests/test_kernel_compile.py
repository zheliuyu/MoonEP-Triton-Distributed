"""Static compile-all for every @td.jit kernel (no torchrun)."""

import pytest

pytestmark = pytest.mark.kernel_compile


def test_all_kernels_compile():
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "devtools" / "compile_kernels.py"
    env = os.environ.copy()
    env.setdefault("LD_PRELOAD", "/usr/lib/x86_64-linux-gnu/libstdc++.so.6")
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.4")
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    triton_dist = env.get("TRITON_DIST_ROOT", "/root/Triton-distributed")
    env["PYTHONPATH"] = f"{triton_dist}/python{os.pathsep}{env['PYTHONPATH']}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0 and "skip compile_kernels" not in proc.stdout:
        raise AssertionError(
            f"compile_kernels failed (rc={proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
