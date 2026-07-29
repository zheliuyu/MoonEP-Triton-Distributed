#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

echo "== import moonep_td =="
python -c "
try:
    from moonep_td import Buffer, MoonEPCommPlan
    print('ok', Buffer, MoonEPCommPlan)
except ImportError as e:
    print('skip: triton_dist not installed —', e)
"

echo "== static pytest =="
python -m pytest tests/test_layout.py tests/test_planning_reference_static.py tests/test_kernel_compile.py -v --tb=short

echo "== API parity (optional: upstream moonep, NOT moonep_td) =="
if python -c "import moonep" 2>/dev/null; then
  python -m pytest tests/test_api_signatures.py -v --tb=short
else
  echo "skip: upstream moonep not installed (optional — compares moonep vs moonep_td signatures)"
fi

echo "STATIC CHECK PASSED"
