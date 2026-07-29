#!/usr/bin/env bash
# Run full GPU test suite and refresh STATUS.md "Last CI run" section.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LOG="$(mktemp)"
TS="$(date -u +"%Y-%m-%d %H:%M UTC")"
NPROC="${NPROC:-2}"
FILTER="${PYTEST_FILTER:-not large_hidden and not i64_offset and not 8rank_smoke}"

set +e
bash scripts/run_gpu_tests.sh 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

PASSED="$(grep -oE '[0-9]+ passed' "$LOG" | tail -1 || true)"
FAILED="$(grep -oE '[0-9]+ failed' "$LOG" | tail -1 || true)"
SKIPPED="$(grep -oE '[0-9]+ skipped' "$LOG" | tail -1 || true)"

STATUS_FILE="$ROOT/STATUS.md"
if [[ -f "$STATUS_FILE" ]]; then
  python3 - "$STATUS_FILE" "$TS" "$NPROC" "$FILTER" "$RC" "$PASSED" "$FAILED" "$SKIPPED" <<'PY'
import sys
from pathlib import Path

path, ts, nproc, filt, rc, passed, failed, skipped = sys.argv[1:9]
text = Path(path).read_text()
block = f"""## Last CI run (auto)

- **Time**: {ts}
- **GPUs**: {nproc} (`NPROC`)
- **Filter**: `{filt}`
- **Result**: {"PASS" if rc == "0" else "FAIL"} (exit {rc})
- **Summary**: {passed or "n/a"}, {failed or "none failed"}, {skipped or "none skipped"}
"""
marker = "## Last CI run (auto)"
if marker in text:
    pre, _ = text.split(marker, 1)
    rest = _.split("\n## ", 1)
    tail = "" if len(rest) == 1 else "\n## " + rest[1]
    text = pre.rstrip() + "\n\n" + block.rstrip() + tail
else:
    text = text.rstrip() + "\n\n" + block
Path(path).write_text(text)
PY
fi

rm -f "$LOG"
exit "$RC"
