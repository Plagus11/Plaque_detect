#!/usr/bin/env bash
# CI entry point for calibration regression tests.
#
# Usage:
#   tests/run_regressions.sh           # check against golden (fails on drift)
#   tests/run_regressions.sh freeze    # overwrite golden with current
#
# Exit codes:
#   0 = pass
#   1 = drift detected
#   2 = golden missing or malformed
set -euo pipefail
cd "$(dirname "$0")/.."

CMD="${1:-check}"
echo "[ci] python: $(python3 --version)"
echo "[ci] pwd:     $(pwd)"
echo "[ci] running regression suite (${CMD})"
python3 tests/regression_calibration.py "${CMD}"
