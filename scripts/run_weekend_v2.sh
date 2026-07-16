#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-weekend-v2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/weekend_v2_search}"
PRIOR_ROOT="${PRIOR_ROOT:-outputs/overnight_anomaly_search}"
STAGE="${STAGE:-all}"
MAX_HOURS="${MAX_HOURS:-0}"
LOG_FILE="$OUTPUT_DIR/search.log"

mkdir -p "$OUTPUT_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is tuned for macOS/Apple Silicon." >&2
  exit 2
fi
if [[ ! -f data/train_data.parquet || ! -f data/test_data.parquet ]]; then
  echo "Missing data/train_data.parquet or data/test_data.parquet." >&2
  exit 2
fi
if [[ ! -f "$PRIOR_ROOT/recommendation.json" ]]; then
  echo "Missing $PRIOR_ROOT/recommendation.json; copy the completed overnight output into place." >&2
  exit 2
fi

EXTRA_ARGS=("$@")
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--retry-failed)
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fail-fast)
fi

{
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] profile=$PROFILE stage=$STAGE output=$OUTPUT_DIR prior=$PRIOR_ROOT"
  uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable in this uv environment")
PY
} 2>&1 | tee -a "$LOG_FILE"

caffeinate -dimsu uv run python ml/run_weekend_v2_search.py \
  --profile "$PROFILE" \
  --stage "$STAGE" \
  --prior-root "$PRIOR_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device mps \
  --max-hours "$MAX_HOURS" \
  --resume \
  "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
