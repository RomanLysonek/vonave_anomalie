#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PROFILE="${PROFILE:-overnight}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overnight_anomaly_search}"
LOG_FILE="$OUTPUT_DIR/search.log"

mkdir -p "$OUTPUT_DIR"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This launcher is tuned for macOS/Apple Silicon. Use the Python command directly on other systems." >&2
  exit 2
fi

{
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] profile=$PROFILE output=$OUTPUT_DIR"
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] checking MPS and search configuration"
  uv run python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"mps_built={torch.backends.mps.is_built()}")
print(f"mps_available={torch.backends.mps.is_available()}")
if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable in this uv environment")
PY
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] starting resumable search"
} 2>&1 | tee -a "$LOG_FILE"

caffeinate -dimsu uv run python ml/run_overnight_anomaly_search.py \
  --profile "$PROFILE" \
  --output-dir "$OUTPUT_DIR" \
  --device mps \
  --resume \
  "$@" 2>&1 | tee -a "$LOG_FILE"
