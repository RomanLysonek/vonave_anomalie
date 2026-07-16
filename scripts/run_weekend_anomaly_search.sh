#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROFILE=weekend OUTPUT_DIR="${OUTPUT_DIR:-outputs/weekend_anomaly_search}" \
  scripts/run_overnight_anomaly_search.sh "$@"
