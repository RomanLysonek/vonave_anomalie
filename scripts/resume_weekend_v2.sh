#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PROFILE="${PROFILE:-weekend-v2}" OUTPUT_DIR="${OUTPUT_DIR:-outputs/weekend_v2_search}" \
  scripts/run_weekend_v2.sh "$@"
