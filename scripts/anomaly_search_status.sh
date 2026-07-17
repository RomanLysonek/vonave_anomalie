#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/overnight_anomaly_search}"

printf 'Output: %s\n' "$OUTPUT_DIR"
printf 'Completed diagnostic trials: %s\n' "$(find "$OUTPUT_DIR" -type f -path '*/diagnostic/*/result.json' 2>/dev/null | wc -l | tr -d ' ')"
printf 'Completed proxy trials:      %s\n' "$(find "$OUTPUT_DIR" -type f -path '*/proxy/*/result.json' 2>/dev/null | wc -l | tr -d ' ')"
printf 'Completed neural trials:     %s\n' "$(find "$OUTPUT_DIR" -type f -path '*/neural/*/result.json' 2>/dev/null | wc -l | tr -d ' ')"
printf 'Completed confirmations:     %s\n' "$(find "$OUTPUT_DIR" -type f -path '*/confirmation/*/result.json' 2>/dev/null | wc -l | tr -d ' ')"
printf 'Failed trials:               %s\n' "$(find "$OUTPUT_DIR" -name failure.json 2>/dev/null | wc -l | tr -d ' ')"

for file in diagnostic_leaderboard.csv proxy_leaderboard.csv neural_leaderboard.csv confirmation_leaderboard.csv; do
  if [[ -s "$OUTPUT_DIR/$file" ]]; then
    printf '\n== %s ==\n' "$file"
    head -n 8 "$OUTPUT_DIR/$file"
  fi
done

if [[ -f "$OUTPUT_DIR/recommendation.json" ]]; then
  printf '\n== recommendation ==\n'
  uv run python - "$OUTPUT_DIR/recommendation.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(json.dumps({
    "winner": payload["winner"],
    "promote_anomaly_layer": payload["promote_anomaly_layer"],
    "status": payload.get("status"),
    "provenance_status": payload.get("provenance_status"),
    "execution_enabled": payload.get("execution_enabled", False),
}, indent=2))
PY
fi
