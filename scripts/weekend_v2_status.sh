#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/weekend_v2_search}"

printf 'Output: %s\n' "$OUTPUT_DIR"
for stage in screen refine confirmation; do
  stage_dir="$OUTPUT_DIR/$stage"
  if [[ -d "$stage_dir" ]]; then
    complete="$(find "$stage_dir" -type f -name result.json 2>/dev/null | wc -l | tr -d ' ')"
    failed="$(find "$stage_dir" -type f -name failure.json 2>/dev/null | wc -l | tr -d ' ')"
  else
    complete=0
    failed=0
  fi
  printf '%-13s complete=%-3s failed=%s\n' "$stage" "$complete" "$failed"
done

for file in screen_leaderboard.csv refine_leaderboard.csv confirmation_leaderboard.csv ensemble_leaderboard.csv; do
  if [[ -s "$OUTPUT_DIR/$file" ]]; then
    printf '\n== %s ==\n' "$file"
    head -n 10 "$OUTPUT_DIR/$file"
  fi
done

for selection_file in refine_selection.json confirmation_selection.json; do
  if [[ -f "$OUTPUT_DIR/$selection_file" ]]; then
    printf '\n== %s ==\n' "$selection_file"
    uv run python - "$OUTPUT_DIR/$selection_file" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
print(json.dumps({"method": payload.get("method"), "selected": payload.get("selected")}, indent=2))
PY
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
    "promote_weekend_v2": payload["promote_weekend_v2"],
    "status": payload.get("status"),
    "execution_enabled": payload.get("execution_enabled", False),
    "final_submission_command": payload.get("final_submission_command"),
}, indent=2))
PY
fi
