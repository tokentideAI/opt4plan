#!/bin/bash
# Run one Shopping benchmark case.
# Usage: bash cyt_test_run.sh [case_id]
# Example: BENCHMARK_MODEL=deepseek-flash SHOPPING_LEVEL=2 bash cyt_test_run.sh 7
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ ! -f "$ROOT/.env" ] || { set -a; source "$ROOT/.env"; set +a; }

CASE_ID="${1:-${SHOPPING_CASE_ID:-1}}"
LEVEL="${SHOPPING_LEVEL:-1}"
MODEL="${BENCHMARK_MODEL:-deepseek-flash}"
MAX_CALLS="${SHOPPING_MAX_LLM_CALLS:-50}"
DATA_ROOT="${BENCHMARK_DATA_DIR:-$(cd "$ROOT/../../.." && pwd)/data}"
SOURCE_DB="$DATA_ROOT/database_level$LEVEL"

case "$LEVEL" in 1|2|3) ;; *) echo "Error: SHOPPING_LEVEL must be 1, 2, or 3"; exit 2;; esac
[ -d "$SOURCE_DB/case_$CASE_ID" ] || {
    echo "Error: case database not found: $SOURCE_DB/case_$CASE_ID"
    exit 1
}

cd "$ROOT/shoppingplanning"
mkdir -p test_runs
RUN_DIR="$(mktemp -d "$PWD/test_runs/level${LEVEL}_case${CASE_ID}_$(date +%Y%m%d%H%M%S)_XXXXXX")"
CASE_DB="$RUN_DIR/database/case_$CASE_ID"
QUERY_FILE="$RUN_DIR/query.json"
mkdir -p "$(dirname "$CASE_DB")"
cp -R "$SOURCE_DB/case_$CASE_ID" "$CASE_DB"

# Extract exactly this case from the level's query file.
python - "$PWD/data/level_${LEVEL}_query_meta.json" "$CASE_ID" "$QUERY_FILE" <<'PY'
import json
import sys
from pathlib import Path

source, case_id, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
matches = [x for x in json.loads(source.read_text(encoding="utf-8"))
           if str(x.get("id")) == case_id]
if len(matches) != 1:
    raise SystemExit(f"Error: expected one query for case {case_id}, found {len(matches)}")
output.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "Running Shopping Level $LEVEL, case $CASE_ID with $MODEL"
echo "Artifacts: $RUN_DIR"

python run.py \
    --model "$MODEL" \
    --level "$LEVEL" \
    --workers 1 \
    --max-llm-calls "$MAX_CALLS" \
    --test-data "$QUERY_FILE" \
    --database-dir "$RUN_DIR/database"

python evaluation/evaluation_pipeline.py \
    --database_dir "$RUN_DIR/database" \
    --output_dir "$(basename "$RUN_DIR")" \
    --case_filter "case_$CASE_ID"

echo "Done. Evaluation: $PWD/result_report/$(basename "$RUN_DIR")"
