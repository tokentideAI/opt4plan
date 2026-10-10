#!/bin/bash
# Run one TravelPlan ReAct case through inference, conversion, and evaluation.
#
# Usage:
#   bash run_travelplan_case.sh [case_id] [additional run.py arguments]
#
# Examples:
#   bash run_travelplan_case.sh 0
#   TRAVEL_LANGUAGE=en bash run_travelplan_case.sh 5
#   BENCHMARK_MODEL=deepseek-flash TRAVEL_MAX_LLM_CALLS=100 \
#     bash run_travelplan_case.sh 0 --debug
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load API keys from the project-local .env when present. Existing exported
# variables take precedence over values loaded by the Python agent.
if [ -f "$ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$ROOT/.env"
    set +a
fi

CASE_ID="${1:-${TRAVEL_CASE_ID:-0}}"
if [ "$#" -gt 0 ]; then
    shift
fi

MODEL="${BENCHMARK_MODEL:-${TRAVEL_AGENT_MODEL:-deepseek-flash}}"
LANGUAGE="${TRAVEL_LANGUAGE:-zh}"
WORKERS="${TRAVEL_WORKERS:-1}"
MAX_LLM_CALLS="${TRAVEL_MAX_LLM_CALLS:-80}"
DATA_DIR="${BENCHMARK_DATA_DIR:-$ROOT/data}"
OUTPUT_DIR="${TRAVEL_OUTPUT_DIR:-$ROOT/travelplanning/results/react_smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$CASE_ID" in
    ''|*[!0-9]*) echo "Error: case_id must be a non-negative integer"; exit 2 ;;
esac

case "$LANGUAGE" in
    zh|en) ;;
    *) echo "Error: TRAVEL_LANGUAGE must be 'zh' or 'en'"; exit 2 ;;
esac

case "$WORKERS" in
    ''|*[!0-9]*|0) echo "Error: TRAVEL_WORKERS must be a positive integer"; exit 2 ;;
esac

case "$MAX_LLM_CALLS" in
    ''|*[!0-9]*|0) echo "Error: TRAVEL_MAX_LLM_CALLS must be a positive integer"; exit 2 ;;
esac

QUERY_FILE="$ROOT/travelplanning/data/travelplanning_query_${LANGUAGE}.json"
CASE_DATABASE="$DATA_DIR/database_${LANGUAGE}/id_${CASE_ID}"

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "Error: Python executable not found: $PYTHON_BIN"
    exit 1
}
[ -f "$QUERY_FILE" ] || { echo "Error: query file not found: $QUERY_FILE"; exit 1; }
[ -d "$CASE_DATABASE" ] || { echo "Error: case database not found: $CASE_DATABASE"; exit 1; }
[ -f "$ROOT/models_config.json" ] || { echo "Error: models_config.json not found"; exit 1; }

echo "Running TravelPlan ReAct case"
echo "  case:          $CASE_ID"
echo "  model:         $MODEL"
echo "  language:      $LANGUAGE"
echo "  workers:       $WORKERS"
echo "  max LLM calls: $MAX_LLM_CALLS"
echo "  data:          $DATA_DIR"
echo "  output:        $OUTPUT_DIR/${MODEL}_${LANGUAGE}"

cd "$ROOT"
export BENCHMARK_DATA_DIR="$DATA_DIR"

exec "$PYTHON_BIN" travelplanning/run.py \
    --model "$MODEL" \
    --language "$LANGUAGE" \
    --workers "$WORKERS" \
    --max-llm-calls "$MAX_LLM_CALLS" \
    --start-from inference \
    --rerun-ids "$CASE_ID" \
    --output-dir "$OUTPUT_DIR" \
    --verbose \
    "$@"
