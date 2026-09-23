#!/bin/bash
# Run ShoppingBench inference and evaluation on the first N cases of each level.
#
# Defaults: levels 1/2/3, 10 cases per level, deepseek-flash, 2 workers.
# Example:
#   BENCHMARK_MODEL=deepseek-flash SHOPPING_SAMPLE_COUNT=10 bash test_run.sh
# Optional:
#   SHOPPING_LEVELS="1 3" SHOPPING_WORKERS=4 SHOPPING_MAX_LLM_CALLS=50 bash test_run.sh
# Retrieval budget (does not consume extra LLM calls):
#   SHOPPING_L1_REDUNDANCY=5 SHOPPING_MAX_CANDIDATES_PER_SLOT=200 \
#   SHOPPING_MAX_PRODUCT_DETAILS=1000 SHOPPING_MAX_RETRIEVAL_TOOL_CALLS=250 bash test_run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$ROOT/.env" ]; then
    set -a
    source "$ROOT/.env"
    set +a
fi

MODEL="${BENCHMARK_MODEL:-deepseek-flash}"
LEVELS_TEXT="${SHOPPING_LEVELS:-1 2 3}"
SAMPLE_COUNT="${SHOPPING_SAMPLE_COUNT:-10}"
WORKERS="${SHOPPING_WORKERS:-2}"
MAX_LLM_CALLS="${SHOPPING_MAX_LLM_CALLS:-50}"
DATA_DIR="${BENCHMARK_DATA_DIR:-$(cd "$ROOT/../../.." && pwd)/data}"

case "$SAMPLE_COUNT" in
    ''|*[!0-9]*|0) echo "Error: SHOPPING_SAMPLE_COUNT must be a positive integer"; exit 2 ;;
esac

LEVELS=($LEVELS_TEXT)
[ "${#LEVELS[@]}" -gt 0 ] || { echo "Error: no shopping levels selected"; exit 2; }
for LEVEL in "${LEVELS[@]}"; do
    case "$LEVEL" in
        1|2|3) ;;
        *) echo "Error: SHOPPING_LEVELS may contain only 1, 2, or 3"; exit 2 ;;
    esac
done

cd "$ROOT/shoppingplanning"
mkdir -p test_runs result_report
BATCH_DIR="$(mktemp -d "$PWD/test_runs/cyt_test_levels_$(date +%Y%m%d%H%M%S)_XXXXXX")"
BATCH_NAME="$(basename "$BATCH_DIR")"
REPORT_DIRS=()

prepare_level_data() {
    local level="$1"
    local source_database="$2"
    local test_data="$3"
    local database_dir="$4"

    python3 - "$PWD/data/level_${level}_query_meta.json" "$source_database" \
        "$test_data" "$database_dir" "$SAMPLE_COUNT" <<'PY'
import json
import shutil
import sys
from pathlib import Path

query_path, source_database, test_data, database_dir = map(Path, sys.argv[1:5])
sample_count = int(sys.argv[5])

if not query_path.is_file():
    raise FileNotFoundError(query_path)
if not source_database.is_dir():
    raise FileNotFoundError(source_database)
database_dir.mkdir(parents=True, exist_ok=True)

samples = json.loads(query_path.read_text(encoding="utf-8"))[:sample_count]
if len(samples) != sample_count:
    raise ValueError(
        f"{query_path.name} contains {len(samples)} cases; {sample_count} requested"
    )

case_names = [f"case_{sample['id']}" for sample in samples]
if len(set(case_names)) != sample_count:
    raise ValueError(f"{query_path.name} contains duplicate case IDs")
for case_name in case_names:
    source = source_database / case_name
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, database_dir / case_name)

test_data.write_text(
    json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"Selected Level {query_path.stem.split('_')[1]} cases: "
      f"{[sample['id'] for sample in samples]}")
PY
}

export SHOPPING_AGENT_MODEL="$MODEL"
echo "Shopping test batch: $BATCH_DIR"
echo "model=$MODEL, levels=${LEVELS[*]}, samples_per_level=$SAMPLE_COUNT, workers=$WORKERS, max_llm_calls=$MAX_LLM_CALLS"

for LEVEL in "${LEVELS[@]}"; do
    SOURCE_DATABASE="$DATA_DIR/database_level$LEVEL"
    RUN_DIR="$BATCH_DIR/level_$LEVEL"
    DATABASE_DIR="$RUN_DIR/database"
    TEST_DATA="$RUN_DIR/queries.json"
    REPORT_NAME="${BATCH_NAME}_level${LEVEL}"
    mkdir -p "$RUN_DIR"

    prepare_level_data "$LEVEL" "$SOURCE_DATABASE" "$TEST_DATA" "$DATABASE_DIR"

    echo
    echo "================================================================================"
    echo "Running Shopping Level $LEVEL: $SAMPLE_COUNT cases"
    echo "Artifacts: $RUN_DIR"
    echo "================================================================================"

    python3 run.py \
        --model "$MODEL" \
        --level "$LEVEL" \
        --workers "$WORKERS" \
        --max-llm-calls "$MAX_LLM_CALLS" \
        --test-data "$TEST_DATA" \
        --database-dir "$DATABASE_DIR"

    python3 evaluation/evaluation_pipeline.py \
        --database_dir "$DATABASE_DIR" \
        --output_dir "$REPORT_NAME"

    REPORT_DIRS+=("$PWD/result_report/$REPORT_NAME")
    echo "✅ Level $LEVEL completed: $PWD/result_report/$REPORT_NAME"
done

echo
python3 - "${REPORT_DIRS[@]}" <<'PY'
import json
import sys
from pathlib import Path

print("Shopping test summary")
print("level | exact cases | product/coupon matches | incomplete")
print("------|-------------|------------------------|-----------")
for directory in map(Path, sys.argv[1:]):
    report = json.loads((directory / "summary_report.json").read_text(encoding="utf-8"))
    stats = report["overall_statistics"]
    run_level = directory.name.rsplit("level", 1)[-1]
    exact = f"{stats['successful_cases']}/{stats['total_cases']} ({stats['average_case_score']:.2%})"
    matches = f"{stats['total_matched_products']}/{stats['total_expected_products']} ({stats['overall_match_rate']:.2%})"
    print(f"{run_level} | {exact} | {matches} | {stats['incomplete_cases']}")
PY

echo
echo "✅ All requested shopping levels completed."
echo "Artifacts: $BATCH_DIR"
echo "Reports: ${REPORT_DIRS[*]}"
