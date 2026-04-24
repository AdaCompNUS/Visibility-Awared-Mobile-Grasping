#!/bin/bash
#
# Run All Benchmarks
#
# Runs every scheduler method sequentially, each with parallel scene evaluation.
# Results are saved under results/<method_name>/.
#
# Usage:
#   ./scripts/run_all_benchmarks.sh [gpus] [repeats] [extra_args]
#
# Examples:
#   ./scripts/run_all_benchmarks.sh                    # GPUs 0,1; 1 repeat each
#   ./scripts/run_all_benchmarks.sh 0,1,2,3 3          # 4 GPUs; 3 repeats each
#   ./scripts/run_all_benchmarks.sh 0,1 1 "-t"         # save trajectories
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ---------- arguments ----------
GPUS="${1:-0,1}"
REPEATS="${2:-1}"
EXTRA_ARGS="${3:-}"

# ---------- methods ----------
# Each entry: "label:config_path"
METHODS=(
    "ours:grasp_anywhere/configs/maniskill_fetch.yaml"
    "sequential:grasp_anywhere/configs/maniskill_fetch_baseline_sequential_scheduler.yaml"
    "nav_manip:grasp_anywhere/configs/maniskill_fetch_baseline_nav_manip.yaml"
    "nav_prepose:grasp_anywhere/configs/maniskill_fetch_nav_prepose.yaml"
    "closed_loop:grasp_anywhere/configs/maniskill_fetch_closed_loop.yaml"
    "no_velocity:grasp_anywhere/configs/maniskill_fetch_baseline_no_velocity.yaml"
)

# ---------- summary counters ----------
TOTAL=0
PASSED=0
FAILED=0
FAILED_LIST=()

NUM_METHODS=${#METHODS[@]}
TOTAL_RUNS=$((NUM_METHODS * REPEATS))

echo "========================================"
echo "  Run All Benchmarks"
echo "========================================"
echo "  GPUs:        $GPUS"
echo "  Repeats:     $REPEATS"
echo "  Extra args:  ${EXTRA_ARGS:-(none)}"
echo "  Methods:     $NUM_METHODS"
echo "  Total runs:  $TOTAL_RUNS"
echo "----------------------------------------"
for entry in "${METHODS[@]}"; do
    IFS=':' read -r label config <<< "$entry"
    echo "  - $label  ($config)"
done
echo "========================================"
echo ""

for entry in "${METHODS[@]}"; do
    IFS=':' read -r label config <<< "$entry"

    for ((run=1; run<=REPEATS; run++)); do
        TOTAL=$((TOTAL + 1))
        echo ""
        echo "========================================"
        echo "[$TOTAL/$TOTAL_RUNS] $label  (run $run/$REPEATS)"
        echo "  Config: $config"
        echo "  GPUs:   $GPUS"
        echo "  Start:  $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================"

        EXIT_CODE=0
        python experiments/run_maniskill_benchmark.py \
            -c "$config" \
            -g "$GPUS" \
            -p \
            $EXTRA_ARGS \
        || EXIT_CODE=$?

        if [[ $EXIT_CODE -eq 0 ]]; then
            PASSED=$((PASSED + 1))
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label run $run PASSED"
        else
            FAILED=$((FAILED + 1))
            FAILED_LIST+=("$label run $run (exit $EXIT_CODE)")
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] $label run $run FAILED (exit $EXIT_CODE)"
        fi
    done
done

echo ""
echo "========================================"
echo "  All Benchmarks Finished"
echo "========================================"
echo "  Total:     $TOTAL_RUNS"
echo "  Passed:    $PASSED"
echo "  Failed:    $FAILED"
if [[ ${#FAILED_LIST[@]} -gt 0 ]]; then
    echo "  Failures:"
    for f in "${FAILED_LIST[@]}"; do
        echo "    - $f"
    done
fi
echo "========================================"
