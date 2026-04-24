#!/bin/bash
#
# Run a single benchmark with specified GPUs
#
# Usage:
#   ./scripts/run_benchmark.sh <scheduler_type> <gpus> [extra_args]
#
# Examples:
#   ./scripts/run_benchmark.sh nav_manip 0,1 "-p -t"
#   ./scripts/run_benchmark.sh closed_loop 2,3 "-p -t"
#   ./scripts/run_benchmark.sh nav_prepose 4,5 "-p -t -n 20"
#
# Scheduler types:
#   nav_manip   - NavManipScheduler (decoupled nav + manipulation)
#   nav_prepose - NavPreposeScheduler (prepose sampler + decoupled)
#   closed_loop - ClosedLoopScheduler (minimal prepose->grasp loop)
#   default     - Main scheduler (uses maniskill_fetch.yaml)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
SCHEDULER_TYPE="${1:-nav_manip}"
GPUS="${2:-0,1}"
EXTRA_ARGS="${3:--p -t}"

# Map scheduler type to config file
case "$SCHEDULER_TYPE" in
    nav_manip)
        CONFIG="grasp_anywhere/configs/maniskill_fetch_baseline_nav_manip.yaml"
        ;;
    nav_prepose)
        CONFIG="grasp_anywhere/configs/maniskill_fetch_nav_prepose.yaml"
        ;;
    closed_loop)
        CONFIG="grasp_anywhere/configs/maniskill_fetch_closed_loop.yaml"
        ;;
    default|ours)
        CONFIG="grasp_anywhere/configs/maniskill_fetch.yaml"
        ;;
    sequential)
        CONFIG="grasp_anywhere/configs/maniskill_fetch_baseline_sequential_scheduler.yaml"
        ;;
    no_velocity)
        CONFIG="grasp_anywhere/configs/maniskill_fetch_baseline_no_velocity.yaml"
        ;;
    *)
        echo "Unknown scheduler type: $SCHEDULER_TYPE"
        echo "Available types: nav_manip, nav_prepose, closed_loop, default, sequential, no_velocity"
        exit 1
        ;;
esac

echo "========================================"
echo "Running ManiSkill Benchmark"
echo "  Scheduler: $SCHEDULER_TYPE"
echo "  Config: $CONFIG"
echo "  GPUs: $GPUS"
echo "  Extra args: $EXTRA_ARGS"
echo "========================================"

python experiments/run_maniskill_benchmark.py \
    -c "$CONFIG" \
    -g "$GPUS" \
    $EXTRA_ARGS

echo "========================================"
echo "Benchmark completed"
echo "========================================"
