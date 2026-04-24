#!/bin/bash
#
# Experiment: Dynamic Obstacle Trigger Distance Analysis
#
# Analyzes system performance under different initial distances of
# suddenly appearing dynamic obstacles. Runs our method (default scheduler)
# with varying nav_trigger_distance values, 4 repetitions each.
#
# Usage:
#   ./scripts/run_trigger_distance_experiment.sh [gpus]
#
# Examples:
#   ./scripts/run_trigger_distance_experiment.sh          # uses GPUs 0,1,2,3
#   ./scripts/run_trigger_distance_experiment.sh 2,3,4,5  # specify GPUs
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# --- Configuration ---
BASE_CONFIG="grasp_anywhere/configs/maniskill_fetch.yaml"
GPUS="${1:-0,1,2,3}"
RUNS_PER_DISTANCE=4
PARALLEL_EXPERIMENTS=2
GPUS_PER_EXPERIMENT=2
GPUS_REQUIRED=$((GPUS_PER_EXPERIMENT * PARALLEL_EXPERIMENTS))
GPU_MEMORY_THRESHOLD=1000  # MB
CHECK_INTERVAL=30

# Trigger distances to test (meters)
DISTANCES=(1.0 1.5 2.0 2.5)

# Directory for generated configs
CONFIG_DIR="$PROJECT_DIR/grasp_anywhere/configs"

# --- Helper Functions ---

generate_config() {
    local distance="$1"
    local output_path="$2"

    # Copy base config and append nav_trigger_distance under benchmark section
    python3 -c "
import yaml, sys

with open('$BASE_CONFIG', 'r') as f:
    config = yaml.safe_load(f)

config.setdefault('benchmark', {})['nav_trigger_distance'] = $distance
config['benchmark']['enable_dynamic_challenges'] = True

with open('$output_path', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
"
    echo "  Generated config: $output_path (trigger_distance=$distance)"
}

get_free_gpus() {
    local free_gpus=()
    while IFS= read -r line; do
        local gpu_id=$(echo "$line" | awk -F', ' '{print $1}')
        local used_mem=$(echo "$line" | awk '{print $2}' | sed 's/MiB//')
        if [[ "$used_mem" -lt "$GPU_MEMORY_THRESHOLD" ]]; then
            free_gpus+=("$gpu_id")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    echo "${free_gpus[@]}"
}

wait_for_free_gpus() {
    local required=$1
    while true; do
        local free_gpus=($(get_free_gpus))
        local num_free=${#free_gpus[@]}
        if [[ $num_free -ge $required ]]; then
            for ((exp=0; exp<PARALLEL_EXPERIMENTS; exp++)); do
                local start=$((exp * GPUS_PER_EXPERIMENT))
                local group=""
                for ((i=0; i<GPUS_PER_EXPERIMENT; i++)); do
                    local idx=$((start + i))
                    if [[ -n "$group" ]]; then
                        group="$group,${free_gpus[$idx]}"
                    else
                        group="${free_gpus[$idx]}"
                    fi
                done
                echo "$group"
            done
            return 0
        fi
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $required free GPUs... ($num_free available)" >&2
        sleep "$CHECK_INTERVAL"
    done
}

# --- Main ---

echo "========================================"
echo "Trigger Distance Experiment"
echo "========================================"
echo "  Method: ours (default scheduler)"
echo "  Base config: $BASE_CONFIG"
echo "  Distances: ${DISTANCES[*]}"
echo "  Runs per distance: $RUNS_PER_DISTANCE"
echo "  GPUs: $GPUS"
echo "========================================"
echo ""

# Step 1: Generate configs for each distance
echo "Generating configs..."
for dist in "${DISTANCES[@]}"; do
    config_path="$CONFIG_DIR/maniskill_fetch_trigger_${dist}.yaml"
    generate_config "$dist" "$config_path"
done
echo ""

# Step 2: Build flat experiment list interleaved by distance
# Order: [d0 run1, d1 run1, d2 run1, ..., d0 run2, d1 run2, ...]
# This ensures parallel batches always test DIFFERENT distances.
declare -a all_experiments=()
for ((run=1; run<=RUNS_PER_DISTANCE; run++)); do
    for dist in "${DISTANCES[@]}"; do
        config_path="$CONFIG_DIR/maniskill_fetch_trigger_${dist}.yaml"
        all_experiments+=("$config_path:$dist:$run")
    done
done

total_experiments=${#all_experiments[@]}
completed_experiments=0
failed_experiments=0

echo "Total experiments: $total_experiments (${#DISTANCES[@]} distances x $RUNS_PER_DISTANCE runs)"
echo ""

# Step 3: Run experiments in parallel batches
idx=0
while [[ $idx -lt $total_experiments ]]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $GPUS_REQUIRED free GPUs..."
    mapfile -t gpu_groups < <(wait_for_free_gpus "$GPUS_REQUIRED")
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Allocated GPU groups: ${gpu_groups[*]}"

    declare -a pids=()
    declare -a exp_indices=()

    for ((i=0; i<PARALLEL_EXPERIMENTS && idx+i<total_experiments; i++)); do
        exp_idx=$((idx + i))
        IFS=':' read -r config_path dist run_num <<< "${all_experiments[$exp_idx]}"

        echo ""
        echo "========================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching: distance=${dist}m, run=${run_num}/${RUNS_PER_DISTANCE}"
        echo "  Config: $config_path"
        echo "  GPUs: ${gpu_groups[$i]}"
        echo "========================================"

        (
            python experiments/run_maniskill_benchmark.py \
                -c "$config_path" \
                -g "${gpu_groups[$i]}" \
                -p -t
        ) &
        pids+=($!)
        exp_indices+=($exp_idx)
    done

    # Wait for batch to complete
    batch_size=${#pids[@]}
    for ((i=0; i<batch_size; i++)); do
        exit_code=0
        wait ${pids[$i]} || exit_code=$?
        exp_idx=${exp_indices[$i]}
        IFS=':' read -r config_path dist run_num <<< "${all_experiments[$exp_idx]}"

        if [[ $exit_code -eq 0 ]]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: distance=${dist}m, run=${run_num} - SUCCESS"
            completed_experiments=$((completed_experiments + 1))
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: distance=${dist}m, run=${run_num} - FAILED (exit $exit_code)"
            failed_experiments=$((failed_experiments + 1))
        fi
    done

    idx=$((idx + batch_size))
    echo ""
    echo "Progress: $((completed_experiments + failed_experiments))/$total_experiments (ok: $completed_experiments, fail: $failed_experiments)"
done

echo ""
echo "========================================"
echo "Experiment Complete"
echo "  Total: $total_experiments"
echo "  Completed: $completed_experiments"
echo "  Failed: $failed_experiments"
echo ""
echo "Results saved under results/ours_dyn/ with separate run folders."
echo "Each config used trigger distances: ${DISTANCES[*]}"
echo "========================================"
