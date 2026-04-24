#!/bin/bash
#
# Continuous Benchmark Runner
# Automatically allocates 4 free GPUs and runs 2 experiments in parallel (2 GPUs each)
#
# Usage: ./scripts/run_continuous_benchmark.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Configuration
GPUS_PER_EXPERIMENT=2
PARALLEL_EXPERIMENTS=2  # Run 2 experiments simultaneously
GPUS_REQUIRED=$((GPUS_PER_EXPERIMENT * PARALLEL_EXPERIMENTS))  # Need 4 GPUs total
CHECK_INTERVAL=30  # seconds to wait before rechecking GPU availability
GPU_MEMORY_THRESHOLD=1000  # MB - consider GPU free if used memory < this

# Config files for different schedulers
CONFIG_NO_VELOCITY="grasp_anywhere/configs/maniskill_fetch_baseline_no_velocity.yaml"
CONFIG_NAV_MANIP="grasp_anywhere/configs/maniskill_fetch_baseline_nav_manip.yaml"
CONFIG_NAV_PREPOSE="grasp_anywhere/configs/maniskill_fetch_nav_prepose.yaml"
CONFIG_CLOSED_LOOP="grasp_anywhere/configs/maniskill_fetch_closed_loop.yaml"
CONFIG_DIRECT_GRASPING="grasp_anywhere/configs/maniskill_fetch_baseline_sequential_scheduler.yaml"

# Experiment queue: (config_file, run_count, extra_args)
declare -a EXPERIMENTS=(
    # "$CONFIG_NO_VELOCITY:2:-p -t"
    # "$CONFIG_NAV_MANIP:1:-p -t"
    # "$CONFIG_DIRECT_GRASPING:1:-p -t"
    # "$CONFIG_CLOSED_LOOP:3:-p -t"
    "$CONFIG_NAV_PREPOSE:2:-p -t"
)

# Function to get free GPUs
get_free_gpus() {
    local free_gpus=()

    # Get GPU memory usage using nvidia-smi
    while IFS= read -r line; do
        local gpu_id=$(echo "$line" | awk -F', ' '{print $1}')
        local used_mem=$(echo "$line" | awk '{print $2}' | sed 's/MiB//')

        if [[ "$used_mem" -lt "$GPU_MEMORY_THRESHOLD" ]]; then
            free_gpus+=("$gpu_id")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

    echo "${free_gpus[@]}"
}

# Function to wait for N free GPUs and return them as an array (one group per line)
wait_for_free_gpus() {
    local required=$1

    while true; do
        local free_gpus=($(get_free_gpus))
        local num_free=${#free_gpus[@]}

        if [[ $num_free -ge $required ]]; then
            # Return GPU groups for parallel experiments
            # Each line contains comma-separated GPUs for one experiment
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

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $required free GPUs... (currently $num_free available)" >&2
        sleep "$CHECK_INTERVAL"
    done
}

# Function to run a single experiment
run_experiment() {
    local config="$1"
    local extra_args="$2"
    local gpus="$3"
    local run_num="$4"
    local total_runs="$5"

    echo "========================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting experiment"
    echo "  Config: $config"
    echo "  Run: $run_num / $total_runs"
    echo "  GPUs: $gpus"
    echo "  Args: $extra_args"
    echo "========================================"

    python experiments/run_maniskill_benchmark.py \
        -c "$config" \
        -g "$gpus" \
        $extra_args

    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment completed successfully"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment failed with exit code $exit_code"
    fi

    return $exit_code
}

# Main loop
main() {
    echo "========================================"
    echo "Continuous Benchmark Runner"
    echo "========================================"
    echo "Experiments to run:"
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r config runs args <<< "$exp"
        echo "  - $config x$runs ($args)"
    done
    echo "========================================"
    echo ""

    local total_experiments=0
    local completed_experiments=0
    local failed_experiments=0

    # Count total experiments
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r config runs args <<< "$exp"
        total_experiments=$((total_experiments + runs))
    done

    echo "Total experiments to run: $total_experiments"
    echo ""

    # Build flat list of all experiments: "config:args:run_num:total_runs"
    declare -a all_experiments=()
    for exp in "${EXPERIMENTS[@]}"; do
        IFS=':' read -r config runs args <<< "$exp"
        for ((run=1; run<=runs; run++)); do
            all_experiments+=("$config:$args:$run:$runs")
        done
    done

    # Process experiments in parallel batches
    local idx=0
    local num_experiments=${#all_experiments[@]}

    while [[ $idx -lt $num_experiments ]]; do
        echo ""
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for $GPUS_REQUIRED free GPUs..."

        # Wait for free GPUs (returns one GPU group per line)
        mapfile -t gpu_groups < <(wait_for_free_gpus "$GPUS_REQUIRED")

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Allocated GPU groups: ${gpu_groups[*]}"

        # Launch parallel experiments
        declare -a pids=()
        declare -a exp_indices=()

        for ((i=0; i<PARALLEL_EXPERIMENTS && idx+i<num_experiments; i++)); do
            local exp_idx=$((idx + i))
            IFS=':' read -r config args run_num total_runs <<< "${all_experiments[$exp_idx]}"

            echo ""
            echo "========================================"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting experiment (parallel slot $i)"
            echo "  Config: $config"
            echo "  Run: $run_num / $total_runs"
            echo "  GPUs: ${gpu_groups[$i]}"
            echo "  Args: $args"
            echo "========================================"

            # Run experiment in background
            (
                python experiments/run_maniskill_benchmark.py \
                    -c "$config" \
                    -g "${gpu_groups[$i]}" \
                    $args
            ) &
            pids+=($!)
            exp_indices+=($exp_idx)
        done

        # Wait for all parallel experiments to complete
        local batch_size=${#pids[@]}
        for ((i=0; i<batch_size; i++)); do
            local exit_code=0
            wait ${pids[$i]} || exit_code=$?
            local exp_idx=${exp_indices[$i]}
            IFS=':' read -r config args run_num total_runs <<< "${all_experiments[$exp_idx]}"

            if [[ $exit_code -eq 0 ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment completed: $config (run $run_num)"
                completed_experiments=$((completed_experiments + 1))
            else
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Experiment failed (exit $exit_code): $config (run $run_num)"
                failed_experiments=$((failed_experiments + 1))
            fi
        done

        idx=$((idx + batch_size))

        echo ""
        echo "Progress: $((completed_experiments + failed_experiments))/$total_experiments (completed: $completed_experiments, failed: $failed_experiments)"
    done

    echo ""
    echo "========================================"
    echo "All experiments completed!"
    echo "  Total: $total_experiments"
    echo "  Completed: $completed_experiments"
    echo "  Failed: $failed_experiments"
    echo "========================================"
}

# Run main function
main "$@"
