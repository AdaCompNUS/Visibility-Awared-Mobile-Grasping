#!/usr/bin/env bash
# Launch TidyHouse-Pick dataset generation for all 9 objects on GPUs 4-7.
# Two waves (5 jobs, then 4) keep total host RAM ~45 GB (< the 60 GB datagen
# budget); each job is ~9 GB RSS at 64 envs.
# Usage: from the repo root:  bash baseline/VLA/datagen/run_all.sh [NUM_ENVS]
# Logs: baseline/VLA/datagen/logs/<object>.log
set -u
cd "$(dirname "$0")/../../.."   # repo root
export PATH="$HOME/.pixi/bin:$PATH"

NUM_ENVS="${1:-64}"
LOGDIR=baseline/VLA/datagen/logs
mkdir -p "$LOGDIR"

run_wave () {
  # args: obj:gpu pairs
  local pids=()
  for spec in "$@"; do
    obj="${spec%%:*}"; gpu="${spec##*:}"
    echo "$(date +%T) launching $obj on GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" nohup pixi run -e rl python \
      baseline/VLA/datagen/gen_pick_dataset.py "$obj" --num-envs "$NUM_ENVS" \
      > "$LOGDIR/$obj.log" 2>&1 &
    pids+=($!)
    sleep 5   # stagger sapien/vulkan init
  done
  wait "${pids[@]}"
}

echo "=== wave 1 (5 jobs) ==="
run_wave 002_master_chef_can:4 003_cracker_box:5 004_sugar_box:6 \
         005_tomato_soup_can:7 007_tuna_fish_can:4

echo "=== wave 2 (4 jobs) ==="
run_wave 008_pudding_box:4 009_gelatin_box:5 010_potted_meat_can:6 024_bowl:7

echo "all generation jobs finished"
