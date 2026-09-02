#!/usr/bin/env bash
# Full RL-baseline result matrix: {easy,hard} x {static,dynamic} x 5 repeats,
# each over the complete 400-task benchmark. One GPU runs a queue sequentially:
#   bash run_matrix.sh <gpu> <mode>     # mode: static | dynamic
# Repeats use --seed-offset=r (varies the easy-protocol spawn sampling; hard
# starts are fixed by the benchmark, so hard repeats vary only through
# simulation nondeterminism).
set -euo pipefail
GPU="$1"; MODE="$2"
DYNFLAG=""; [[ "$MODE" == "dynamic" ]] && DYNFLAG="--dynamic"
cd "$(dirname "$0")/../.."
mkdir -p baseline/RL/results/logs
for r in 0 1 2 3 4; do
  for proto in easy hard; do
    tag="${proto}_${MODE}_r${r}"
    if grep -q '"total": 400' "baseline/RL/results/logs/${tag}.log" 2>/dev/null; then
      echo "=== skipping $tag (already complete) ==="
      continue
    fi
    echo "=== [$(date +%H:%M:%S)] GPU$GPU starting $tag ==="
    CUDA_VISIBLE_DEVICES="$GPU" python baseline/RL/eval_rl_on_benchmark.py \
        --gpu "$GPU" --protocol "$proto" $DYNFLAG --seed-offset "$r" \
        > "baseline/RL/results/logs/${tag}.log" 2>&1
    echo "=== [$(date +%H:%M:%S)] GPU$GPU finished $tag ==="
  done
done
echo "QUEUE DONE ($MODE)"
