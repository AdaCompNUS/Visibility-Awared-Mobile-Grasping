#!/usr/bin/env bash
# One-GPU, many-worker matrix runner. Each (protocol, mode, repeat) cell runs
# WORKERS processes in parallel, each owning a slice of the 20 benchmark
# scenes (~1.5 GB VRAM per worker — a single serial env leaves the GPU idle).
# Worker JSONs land in results/matrix_parts/<tag>/ and are merged to
# results/matrix/<tag>.json. Completed cells (merged file with 400 tasks, or a
# legacy full-run log) are skipped, so the script is resumable.
#   bash run_matrix_parallel.sh <gpu> <workers>
set -uo pipefail
WORKERS="${1:-12}"  # spread over GPUs 4-7 (worker w -> GPU 4+w%4)
CKPT="${2:-baseline/VLA/ckpts/sgvla_main/latest}"
cd "$(dirname "$0")/../.."
RES=baseline/VLA/results
mkdir -p "$RES/matrix" "$RES/matrix_parts" "$RES/logs"

ALL_SCENES=(); for i in $(seq 0 19); do ALL_SCENES+=("scene_$i"); done

for r in 0 1 2 3 4; do
  for mode in static dynamic; do
    for proto in easy hard; do
      tag="${proto}_${mode}_r${r}"
      if python3 - "$RES/matrix/${tag}.json" <<'PY'
import json, sys, os
p = sys.argv[1]
ok = os.path.exists(p) and json.load(open(p)).get("summary", {}).get("total") == 400
sys.exit(0 if ok else 1)
PY
      then echo "=== skip $tag (merged complete) ==="; continue; fi
      if grep -q '"total": 400' "$RES/logs/${tag}.log" 2>/dev/null; then
        echo "=== skip $tag (legacy full-run log complete) ==="; continue; fi

      DYNFLAG=""; [[ "$mode" == "dynamic" ]] && DYNFLAG="--dynamic"
      PART="$RES/matrix_parts/$tag"; rm -rf "$PART"; mkdir -p "$PART"
      echo "=== [$(date +%H:%M:%S)] $tag: $WORKERS workers on GPUs 4-7 ==="
      pids=()
      for w in $(seq 0 $((WORKERS-1))); do
        SC=$(printf '%s\n' "${ALL_SCENES[@]}" | awk -v w=$w -v n=$WORKERS 'NR % n == w' | paste -sd,)
        [[ -z "$SC" ]] && continue
        WGPU=$((4 + w % 4))
        CUDA_VISIBLE_DEVICES="$WGPU" python baseline/VLA/eval_vla_on_benchmark.py \
            --gpu "$WGPU" --ckpt "$CKPT" --protocol "$proto" $DYNFLAG --seed-offset "$r" \
            --scenes "$SC" --out "$PART" \
            > "$RES/logs/${tag}.w${w}.log" 2>&1 &
        pids+=($!)
      done
      fail=0
      for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
      python3 - "$PART" "$RES/matrix/${tag}.json" <<'PY'
import json, sys, glob, os
part, out = sys.argv[1], sys.argv[2]
scenes, cfg = {}, None
for f in sorted(glob.glob(os.path.join(part, "*.json"))):
    d = json.load(open(f))
    cfg = cfg or d.get("config")
    scenes.update(d.get("scenes", {}))
tasks = [t for recs in scenes.values() for t in recs]
n = len(tasks); s = sum(t["success"] for t in tasks)
res = {"config": cfg, "scenes": scenes,
       "summary": {"total": n, "success": s,
                   "success_rate": round(s / n, 4) if n else 0.0}}
json.dump(res, open(out, "w"), indent=2)
print(f"merged {out}: {n} tasks, success_rate={res['summary']['success_rate']}")
PY
      [[ $fail -ne 0 ]] && echo "!!! $tag had failed workers (merged what completed)"
    done
  done
done
echo "MATRIX DONE"
