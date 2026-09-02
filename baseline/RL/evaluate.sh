#!/usr/bin/env bash
# Validate released MS-HAB checkpoints in their own environment/protocol:
# mshab.evaluate with the upstream configs/evaluate.yml (SequentialTask-v0,
# 252 envs, 1000 trajectories, success_once). Only plumbing is parameterized.
#   POLICY_TYPE: rl_all_obj (default; matches our retraining target)
#                or rl_per_obj (paper's headline RL-Per numbers)
#   SPLIT: train (default) or val
set -euo pipefail
SEED="${SEED:-0}"
GPU="${GPU:-2}"
POLICY_TYPE="${POLICY_TYPE:-rl_all_obj}"
SPLIT="${SPLIT:-train}"
TASK=tidy_house
SUBTASK=pick

TASK_PLAN_FP="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/all.json"

CUDA_VISIBLE_DEVICES="$GPU" SAPIEN_NO_DISPLAY=1 python -m mshab.evaluate configs/evaluate.yml \
        seed="$SEED" \
        task="$TASK" \
        policy_type="$POLICY_TYPE" \
        eval_env.task_plan_fp="$TASK_PLAN_FP" \
        logger.workspace=mshab_exps \
        logger.exp_name="eval/$TASK-$SUBTASK-$SPLIT-$POLICY_TYPE-seed$SEED"
