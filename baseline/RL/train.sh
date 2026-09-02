#!/usr/bin/env bash
# Launcher for the MS-HAB SAC Pick baseline. Mirrors upstream
# scripts/train_sac.sh argument-for-argument; only run plumbing is
# parameterized (seed, GPU, per-seed output dir, asset paths).
#
#   SEED=1 GPU=2 bash train.sh
#
# If the released checkpoint config is present (mshab_checkpoints/rl/...),
# it is used verbatim for hyperparameters, per the MS-HAB README: "the released
# checkpoints may use different hyperparameters. To train using the same args,
# check the config.yml files from the released checkpoints."
set -euo pipefail

SEED="${SEED:-0}"
GPU="${GPU:-1}"

TASK=tidy_house
SUBTASK=pick
SPLIT=train
OBJ=all

ENV_ID="PickSubtaskTrain-v0"
WORKSPACE="mshab_exps"
GROUP="$TASK-rcad-sac-$SUBTASK"
EXP_NAME="$ENV_ID/$GROUP/sac-$SUBTASK-$OBJ-seed$SEED"

TASK_PLAN_FP="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/task_plans/$TASK/$SUBTASK/$SPLIT/$OBJ.json"
SPAWN_DATA_FP="$MS_ASSET_DIR/data/scene_datasets/replica_cad_dataset/rearrange/spawn_data/$TASK/$SUBTASK/$SPLIT/spawn_data.pt"

CKPT_CFG="mshab_checkpoints/rl/$TASK/$SUBTASK/$OBJ/config.yml"
if [[ -f "$CKPT_CFG" ]]; then
    # The released config embeds `model_ckpt:` pointing at its own final
    # policy.pt, and mshab's CLI cannot express "no checkpoint" (parse_cfg maps
    # any `x=null` to True). Strip that line into a derived config so the run
    # trains from scratch with otherwise identical hyperparameters.
    BASE_CFG="configs/generated_scratch_seed$SEED.yml"
    sed '/^model_ckpt:/d' "$CKPT_CFG" > "$BASE_CFG"
    echo ">>> from-scratch run with released-checkpoint hyperparameters ($CKPT_CFG minus model_ckpt)"
else
    BASE_CFG="configs/sac_pick.yml"
    echo ">>> checkpoint config not found; using upstream configs/sac_pick.yml"
fi

# Overrides: paths + seed + run naming (plumbing), and the paper's Pick budget
# (50M steps; the upstream shell script leaves an open-ended 1B).
CUDA_VISIBLE_DEVICES="$GPU" SAPIEN_NO_DISPLAY=1 python -m mshab.train_sac "$BASE_CFG" \
        logger.clear_out="True" \
        logger.wandb="False" \
        logger.tensorboard="True" \
        logger.wandb_cfg.group="$GROUP" \
        logger.exp_name="$EXP_NAME" \
        logger.workspace="$WORKSPACE" \
        logger.best_stats_cfg="{eval/success_once: 1, eval/return_per_step: 1}" \
        seed="$SEED" \
        env.env_id="$ENV_ID" \
        env.task_plan_fp="$TASK_PLAN_FP" \
        env.spawn_data_fp="$SPAWN_DATA_FP" \
        eval_env.task_plan_fp="$TASK_PLAN_FP" \
        eval_env.spawn_data_fp="$SPAWN_DATA_FP" \
        algo.total_timesteps=50_000_000
