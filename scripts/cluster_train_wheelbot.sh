#!/usr/bin/env bash
set -euo pipefail

# Train all losses on the Wheelbot racing data on a Slurm cluster, one GPU per
# run. Launch from an environment with GPU support (uv sync --extra cuda).
# Afterwards compute the metrics for the results table with
#   uv run python runs/eval_wheelbot.py <wb_sweep_name>
#
# Usage:
#   bash scripts/cluster_train_wheelbot.sh <wb_sweep_name>
#
# Optional overrides: SEEDS (default 42), SLURM_ACCOUNT (omitted when unset),
# PARTITION (GPU partition; omitted when empty).

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <wb_sweep_name>"
  exit 1
fi

SWEEP_NAME="$1"
PARTITION_OVERRIDE=()
if [ -n "${PARTITION:-}" ]; then
  PARTITION_OVERRIDE=(+hydra.launcher.partition="$PARTITION")
fi

python runs/wheelbot_dynamics.py \
  'loss=single_step_elbo_hbatch,multi_step_elbo,multi_step_kde,multi_step_factored_crps,multi_step_ar_crps,multi_step_ar_elbo' \
  "seed=${SEEDS:-42}" \
  path=cluster \
  sweep_name="$SWEEP_NAME" \
  +launcher=slurm \
  hydra.launcher.gpus_per_node=1 \
  "${PARTITION_OVERRIDE[@]}" \
  --multirun
