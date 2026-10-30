#!/usr/bin/env bash
set -euo pipefail

# Run one unified sweep name across:
# 1) all non-dysts experiment configs
# 2) dysts experiment over all configured systems
#
# Model is always pinned to vbll_delta.
#
# Usage:
#   bash scripts/cluster_train_all_envs.sh <sweep_name>
# Example:
#   bash scripts/cluster_train_all_envs.sh full_envs_v1
#
# Optional overrides: EXPERIMENT (default paper), SEEDS (default 42),
# e.g. SEEDS=42,43,44 EXPERIMENT=paper bash ...
# The Slurm account is read from SLURM_ACCOUNT (omitted when unset).

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <sweep_name>"
  echo "Example: $0 full_envs_v1"
  exit 1
fi

SWEEP_NAME="$1"

TRAINING_MODES='training=single_step_elbo,single_step_elbo_hbatch,multi_step_elbo,multi_step_kde,multi_step_factored_crps,multi_step_ar_crps,multi_step_ar_elbo'
EXPERIMENT_PROFILE="+experiment=${EXPERIMENT:-paper}"
SEEDS="seed=${SEEDS:-42}"
NON_DYSTS_ENVIRONMENTS='environment=spiral,swimmer,variable_speed_limit_cycle,bifurcating_limit_cycle,glycolytic_oscillator'
DYSTS_SYSTEMS='dysts_system=lorenz,rossler,thomas,chen,chua,dadras,aizawa,rucklidge,sprott_a,sprott_b,nose_hoover,lorenz84,rabinovich_fabrikant'

run_non_dysts() {
  python runs/train.py \
    "$EXPERIMENT_PROFILE" \
    "$NON_DYSTS_ENVIRONMENTS" \
    "$TRAINING_MODES" \
    "$SEEDS" \
    model=vbll_delta \
    logger=csv_logger \
    path=cluster \
    sweep_name="$SWEEP_NAME" \
    +launcher=slurm \
    --multirun
}

run_dysts() {
  python runs/train.py \
    "$EXPERIMENT_PROFILE" \
    environment=dysts \
    "$TRAINING_MODES" \
    "$SEEDS" \
    "$DYSTS_SYSTEMS" \
    model=vbll_delta \
    logger=csv_logger \
    path=cluster \
    sweep_name="$SWEEP_NAME" \
    +launcher=slurm \
    --multirun
}

# Hydra's Submitit launcher waits for its Slurm jobs before returning. Start
# both independent sweeps in the background so both arrays are submitted by a
# single invocation of this script, then keep the script alive until both
# launcher processes finish.
echo "Starting non-dysts and dysts Slurm sweeps: $SWEEP_NAME"
run_non_dysts &
NON_DYSTS_PID=$!
run_dysts &
DYSTS_PID=$!

STATUS=0
if ! wait "$NON_DYSTS_PID"; then
  echo "Non-dysts sweep failed." >&2
  STATUS=1
fi
if ! wait "$DYSTS_PID"; then
  echo "Dysts sweep failed." >&2
  STATUS=1
fi

if [ "$STATUS" -eq 0 ]; then
  echo "Both Slurm sweeps completed successfully: $SWEEP_NAME"
fi

exit "$STATUS"
