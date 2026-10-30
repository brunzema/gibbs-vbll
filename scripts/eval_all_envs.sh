#!/usr/bin/env bash

# Evaluate all loss functions across all environments for a given sweep.
#
# Each environment is evaluated in a separate eval.py call (required because
# eval.py reconstructs the environment from the first loaded model and all
# models in a single run must share the same environment).
#
# Usage:
#   bash scripts/eval_all_envs.sh <sweep_name> [seed]
# Examples:
#   bash scripts/eval_all_envs.sh exp_v1
#   bash scripts/eval_all_envs.sh exp_v1 43
#   INCLUDE_DYSTS=1 bash scripts/eval_all_envs.sh exp_v1

if [ -z "$1" ]; then
  echo "Usage: $0 <sweep_name> [seed]"
  echo "Example: $0 exp_v1"
  exit 1
fi

SWEEP_NAME=$1
SEED=${2:-42}

ENVS=(
  spiral
  swimmer
  variable_speed_limit_cycle
  bifurcating_limit_cycle
  glycolytic_oscillator
)

DYSTS_SYSTEMS=(
  lorenz
  rossler
  thomas
  chen
  chua
  dadras
  aizawa
  rucklidge
  sprott_a
  sprott_b
  nose_hoover
  lorenz84
  rabinovich_fabrikant
)

FAILED=()

for ENV in "${ENVS[@]}"; do
  echo "========================================"
  echo "Evaluating: ${ENV}  (sweep=${SWEEP_NAME}, seed=${SEED})"
  echo "========================================"
  if ! python runs/eval.py \
    --config-name "eval_${ENV}" \
    sweep_name="${SWEEP_NAME}" \
    seed="${SEED}"; then
    FAILED+=("${ENV}")
  fi
done

if [ "${INCLUDE_DYSTS:-0}" = "1" ]; then
  for SYS in "${DYSTS_SYSTEMS[@]}"; do
    echo "========================================"
    echo "Evaluating: dysts_${SYS}  (sweep=${SWEEP_NAME}, seed=${SEED})"
    echo "========================================"
    if ! python runs/eval.py \
      --config-name "eval_dysts" \
      sweep_name="${SWEEP_NAME}" \
      seed="${SEED}" \
      dysts_system="${SYS}"; then
      FAILED+=("dysts_${SYS}")
    fi
  done
fi

echo ""
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "Evaluations completed with failures:"
  for F in "${FAILED[@]}"; do
    echo "  - ${F}"
  done
  echo "Partial results saved to: results/eval/${SWEEP_NAME}/"
  exit 1
fi

echo "All evaluations complete. Results saved to: results/eval/${SWEEP_NAME}/"
