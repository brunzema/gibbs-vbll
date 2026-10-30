#!/usr/bin/env bash
set -euo pipefail

# Evaluate all environments for a sweep on a Slurm cluster in parallel.
#
# This is the cluster-parallel counterpart of scripts/eval_all_envs.sh. Run it
# from the repository root on the login node; it submits one Slurm array task per
# environment. Each task runs one runs/eval.py invocation, because eval.py loads
# models for a single environment at a time.
#
# Usage:
#   bash scripts/cluster_eval_all_envs.sh <sweep_name> [seed]
#
# Examples:
#   bash scripts/cluster_eval_all_envs.sh exp_v1
#   bash scripts/cluster_eval_all_envs.sh exp_v1 43
#   INCLUDE_DYSTS=1 bash scripts/cluster_eval_all_envs.sh exp_v1 42
#
# Optional environment overrides:
#   MAX_PARALLEL=5              maximum simultaneous array tasks
#   TIME=08:00:00               Slurm wall time
#   CPUS_PER_TASK=8             CPU cores per task
#   MEM=32G                     memory per task
#   SLURM_ACCOUNT=<account>     Slurm account; omitted when empty
#   PARTITION=gpu               Slurm partition; omitted when empty
#   VENV_PATH=/path/to/.venv    virtualenv to activate; defaults to .venv
#   RESULTS_ROOT=/path/results  root that contains <sweep_name>/vbll_delta/...
#   INCLUDE_DYSTS=1             also evaluate all DYSTS systems from eval_all_envs.sh
#   EVALUATION=paper            evaluation profile (config/evaluation/); default from eval_*.yaml

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

usage() {
  echo "Usage: $0 <sweep_name> [seed]"
  echo "Example: $0 exp_v1"
}

build_tasks() {
  TASK_CONFIGS=()
  TASK_NAMES=()
  TASK_DYSTS_SYSTEMS=()

  for env in "${ENVS[@]}"; do
    TASK_CONFIGS+=("eval_${env}")
    TASK_NAMES+=("${env}")
    TASK_DYSTS_SYSTEMS+=("")
  done

  if [ "${INCLUDE_DYSTS:-0}" = "1" ]; then
    for sys in "${DYSTS_SYSTEMS[@]}"; do
      TASK_CONFIGS+=("eval_dysts")
      TASK_NAMES+=("dysts_${sys}")
      TASK_DYSTS_SYSTEMS+=("${sys}")
    done
  fi
}

default_results_root() {
  local base_path
  base_path="$(awk -F': *' '$1 == "base_path" {print $2; exit}' config/path/cluster.yaml 2>/dev/null || true)"
  base_path="${base_path%%#*}"
  base_path="${base_path%"${base_path##*[![:space:]]}"}"
  base_path="${base_path%\"}"
  base_path="${base_path#\"}"

  if [ -z "$base_path" ] || [ "$base_path" = "???" ]; then
    echo "Set base_path in config/path/cluster.yaml or pass RESULTS_ROOT." >&2
    exit 1
  fi
  echo "${base_path}/results"
}

run_worker() {
  build_tasks

  task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
  if [ "$task_id" -lt 0 ] || [ "$task_id" -ge "${#TASK_NAMES[@]}" ]; then
    echo "Invalid SLURM_ARRAY_TASK_ID=${task_id}; expected 0..$((${#TASK_NAMES[@]} - 1))"
    exit 2
  fi

  config_name="${TASK_CONFIGS[$task_id]}"
  task_name="${TASK_NAMES[$task_id]}"
  dysts_system="${TASK_DYSTS_SYSTEMS[$task_id]}"
  venv_path="${VENV_PATH:-.venv}"
  results_root="${RESULTS_ROOT:-$(default_results_root)}"

  echo "========================================"
  echo "Evaluating: ${task_name}  (sweep=${SWEEP_NAME}, seed=${SEED})"
  echo "Config: ${config_name}"
  echo "Results root: ${results_root}"
  echo "Job: ${SLURM_JOB_ID:-unknown}, array task: ${task_id}"
  echo "Host: $(hostname)"
  echo "Working directory: $(pwd)"
  echo "========================================"

  if [ -f "${venv_path}/bin/activate" ]; then
    # shellcheck disable=SC1090
    . "${venv_path}/bin/activate"
    echo "Activated venv: ${venv_path}"
  else
    echo "No venv found at ${venv_path}; using current Python environment."
  fi

  echo "Python: $(which python)"

  overrides=(sweep_name="$SWEEP_NAME" seed="$SEED" results_root="$results_root")
  if [ -n "${EVALUATION:-}" ]; then
    overrides+=(evaluation="$EVALUATION")
  fi
  if [ -n "$dysts_system" ]; then
    overrides+=(dysts_system="$dysts_system")
  fi

  python runs/eval.py --config-name "$config_name" "${overrides[@]}"
}

submit_array() {
  if [ -z "${1:-}" ]; then
    usage
    exit 1
  fi

  SWEEP_NAME="$1"
  SEED="${2:-42}"
  export SWEEP_NAME SEED

  build_tasks
  mkdir -p "logs/eval/${SWEEP_NAME}"

  max_parallel="${MAX_PARALLEL:-${#TASK_NAMES[@]}}"
  last_task_index="$((${#TASK_NAMES[@]} - 1))"
  results_root="${RESULTS_ROOT:-$(default_results_root)}"

  sbatch_cmd=(
    sbatch
    --job-name "eval_${SWEEP_NAME}"
    --array "0-${last_task_index}%${max_parallel}"
    --cpus-per-task "${CPUS_PER_TASK:-8}"
    --mem "${MEM:-32G}"
    --time "${TIME:-08:00:00}"
    --output "logs/eval/${SWEEP_NAME}/%x_%A_%a.out"
    --error "logs/eval/${SWEEP_NAME}/%x_%A_%a.err"
    --export "ALL,RUN_WORKER=1,SWEEP_NAME=${SWEEP_NAME},SEED=${SEED},INCLUDE_DYSTS=${INCLUDE_DYSTS:-0},EVALUATION=${EVALUATION:-},VENV_PATH=${VENV_PATH:-.venv},RESULTS_ROOT=${results_root}"
  )

  if [ -n "${SLURM_ACCOUNT:-}" ]; then
    sbatch_cmd+=(--account "$SLURM_ACCOUNT")
  fi

  if [ -n "${PARTITION:-}" ]; then
    sbatch_cmd+=(--partition "$PARTITION")
  fi

  echo "Submitting ${#TASK_NAMES[@]} evaluation tasks for sweep=${SWEEP_NAME}, seed=${SEED}"
  echo "Results root: ${results_root}"
  echo "Logs: logs/eval/${SWEEP_NAME}/"
  "${sbatch_cmd[@]}" "$0"
}

if [ "${RUN_WORKER:-0}" = "1" ]; then
  run_worker
else
  submit_array "$@"
fi
