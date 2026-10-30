#!/usr/bin/env bash
# Make all paper figures and the results table into one folder, FIGURES_DIR
# (default figures/<SWEEP_NAME>/, see plots/settings.py). Run from the repo root:
#   SWEEP_NAME=<sweep_name> WB_SWEEP_NAME=<wb_sweep_name> plots/make_all.sh
set -euo pipefail

export PATH=/Library/TeX/texbin:$PATH

uv run python plots/overview_loss_functions.py
uv run python plots/results_vslc.py
uv run python plots/gradients_over_epochs.py
uv run python plots/eval_rank_summary.py
uv run python plots/make_eval_latex_table.py
uv run python plots/make_wheelbot_latex_table.py
uv run python plots/wheelbot_rollout_states.py
uv run python plots/wheelbot_odometry.py

echo "Outputs in: $(uv run python -c 'from plots.settings import FIGURES_DIR; print(FIGURES_DIR)')"
