#!/usr/bin/env python3
"""Paper-style Wheelbot odometry XY rollout grid for two selected losses.

This script mirrors the rollout/data pipeline from
`plots/plot_wheelbot_test_odometry_xy.py`, but renders a compact paper grid
(similar in spirit to `plots/overview_loss_functions_with_arCRPS.py`).

Key idea:
- choose TWO losses at the top via `LOSS_TAGS` (column order = plot order)
- rows are rollout horizons; columns are loss functions
- keep rollout starts shared across horizons and losses for direct comparison
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plots.settings import (
    BASELINE_COLORS,
    BASELINES_NAMES,
    CORL_SETTINGS,
    FIG_FORMAT,
    FIGURES_DIR,
    WB_SWEEP_NAME,
    set_size,
)
from src.utils.wheelbot_model_io import load_wheelbot_model_bundle
from src.wheelbot.racing import (
    FIELDS_ACTIONS,
    FIELDS_STATES,
    ensure_split_meta_files,
    make_cut_fn,
)

try:
    from wheelbot_dataset import Dataset
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "wheelbot_dataset is required for this script. "
        "Run it in the project environment (for example: `.venv/bin/python ...`)."
    ) from exc


plt.rcParams.update(CORL_SETTINGS)


# -----------------------------------------------------------------------------
# Configuration (edit here)
# -----------------------------------------------------------------------------
RESULTS_ROOT = Path("results/wheelbot_dynamics")
SWEEP_NAME = WB_SWEEP_NAME
TRAIN_HORIZON = 40  # multi-step run directories are named <loss>_<H>
ENV_NAME = "wheelbot_racing"
RUN_ID: Optional[str] = "42"  # None -> newest run dir

# Choose exactly two losses here (left-to-right column order).
LOSS_TAGS = ["ar_crps", "kde"]
LOSS_TAG_TO_STRATEGY = {
    "single_step_elbo": "single_step_elbo",
    "elbo": "multi_step_elbo",
    "factored_crps": "multi_step_factored_crps",
    "kde": "multi_step_kde",
    "ar_crps": "multi_step_ar_crps",
}
LOSS_TAG_TO_BASELINE = {
    "single_step_elbo": "s-NLL",
    "elbo": "m-NLL",
    "factored_crps": "m-CRPS",
    "kde": "m-KDE",
    "ar_crps": "m-arCRPS",
}

DATA_ROOT = Path("data/wheelbot_racing")
TEST_DIR = "test"
DT = 0.006
TRIM_START = 10.0
TRIM_END = 2.0

WHEEL_RADIUS = 0.032
INITIAL_HEADING_DEG = 0.0
YAW_FIELD = "/q_yrp/yaw"
WHEEL_ANGLE_FIELD = "/q_DR/drive_wheel"
YAW_VEL_FIELD = "/dq_yrp/yaw_vel"
WHEEL_VEL_FIELD = "/dq_DR/drive_wheel"

# Grid layout: rows=horizons, cols=losses.
ROLLOUT_HORIZONS = [100, 500, 700]
N_PLOT_TRAJS = 4
N_SAMPLES = 20
SEED = 42
SHOW_SAMPLES = True
USE_EQUAL_DATA_ASPECT = False  # False -> keep all subplot boxes same size

# Axis window and zoom behavior.
BASE_X_LIM = (-0.5, 2.5)
BASE_Y_LIM = (-2.5, 2.5)
# Set per-horizon axis limits here (used first when available).
# Example:
#   300: {"x": (-0.2, 1.2), "y": (-0.4, 0.4)}
HORIZON_AXIS_LIMITS: Dict[int, Dict[str, Tuple[float, float]]] = {
    100: {"x": (-0.05, 0.5), "y": (-0.2, 0.2)},
    500: {"x": (-0.15, 1.5), "y": (-1.5, 1.5)},
    700: {"x": (-0.5, 2.0), "y": (-2, 2)},
}
# For horizons not listed in HORIZON_AXIS_LIMITS:
# - True: use automatic zoom per horizon
# - False: use BASE_X_LIM / BASE_Y_LIM
AUTO_ZOOM_UNSPECIFIED_HORIZONS = True
ZOOM_PAD_FRAC = 0.12
ZOOM_MIN_SPAN_X = 0.08
ZOOM_MIN_SPAN_Y = 0.08

# Paper figure style.
FIGURE_WIDTH_PT = 0.36 * 397
FIGURE_LAYOUT = {
    "hspace": 0.14,
    "wspace": 0.08,
    "top": 0.9,
    "bottom": 0.25,
    "left": 0.2,
    "right": 0.98,
}

OUT_DIR = FIGURES_DIR
OUT_BASENAME = "wheelbot_odometry_xy_paper_two_losses"

SKIP_PKL_NAMES = {"training_logs.pkl", "one_step_test_metrics.pkl", "stats.pkl"}


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _to_2d(arr: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {a.shape}")
    return a


def _normalize_loss_tag(loss_tag: str) -> str:
    tag = str(loss_tag).strip().lower()
    if tag in LOSS_TAG_TO_STRATEGY:
        return tag

    # Allow passing a strategy name directly.
    if tag.startswith("multi_step_") or tag.startswith("single_step_"):
        for known_tag, strategy in LOSS_TAG_TO_STRATEGY.items():
            if strategy == tag:
                return known_tag
        return tag

    raise ValueError(
        f"Unsupported LOSS_TAG '{loss_tag}'. "
        f"Choose one of {sorted(LOSS_TAG_TO_STRATEGY.keys())} or a direct strategy name."
    )


def _resolve_strategy_and_baseline(loss_tag: str) -> Tuple[str, str, str]:
    norm = _normalize_loss_tag(loss_tag)
    if norm in LOSS_TAG_TO_STRATEGY:
        strategy = LOSS_TAG_TO_STRATEGY[norm]
        baseline = LOSS_TAG_TO_BASELINE.get(norm, strategy)
        return norm, strategy, baseline

    # Direct strategy fallback.
    strategy = norm
    baseline = strategy
    return strategy, strategy, baseline


def _find_run_dir(strategy_dir: Path, run_id: Optional[str]) -> Path:
    if run_id is not None:
        d = strategy_dir / str(run_id)
        if not d.exists():
            raise FileNotFoundError(f"Requested run-id directory not found: {d}")
        return d

    candidates = [d for d in strategy_dir.iterdir() if d.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {strategy_dir}")

    def _sort_key(p: Path) -> Tuple[int, Any]:
        if p.name.isdigit():
            return (0, int(p.name))
        return (1, p.name)

    return sorted(candidates, key=_sort_key)[-1]


def _find_model_bundle(run_dir: Path) -> Path:
    preferred = [
        run_dir / "wheelbot_racing_multistep_vbll_model.pkl",
        run_dir / "wheelbot_racing_vbll_model.pkl",
    ]
    for p in preferred:
        if p.exists():
            return p

    candidates = [
        p
        for p in run_dir.glob("*.pkl")
        if p.is_file() and p.name not in SKIP_PKL_NAMES
    ]
    if not candidates:
        raise FileNotFoundError(f"No model bundle .pkl found in: {run_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _resolve_model_path(
    *,
    results_root: Path,
    sweep_name: str,
    env_name: str,
    strategy: str,
    run_id: Optional[str],
) -> Tuple[Path, Path]:
    horizon = 1 if strategy.startswith("single_step") else TRAIN_HORIZON
    strategy_dir = results_root / sweep_name / env_name / f"{strategy}_{horizon}"
    if not strategy_dir.exists():
        raise FileNotFoundError(f"Strategy directory not found: {strategy_dir}")

    run_dir = _find_run_dir(strategy_dir, run_id)
    model_path = _find_model_bundle(run_dir)
    return run_dir, model_path


def _resolve_state_indices(state_labels: List[str]) -> Tuple[int, int]:
    if YAW_VEL_FIELD not in state_labels:
        raise ValueError(f"State label '{YAW_VEL_FIELD}' not found in model states_labels: {state_labels}")
    if WHEEL_VEL_FIELD not in state_labels:
        raise ValueError(f"State label '{WHEEL_VEL_FIELD}' not found in model states_labels: {state_labels}")
    return state_labels.index(YAW_VEL_FIELD), state_labels.index(WHEEL_VEL_FIELD)


def _load_test_experiments(
    state_fields: List[str],
    action_fields: List[str],
) -> List[Dict[str, Any]]:
    test_split_dir = DATA_ROOT / TEST_DIR
    if not test_split_dir.exists():
        raise FileNotFoundError(f"Test split directory not found: {test_split_dir}")

    ensure_split_meta_files(DATA_ROOT)
    cut_fn = make_cut_fn(dt=DT, trim_start=TRIM_START, trim_end=TRIM_END)
    ds = Dataset(str(test_split_dir)).map(cut_fn)

    experiments: List[Dict[str, Any]] = []
    for group_name in sorted(ds.groups):
        group = ds.groups[group_name]
        for exp_idx, exp in enumerate(group.experiments):
            states = _to_2d(exp.to_numpy(state_fields), "states")
            actions = _to_2d(exp.to_numpy(action_fields), "actions")
            yaw = _to_2d(exp.to_numpy([YAW_FIELD]), "yaw")[:, 0]
            q_drive = _to_2d(exp.to_numpy([WHEEL_ANGLE_FIELD]), "q_drive")[:, 0]

            min_len = min(states.shape[0], actions.shape[0], yaw.shape[0], q_drive.shape[0])
            if min_len < 2:
                continue

            states = states[:min_len]
            actions = actions[:min_len]
            yaw = yaw[:min_len]
            q_drive = q_drive[:min_len]

            finite_mask = (
                np.isfinite(states).all(axis=1)
                & np.isfinite(actions).all(axis=1)
                & np.isfinite(yaw)
                & np.isfinite(q_drive)
            )
            states = states[finite_mask]
            actions = actions[finite_mask]
            yaw = yaw[finite_mask]
            q_drive = q_drive[finite_mask]
            if len(yaw) < 2:
                continue

            experiments.append(
                {
                    "name": f"{group_name}/exp_{exp_idx}",
                    "states": states,
                    "actions": actions,
                    "yaw": yaw,
                    "q_drive": q_drive,
                }
            )
    return experiments


def _build_rollout_candidates(experiments: List[Dict[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for exp in experiments:
        states = exp["states"]
        actions = exp["actions"]
        yaw = exp["yaw"]
        q_drive = exp["q_drive"]

        max_start = min(
            actions.shape[0] - horizon,
            states.shape[0] - 1,
            yaw.shape[0] - 1,
            q_drive.shape[0] - 1,
        )
        if max_start < 0:
            continue

        for start in range(max_start + 1):
            candidates.append(
                {
                    "name": exp["name"],
                    "start": int(start),
                    "init_state": states[start],
                    "actions": actions[start : start + horizon],
                    "yaw0": float(yaw[start]),
                    "q_drive0": float(q_drive[start]),
                    "gt_yaw": yaw[start : start + horizon + 1],
                    "gt_q_drive": q_drive[start : start + horizon + 1],
                }
            )
    return candidates


def _select_candidates(candidates: List[Dict[str, Any]], n_select: int, seed: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    n = min(n_select, len(candidates))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(candidates), size=n, replace=False)
    return [candidates[i] for i in sorted(idx.tolist())]


def _truncate_candidate(candidate: Dict[str, Any], horizon: int) -> Dict[str, Any]:
    return {
        "name": candidate["name"],
        "start": int(candidate["start"]),
        "init_state": np.asarray(candidate["init_state"], dtype=float),
        "actions": np.asarray(candidate["actions"], dtype=float)[:horizon],
        "yaw0": float(candidate["yaw0"]),
        "q_drive0": float(candidate["q_drive0"]),
        "gt_yaw": np.asarray(candidate["gt_yaw"], dtype=float)[: horizon + 1],
        "gt_q_drive": np.asarray(candidate["gt_q_drive"], dtype=float)[: horizon + 1],
    }


def _reconstruct_yaw_and_qdrive(
    pred_states_raw: np.ndarray,
    *,
    yaw_vel_idx: int,
    wheel_vel_idx: int,
    yaw0: float,
    q_drive0: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    horizon = pred_states_raw.shape[0] - 1
    yaw_vel = pred_states_raw[:, yaw_vel_idx]
    wheel_vel = pred_states_raw[:, wheel_vel_idx]

    yaw = np.empty(horizon + 1, dtype=float)
    q_drive = np.empty(horizon + 1, dtype=float)
    yaw[0] = yaw0
    q_drive[0] = q_drive0
    for t in range(horizon):
        yaw[t + 1] = yaw[t] + yaw_vel[t] * dt
        q_drive[t + 1] = q_drive[t] + wheel_vel[t] * dt
    return yaw, q_drive


def _compute_odometry_xy(
    yaw: np.ndarray,
    wheel_angle: np.ndarray,
    *,
    wheel_radius: float,
    initial_heading_rad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    yaw_world = np.unwrap(yaw) - float(yaw[0]) + initial_heading_rad
    displacement = np.diff(wheel_angle, prepend=wheel_angle[0]) * wheel_radius
    x = np.cumsum(np.cos(yaw_world) * displacement)
    y = np.cumsum(np.sin(yaw_world) * displacement)
    return x, y


def _clip_xy_after_nonfinite(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_out = np.asarray(x, dtype=float).copy()
    y_out = np.asarray(y, dtype=float).copy()
    finite = np.isfinite(x_out) & np.isfinite(y_out)
    bad_idx = np.flatnonzero(~finite)
    if bad_idx.size > 0:
        first_bad = int(bad_idx[0])
        x_out[first_bad:] = np.nan
        y_out[first_bad:] = np.nan
    return x_out, y_out


def _compute_rollouts_for_horizons(
    *,
    model_path: Path,
    selected_by_horizon: Dict[int, List[Dict[str, Any]]],
    expected_state_dim: int,
    expected_action_dim: int,
) -> Dict[int, List[Dict[str, Any]]]:
    model, bundle = load_wheelbot_model_bundle(str(model_path), rng_seed=SEED)

    model_init = dict(bundle.get("model_init", {}))
    state_dim = int(model_init["state_dim"])
    action_dim = int(model_init["action_dim"])
    if state_dim != expected_state_dim or action_dim != expected_action_dim:
        raise ValueError(
            f"Dimension mismatch for model {model_path}: "
            f"({state_dim}, {action_dim}) vs expected ({expected_state_dim}, {expected_action_dim})"
        )

    metadata = dict(bundle.get("metadata", {}))
    state_fields = list(metadata.get("states_labels", list(FIELDS_STATES)))
    yaw_vel_idx, wheel_vel_idx = _resolve_state_indices(state_fields)

    norm = dict(bundle["normalization"])
    state_mean = np.asarray(norm["state_mean"], dtype=float)
    state_std = np.asarray(norm["state_std"], dtype=float)
    action_mean = np.asarray(norm["action_mean"], dtype=float)
    action_std = np.asarray(norm["action_std"], dtype=float)

    rollout_mean_fn = jax.jit(jax.vmap(model.rollout_mean_trajectory, in_axes=(0, 0)))

    @jax.jit
    def rollout_samples_fn(init_states, action_sequences, sample_keys):
        def _sample_one_chunk(x0, acts, keys_for_chunk):
            return jax.vmap(lambda k: model.simulate(x0, acts, k))(keys_for_chunk)

        return jax.vmap(_sample_one_chunk, in_axes=(0, 0, 0))(init_states, action_sequences, sample_keys)

    out: Dict[int, List[Dict[str, Any]]] = {}
    initial_heading_rad = math.radians(INITIAL_HEADING_DEG)

    for horizon, selected in selected_by_horizon.items():
        init_states_raw = np.stack([c["init_state"] for c in selected], axis=0)
        actions_raw = np.stack([c["actions"] for c in selected], axis=0)
        init_states_norm = (init_states_raw - state_mean) / state_std
        actions_norm = (actions_raw - action_mean) / action_std

        pred_traj_norm = np.array(rollout_mean_fn(jnp.array(init_states_norm), jnp.array(actions_norm)))
        pred_traj_raw = pred_traj_norm * state_std[None, None, :] + state_mean[None, None, :]

        key = jax.random.PRNGKey(SEED + 1000 + horizon)
        sample_keys = jax.random.split(key, len(selected) * N_SAMPLES).reshape(len(selected), N_SAMPLES, 2)
        sample_traj_norm = np.array(
            rollout_samples_fn(
                jnp.array(init_states_norm),
                jnp.array(actions_norm),
                jnp.array(sample_keys),
            )
        )
        sample_traj_raw = sample_traj_norm * state_std[None, None, None, :] + state_mean[None, None, None, :]

        rollouts: List[Dict[str, Any]] = []
        for i, c in enumerate(selected):
            gt_x, gt_y = _compute_odometry_xy(
                yaw=c["gt_yaw"],
                wheel_angle=c["gt_q_drive"],
                wheel_radius=WHEEL_RADIUS,
                initial_heading_rad=initial_heading_rad,
            )
            gt_x, gt_y = _clip_xy_after_nonfinite(gt_x, gt_y)

            yaw_pred, q_pred = _reconstruct_yaw_and_qdrive(
                pred_states_raw=pred_traj_raw[i],
                yaw_vel_idx=yaw_vel_idx,
                wheel_vel_idx=wheel_vel_idx,
                yaw0=c["yaw0"],
                q_drive0=c["q_drive0"],
                dt=DT,
            )
            mean_x, mean_y = _compute_odometry_xy(
                yaw=yaw_pred,
                wheel_angle=q_pred,
                wheel_radius=WHEEL_RADIUS,
                initial_heading_rad=initial_heading_rad,
            )
            mean_x, mean_y = _clip_xy_after_nonfinite(mean_x, mean_y)

            samples_x = np.zeros((N_SAMPLES, horizon + 1), dtype=float)
            samples_y = np.zeros((N_SAMPLES, horizon + 1), dtype=float)
            for si in range(N_SAMPLES):
                yaw_s, q_s = _reconstruct_yaw_and_qdrive(
                    pred_states_raw=sample_traj_raw[i, si],
                    yaw_vel_idx=yaw_vel_idx,
                    wheel_vel_idx=wheel_vel_idx,
                    yaw0=c["yaw0"],
                    q_drive0=c["q_drive0"],
                    dt=DT,
                )
                x_s, y_s = _compute_odometry_xy(
                    yaw=yaw_s,
                    wheel_angle=q_s,
                    wheel_radius=WHEEL_RADIUS,
                    initial_heading_rad=initial_heading_rad,
                )
                x_s, y_s = _clip_xy_after_nonfinite(x_s, y_s)
                samples_x[si] = x_s
                samples_y[si] = y_s

            rollouts.append(
                {
                    "name": c["name"],
                    "start": c["start"],
                    "gt_x": gt_x,
                    "gt_y": gt_y,
                    "mean_x": mean_x,
                    "mean_y": mean_y,
                    "samples_x": samples_x,
                    "samples_y": samples_y,
                }
            )
        out[horizon] = rollouts

    return out


def _title_bbox_for_baseline(baseline: str) -> dict:
    if baseline in BASELINE_COLORS:
        return {
            "boxstyle": "round,pad=0.25",
            "facecolor": BASELINE_COLORS[baseline],
            "edgecolor": "none",
            "alpha": 0.20,
        }
    return {
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.35,
    }


def _draw_background_grid(ax, x_lim: Tuple[float, float], y_lim: Tuple[float, float]) -> None:
    x_ticks = np.linspace(x_lim[0], x_lim[1], 4)
    y_ticks = np.linspace(y_lim[0], y_lim[1], 5)
    for xv in x_ticks:
        ax.vlines(
            xv,
            y_lim[0],
            y_lim[1],
            color="gray",
            linestyle="-",
            alpha=0.20,
            lw=0.8,
            zorder=0,
        )
    for yv in y_ticks:
        ax.hlines(
            yv,
            x_lim[0],
            x_lim[1],
            color="gray",
            linestyle="-",
            alpha=0.20,
            lw=0.8,
            zorder=0,
        )


def _compute_auto_limits(
    rollouts: List[Dict[str, Any]],
    base_x_lim: Tuple[float, float],
    base_y_lim: Tuple[float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    x_chunks: List[np.ndarray] = []
    y_chunks: List[np.ndarray] = []
    for r in rollouts:
        x_chunks.append(np.asarray(r["gt_x"], dtype=float))
        x_chunks.append(np.asarray(r["mean_x"], dtype=float))
        y_chunks.append(np.asarray(r["gt_y"], dtype=float))
        y_chunks.append(np.asarray(r["mean_y"], dtype=float))

    if not x_chunks or not y_chunks:
        return base_x_lim, base_y_lim

    x_all = np.concatenate(x_chunks, axis=0)
    y_all = np.concatenate(y_chunks, axis=0)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if not np.any(finite):
        return base_x_lim, base_y_lim

    x_valid = x_all[finite]
    y_valid = y_all[finite]

    x_min = float(np.min(x_valid))
    x_max = float(np.max(x_valid))
    y_min = float(np.min(y_valid))
    y_max = float(np.max(y_valid))

    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)
    x_span = max(x_max - x_min, ZOOM_MIN_SPAN_X)
    y_span = max(y_max - y_min, ZOOM_MIN_SPAN_Y)

    x_half = 0.5 * x_span * (1.0 + 2.0 * ZOOM_PAD_FRAC)
    y_half = 0.5 * y_span * (1.0 + 2.0 * ZOOM_PAD_FRAC)

    x_lim = (
        max(base_x_lim[0], x_center - x_half),
        min(base_x_lim[1], x_center + x_half),
    )
    y_lim = (
        max(base_y_lim[0], y_center - y_half),
        min(base_y_lim[1], y_center + y_half),
    )

    if x_lim[0] >= x_lim[1] or y_lim[0] >= y_lim[1]:
        return base_x_lim, base_y_lim
    return x_lim, y_lim


def _coerce_axis_limit_pair(value: Any, *, label: str) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-item tuple/list, got: {value}")
    lo = float(value[0])
    hi = float(value[1])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"{label} must be finite, got: {value}")
    if lo >= hi:
        raise ValueError(f"{label} lower bound must be < upper bound, got: {value}")
    return lo, hi


def _limits_for_horizon(
    *,
    horizon: int,
    rollouts: List[Dict[str, Any]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    cfg = HORIZON_AXIS_LIMITS.get(int(horizon))
    if cfg is not None:
        x_lim = _coerce_axis_limit_pair(cfg.get("x"), label=f"HORIZON_AXIS_LIMITS[{horizon}]['x']")
        y_lim = _coerce_axis_limit_pair(cfg.get("y"), label=f"HORIZON_AXIS_LIMITS[{horizon}]['y']")
        if x_lim is None or y_lim is None:
            raise ValueError(
                f"HORIZON_AXIS_LIMITS[{horizon}] must contain both 'x' and 'y' keys "
                "with [min, max] values."
            )
        return x_lim, y_lim

    if AUTO_ZOOM_UNSPECIFIED_HORIZONS:
        return _compute_auto_limits(rollouts, BASE_X_LIM, BASE_Y_LIM)
    return BASE_X_LIM, BASE_Y_LIM


def _plot_two_loss_horizon(
    *,
    per_loss_per_horizon: Dict[str, Dict[int, List[Dict[str, Any]]]],
    horizon: int,
    loss_infos: List[Dict[str, Any]],
    out_pdf: Path,
) -> None:
    n_rows = 1
    n_cols = len(loss_infos)

    fig_w, fig_h = set_size(FIGURE_WIDTH_PT, subplots=(n_rows, n_cols))
    fig_h *= 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    traj_colors = list(plt.get_cmap("tab10").colors)

    for col, info in enumerate(loss_infos):
        baseline = str(info["baseline"])
        baseline_title = BASELINES_NAMES.get(baseline, baseline)
        axes[0, col].set_title(
            baseline_title,
            fontsize=7,
            pad=1,
            bbox=_title_bbox_for_baseline(baseline),
        )

    row_rollouts: List[Dict[str, Any]] = []
    for info in loss_infos:
        strategy = str(info["strategy"])
        row_rollouts.extend(per_loss_per_horizon[strategy][horizon])
    x_lim, y_lim = _limits_for_horizon(horizon=horizon, rollouts=row_rollouts)

    for col, info in enumerate(loss_infos):
        strategy = str(info["strategy"])
        ax = axes[0, col]
        rollouts = per_loss_per_horizon[strategy][horizon]

        for i, r in enumerate(rollouts):
            color = traj_colors[i % 10]

            if SHOW_SAMPLES:
                for si in range(r["samples_x"].shape[0]):
                    ax.plot(
                        r["samples_x"][si],
                        r["samples_y"][si],
                        color=color,
                        lw=0.55,
                        alpha=0.15,
                        zorder=1,
                    )

            ax.plot(
                r["mean_x"],
                r["mean_y"],
                color=color,
                lw=1.,
                linestyle="--",
                alpha=0.8,
                zorder=4,
            )
            ax.plot(
                r["gt_x"],
                r["gt_y"],
                color=color,
                lw=1.,
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                [r["gt_x"][0]],
                [r["gt_y"][0]],
                s=3,
                color=color,
                alpha=0.95,
                zorder=5,
            )

        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        if horizon == 500:
            ax.set_xticks([0, 1, 1.5])
            ax.set_xticklabels(["0", "1", ""])
        if horizon == 100:
            ax.set_xticks([0, 0.2, 0.4, 0.5])
            ax.set_xticklabels(["0.0", "0.2", "0.4", ""])
            
        if USE_EQUAL_DATA_ASPECT:
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_aspect("auto")
        _draw_background_grid(ax, x_lim=x_lim, y_lim=y_lim)

        ax.tick_params(axis="x", labelsize=5)
        ax.tick_params(axis="y", labelsize=5)
        ax.set_xlabel("$x$ [m]", fontsize=6, labelpad=1)

        if col == 0:
            ax.set_ylabel(f"$y$ [m]", fontsize=6, labelpad=0)
            ax.text(
                0.95,
                0.95,
                r"$\mathcal{X}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
            )
        else:
            ax.tick_params(labelleft=False)
            ax.set_yticks([])

        sns.despine(
            ax=ax,
            left=(col != 0),
            bottom=False,
            offset=2,
            trim=True,
        )

    fig.subplots_adjust(
        hspace=FIGURE_LAYOUT["hspace"],
        wspace=FIGURE_LAYOUT["wspace"],
        top=FIGURE_LAYOUT["top"],
        bottom=FIGURE_LAYOUT["bottom"],
        left=FIGURE_LAYOUT["left"],
        right=FIGURE_LAYOUT["right"],
    )
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    if len(ROLLOUT_HORIZONS) == 0:
        raise ValueError("ROLLOUT_HORIZONS must contain at least one entry.")
    if any(h <= 0 for h in ROLLOUT_HORIZONS):
        raise ValueError(f"All rollout horizons must be positive, got {ROLLOUT_HORIZONS}")
    if N_PLOT_TRAJS <= 0:
        raise ValueError(f"N_PLOT_TRAJS must be positive, got {N_PLOT_TRAJS}")
    if N_SAMPLES <= 0:
        raise ValueError(f"N_SAMPLES must be positive, got {N_SAMPLES}")

    if len(LOSS_TAGS) != 2:
        raise ValueError(f"LOSS_TAGS must contain exactly 2 entries, got {LOSS_TAGS}")

    loss_infos: List[Dict[str, Any]] = []
    for loss_tag in LOSS_TAGS:
        loss_tag_norm, strategy, baseline = _resolve_strategy_and_baseline(loss_tag)
        run_dir, model_path = _resolve_model_path(
            results_root=RESULTS_ROOT,
            sweep_name=SWEEP_NAME,
            env_name=ENV_NAME,
            strategy=strategy,
            run_id=RUN_ID,
        )
        loss_infos.append(
            {
                "loss_tag": loss_tag,
                "loss_tag_normalized": loss_tag_norm,
                "strategy": strategy,
                "baseline": baseline,
                "run_dir": run_dir,
                "model_path": model_path,
            }
        )

    first_model_path = Path(loss_infos[0]["model_path"])
    _, first_bundle = load_wheelbot_model_bundle(str(first_model_path), rng_seed=SEED)
    first_init = dict(first_bundle.get("model_init", {}))
    state_dim = int(first_init["state_dim"])
    action_dim = int(first_init["action_dim"])

    metadata = dict(first_bundle.get("metadata", {}))
    state_fields = list(metadata.get("states_labels", list(FIELDS_STATES)))
    action_fields = list(metadata.get("actions_labels", list(FIELDS_ACTIONS)))
    if len(state_fields) != state_dim:
        raise ValueError(f"states_labels length {len(state_fields)} != state_dim {state_dim}")
    if len(action_fields) != action_dim:
        raise ValueError(f"actions_labels length {len(action_fields)} != action_dim {action_dim}")

    experiments = _load_test_experiments(state_fields=state_fields, action_fields=action_fields)
    if not experiments:
        raise RuntimeError("No usable test trajectories found.")

    horizons = sorted(set(int(h) for h in ROLLOUT_HORIZONS))
    max_h = max(horizons)

    max_candidates = _build_rollout_candidates(experiments, horizon=max_h)
    shared_selected = _select_candidates(max_candidates, n_select=N_PLOT_TRAJS, seed=SEED)
    if not shared_selected:
        raise RuntimeError(f"No valid rollout candidates for max horizon={max_h}")

    selected_by_horizon: Dict[int, List[Dict[str, Any]]] = {
        h: [_truncate_candidate(c, horizon=h) for c in shared_selected] for h in horizons
    }

    per_loss_per_horizon: Dict[str, Dict[int, List[Dict[str, Any]]]] = {}
    for info in loss_infos:
        strategy = str(info["strategy"])
        model_path = Path(info["model_path"])
        per_loss_per_horizon[strategy] = _compute_rollouts_for_horizons(
            model_path=model_path,
            selected_by_horizon=selected_by_horizon,
            expected_state_dim=state_dim,
            expected_action_dim=action_dim,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    horizons_tag = "-".join(str(h) for h in horizons)
    strategies_tag = "__".join(str(info["strategy"]) for info in loss_infos)
    output_pdfs_by_horizon: Dict[int, Path] = {}
    for horizon in horizons:
        out_pdf_h = OUT_DIR / f"{OUT_BASENAME}__{strategies_tag}__H_{horizon}.{FIG_FORMAT}"
        _plot_two_loss_horizon(
            per_loss_per_horizon=per_loss_per_horizon,
            horizon=horizon,
            loss_infos=loss_infos,
            out_pdf=out_pdf_h,
        )
        output_pdfs_by_horizon[int(horizon)] = out_pdf_h

    summary = {
        "results_root": RESULTS_ROOT,
        "sweep_name": SWEEP_NAME,
        "env_name": ENV_NAME,
        "loss_tags": list(LOSS_TAGS),
        "loss_infos": loss_infos,
        "run_id": RUN_ID,
        "data_root": DATA_ROOT,
        "test_dir": TEST_DIR,
        "dt": DT,
        "trim_start": TRIM_START,
        "trim_end": TRIM_END,
        "wheel_radius": WHEEL_RADIUS,
        "initial_heading_deg": INITIAL_HEADING_DEG,
        "yaw_field": YAW_FIELD,
        "wheel_angle_field": WHEEL_ANGLE_FIELD,
        "yaw_vel_field": YAW_VEL_FIELD,
        "wheel_vel_field": WHEEL_VEL_FIELD,
        "rollout_horizons": horizons,
        "n_cols": len(loss_infos),
        "n_plot_trajs": N_PLOT_TRAJS,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "show_samples": SHOW_SAMPLES,
        "base_x_lim": BASE_X_LIM,
        "base_y_lim": BASE_Y_LIM,
        "horizon_axis_limits": HORIZON_AXIS_LIMITS,
        "auto_zoom_unspecified_horizons": AUTO_ZOOM_UNSPECIFIED_HORIZONS,
        "output_pdfs_by_horizon": output_pdfs_by_horizon,
        "shared_selected_rollouts": [
            {"name": c["name"], "start": int(c["start"])} for c in shared_selected
        ],
    }

    out_json = OUT_DIR / f"{OUT_BASENAME}__{strategies_tag}__H_{horizons_tag}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(summary), f, indent=2)

    print("Losses:")
    for info in loss_infos:
        print(
            f"  {info['loss_tag']} -> {info['strategy']} ({info['baseline']}) | "
            f"run={info['run_dir']}"
        )
    print(f"Horizons:    {horizons}")
    for horizon in horizons:
        print(f"Saved H={horizon}: {output_pdfs_by_horizon[horizon]}")
    print(f"Saved config:{out_json}")


if __name__ == "__main__":
    main()
