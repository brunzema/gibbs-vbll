#!/usr/bin/env python3
"""Narrow per-state Wheelbot rollout plot using eval-style chunk sampling.

This script mirrors chunk selection from `src/wheelbot/eval_wheelbot_vbll_rollouts.py`:
1) build all valid contiguous chunks from trajectory data
2) sample chunk indices via `np.random.default_rng(seed).choice(..., replace=False)`

Each panel shows:
  solid  = ground truth
  dashed = mean prediction
  band   = 10-90th percentile of stochastic rollouts
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from plots.settings import BASELINE_COLORS, CACHE_DIR, CORL_SETTINGS, FIG_FORMAT, FIGURES_DIR, WB_SWEEP_NAME, set_size
from src.training.wheelbot_rollout_eval import build_rollout_chunks_from_trajectories
from src.utils.wheelbot_model_io import load_wheelbot_model_bundle
from src.wheelbot.racing import (
    ensure_split_meta_files,
    extract_full_trajectories,
    load_prediction_split,
    make_cut_fn,
)

SWEEP_NAME = WB_SWEEP_NAME
SWEEP_ROOT = f"results/wheelbot_dynamics/{SWEEP_NAME}"
TRAIN_HORIZON = 40  # run directories are named <loss>_<H>
ENV_NAME = "wheelbot_racing"
RUN_ID = "42"
MODEL_FILENAME = "wheelbot_racing_multistep_vbll_model.pkl"

LOSS_FUNCTIONS = [
    "multi_step_elbo",
    "multi_step_factored_crps",
    "multi_step_ar_crps",
    "multi_step_kde",
]
LOSS_FUNCTION_TITLES = {
    "multi_step_elbo": r"\texttt{m-NLL}",
    "multi_step_factored_crps": r"\texttt{m-CRPS}",
    "multi_step_kde": r"\texttt{m-KDE}",
    "multi_step_ar_crps": r"\texttt{m-arCRPS}",
}
LOSS_FUNCTION_TO_BASELINE = {
    "multi_step_elbo": "m-NLL",
    "multi_step_factored_crps": "m-CRPS",
    "multi_step_kde": "m-KDE",
    "multi_step_ar_crps": "m-arCRPS",
}

DEFAULT_EVAL_TRAJECTORIES_PATH = "examples/dataset/test_trajectories.pkl"
DEFAULT_FIGURE_WIDTH_PT = 397
DEFAULT_OUT_DIR = FIGURES_DIR
DEFAULT_OUT_FILENAME = f"wheelbot_rollout_states_grid.{FIG_FORMAT}"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Trajectory source:
# - "split": load the test split from DATA_ROOT (same preprocessing as training)
# - "pickle": load pre-exported trajectories from EVAL_TRAJECTORIES_PATH
TRAJECTORY_SOURCE = "split"

EVAL_TRAJECTORIES_PATH = DEFAULT_EVAL_TRAJECTORIES_PATH
DATA_ROOT = "data/wheelbot_racing"
TEST_DIR = "test"
RESAMPLE_DT = 0.006
TRIM_START = 2.0
TRIM_END = 2.0

ROLLOUT_LENGTH = 101
N_EVAL_CHUNKS = 100
N_SAMPLES = 50
NUM_PLOT_TRAJS = 5
SEED = 42

# Optional explicit state projection when trajectory state dim != model state dim.
# Example for 10->7 Wheelbot racing subset:
# [1, 2, 3, 4, 5, 8, 9]
STATE_INDEX_OVERRIDE = None

# Map raw state labels to math-form y-axis labels.
# Edit this dictionary to customize symbols.
STATE_LABEL_TO_MATH = {
    "/q_yrp/yaw": r"Yaw $\psi$",
    "/q_yrp/roll": r"Roll $\phi$",
    "/q_yrp/pitch": r"Pitch $\theta$",
    "/dq_yrp/yaw_vel": r"Yaw vel $\dot{\psi}$",
    "/dq_yrp/roll_vel": r"Roll vel $\dot{\phi}$",
    "/dq_yrp/pitch_vel": r"Pitch vel $\dot{\theta}$",
    "/q_DR/drive_wheel": r"Drive wheel $q_{\mathrm{d}}$",
    "/q_DR/reaction_wheel": r"Reaction wheel $q_{\mathrm{r}}$",
    "/dq_DR/drive_wheel": "Drive wheel\n"+r"velocity $\dot{q}_{\mathrm{d}}$",
    "/dq_DR/reaction_wheel": "Reaction wheel\n" + r" velocity $\dot{q}_{\mathrm{r}}$",
}

FIGURE_WIDTH_PT = DEFAULT_FIGURE_WIDTH_PT
OUT_DIR = DEFAULT_OUT_DIR
OUT_FILENAME = DEFAULT_OUT_FILENAME
OUT_FILENAME_SUBSET = f"wheelbot_rollout_states_grid_vel_subset.{FIG_FORMAT}"

# Subset for second plot (order preserved).
SECOND_PLOT_STATE_LABELS = [
    "/dq_yrp/roll_vel",
    "/dq_yrp/yaw_vel",
    "/dq_yrp/pitch_vel",
]

# Rollout cache (to skip expensive recomputation while iterating on plotting).
LOAD_ROLLOUT_CACHE_IF_AVAILABLE = True
SAVE_ROLLOUT_CACHE = True
FORCE_RECOMPUTE_ROLLOUTS = False
ROLLOUT_CACHE_DIR = CACHE_DIR / SWEEP_NAME
ROLLOUT_CACHE_FILENAME = "rollout_states_grid_cache.pkl"

# Figure-layout tuning per plot (independent on purpose).
FULL_GRID_LAYOUT = {
    "hspace": 0.10,
    "wspace": 0.12,
    "top": 0.97,
    "bottom": 0.06,
    "left": 0.07,
    "right": 0.98,
    "supxlabel_y": 0.04,
}
SUBSET_GRID_LAYOUT = {
    "hspace": 0.1,
    "wspace": 0.12,
    "top": 0.95,
    "bottom": 0.12,
    "left": 0.03,
    "right": 0.985,
    "supxlabel_y": -0.1,
}


def _resolve_model_paths() -> Dict[str, str]:
    out = {}
    for loss_name in LOSS_FUNCTIONS:
        model_path = os.path.join(
            SWEEP_ROOT, ENV_NAME, f"{loss_name}_{TRAIN_HORIZON}", RUN_ID, MODEL_FILENAME
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found for '{loss_name}': {model_path}")
        out[loss_name] = model_path
    return out


def _load_trajectories():
    if TRAJECTORY_SOURCE == "split":
        data_root = Path(DATA_ROOT)
        test_split_dir = data_root / TEST_DIR
        if not test_split_dir.exists():
            raise FileNotFoundError(f"Test split directory not found: {test_split_dir}")

        ensure_split_meta_files(data_root)
        cut_fn = make_cut_fn(dt=RESAMPLE_DT, trim_start=TRIM_START, trim_end=TRIM_END)
        _, _, _, ds_test = load_prediction_split(test_split_dir, cut_fn)
        trajectories = extract_full_trajectories(ds_test)
        if not trajectories:
            raise RuntimeError(f"No trajectories extracted from: {test_split_dir}")

        dt = RESAMPLE_DT
        states_labels = []
        source_desc = (
            f"split:{test_split_dir} (dt={RESAMPLE_DT}, "
            f"trim_start={TRIM_START}, trim_end={TRIM_END})"
        )
        return trajectories, states_labels, dt, source_desc

    if TRAJECTORY_SOURCE == "pickle":
        with open(EVAL_TRAJECTORIES_PATH, "rb") as f:
            eval_data = pickle.load(f)

        trajectories = eval_data.get("trajectories", [])
        if not trajectories:
            raise ValueError(f"No trajectories found in {EVAL_TRAJECTORIES_PATH}")

        states_labels = list(eval_data.get("states_labels", []))
        dt = float(eval_data.get("dt", RESAMPLE_DT))
        source_desc = f"pickle:{EVAL_TRAJECTORIES_PATH}"
        return trajectories, states_labels, dt, source_desc

    raise ValueError(
        f"Unsupported TRAJECTORY_SOURCE='{TRAJECTORY_SOURCE}'. "
        "Use 'split' or 'pickle'."
    )


def _project_trajectories_states(
    trajectories: List[dict],
    state_indices: Sequence[int],
) -> List[dict]:
    projected = []
    for traj in trajectories:
        states = np.asarray(traj["states"])
        traj_new = dict(traj)
        traj_new["states"] = states[:, state_indices]
        projected.append(traj_new)
    return projected


def _run_rollouts_for_model(
    model,
    chunks: List[tuple],
    state_mean: np.ndarray,
    state_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
):
    n_eval_local = len(chunks)
    init_states_raw = np.stack([c[0] for c in chunks])
    actions_raw = np.stack([c[1] for c in chunks])

    init_norm = (init_states_raw - state_mean) / state_std
    actions_norm = (actions_raw - action_mean) / action_std

    mean_rollout_fn = jax.jit(jax.vmap(model.rollout_mean_trajectory, in_axes=(0, 0)))
    mean_traj_norm = mean_rollout_fn(jnp.array(init_norm), jnp.array(actions_norm))
    mean_pred_norm = np.array(mean_traj_norm[:, 1:, :])  # (N, T, D)

    key = jax.random.PRNGKey(SEED)
    sample_keys = jax.random.split(key, n_eval_local * N_SAMPLES).reshape(
        n_eval_local, N_SAMPLES, 2
    )

    all_sample_norm = []
    for ci in range(n_eval_local):
        samps = jax.jit(
            jax.vmap(
                lambda k: model.simulate(
                    jnp.array(init_norm[ci]),
                    jnp.array(actions_norm[ci]),
                    k,
                )
            )
        )(sample_keys[ci])
        all_sample_norm.append(np.array(samps))

    sample_preds_norm = np.stack([s[:, 1:, :] for s in all_sample_norm])  # (N, S, T, D)
    mean_pred_raw = (
        mean_pred_norm * state_std[None, None, :] + state_mean[None, None, :]
    )
    sample_pred_raw = (
        sample_preds_norm * state_std[None, None, None, :]
        + state_mean[None, None, None, :]
    )
    return mean_pred_raw, sample_pred_raw


def _cache_path() -> Path:
    return Path(ROLLOUT_CACHE_DIR) / ROLLOUT_CACHE_FILENAME


def _cache_config(model_paths: Dict[str, str]) -> dict:
    return {
        "loss_functions": list(LOSS_FUNCTIONS),
        "model_paths": dict(model_paths),
        "trajectory_source": TRAJECTORY_SOURCE,
        "eval_trajectories_path": EVAL_TRAJECTORIES_PATH,
        "data_root": DATA_ROOT,
        "test_dir": TEST_DIR,
        "resample_dt": RESAMPLE_DT,
        "trim_start": TRIM_START,
        "trim_end": TRIM_END,
        "rollout_length": ROLLOUT_LENGTH,
        "n_eval_chunks": N_EVAL_CHUNKS,
        "n_samples": N_SAMPLES,
        "num_plot_trajs": NUM_PLOT_TRAJS,
        "seed": SEED,
        "state_index_override": list(STATE_INDEX_OVERRIDE) if STATE_INDEX_OVERRIDE is not None else None,
    }


def _load_rollout_cache(expected_config: dict):
    cache_path = _cache_path()
    if not cache_path.exists():
        return None
    with cache_path.open("rb") as f:
        payload = pickle.load(f)
    if payload.get("config") != expected_config:
        print("Rollout cache config mismatch, recomputing rollouts.")
        return None
    print(f"Loaded rollout cache: {cache_path}")
    return payload


def _save_rollout_cache(payload: dict):
    cache_path = _cache_path()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved rollout cache: {cache_path}")


def _resolve_subset_state_indices(states_labels: List[str]) -> List[int]:
    missing = [lab for lab in SECOND_PLOT_STATE_LABELS if lab not in states_labels]
    if missing:
        raise ValueError(
            "SECOND_PLOT_STATE_LABELS contains unknown labels: "
            f"{missing}. Available labels: {states_labels}"
        )
    return [states_labels.index(lab) for lab in SECOND_PLOT_STATE_LABELS]


def _title_bbox_for_loss(loss_name: str) -> dict:
    baseline = LOSS_FUNCTION_TO_BASELINE.get(loss_name)
    if baseline is None or baseline not in BASELINE_COLORS:
        return dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.2)
    return dict(
        boxstyle="round,pad=0.2",
        facecolor=BASELINE_COLORS[baseline],
        edgecolor="none",
        alpha=0.2,
    )


def _draw_gray_background_grid(ax, x_lo: float, x_hi: float, y_lo: float, y_hi: float):
    x_ticks = np.linspace(x_lo, x_hi, 4)
    y_ticks = np.linspace(y_lo, y_hi, 5)
    for xv in x_ticks:
        ax.vlines(
            xv,
            y_lo,
            y_hi,
            color="gray",
            linestyle="-",
            alpha=0.2,
            lw=0.8,
            zorder=0,
        )
    for yv in y_ticks:
        ax.hlines(
            yv,
            x_lo,
            x_hi,
            color="gray",
            linestyle="-",
            alpha=0.2,
            lw=0.8,
            zorder=0,
        )


def _plot_full_grid_from_payload(payload: dict) -> str:
    plt.rcParams.update(CORL_SETTINGS)

    states_labels = list(payload["states_labels"])
    gt_next_raw = np.asarray(payload["gt_next_raw"])
    time_axis = np.asarray(payload["time_axis"])
    plot_idxs = np.asarray(payload["plot_idxs"])
    results = payload["results"]

    state_indices = list(range(len(states_labels)))
    n_rows = len(state_indices)
    n_cols = len(LOSS_FUNCTIONS)

    fig_w, fig_h = set_size(FIGURE_WIDTH_PT, subplots=(n_rows, n_cols))
    fig_h *= 1.15
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        # sharex=True,
        # sharey="row",
        squeeze=False,
    )

    colors = plt.cm.tab10(np.arange(len(plot_idxs)))
    y_limits = {}
    for si in state_indices:
        gt_all = gt_next_raw[plot_idxs, :, si]
        gt_min, gt_max = float(np.min(gt_all)), float(np.max(gt_all))
        span = max(gt_max - gt_min, 1e-8)
        margin = max(0.1 * span, 1e-3)
        y_limits[si] = (gt_min - margin, gt_max + margin)

    for col, loss_name in enumerate(LOSS_FUNCTIONS):
        mean_pred_raw = np.asarray(results[loss_name]["mean_pred_raw"])
        sample_pred_raw = np.asarray(results[loss_name]["sample_pred_raw"])

        col_title = LOSS_FUNCTION_TITLES.get(loss_name, loss_name)
        axes[0, col].set_title(
            col_title,
            fontsize=7,
            bbox=_title_bbox_for_loss(loss_name),
            pad=2,
        )

        for row, si in enumerate(state_indices):
            ax = axes[row, col]
            y_lo, y_hi = y_limits[si]

            for pi, ci in enumerate(plot_idxs):
                color = colors[pi]
                ax.plot(
                    time_axis,
                    gt_next_raw[ci, :, si],
                    "-",
                    color=color,
                    lw=0.8,
                    alpha=0.90,
                    zorder=4,
                )

                ax.plot(
                    time_axis,
                    mean_pred_raw[ci, :, si],
                    "--",
                    color=color,
                    lw=0.75,
                    alpha=0.85,
                    zorder=3,
                )

                samps_si = sample_pred_raw[ci, :, :, si]
                lo = np.nanpercentile(samps_si, 5, axis=0)
                hi = np.nanpercentile(samps_si, 95, axis=0)
                ax.fill_between(time_axis, lo, hi, color=color, alpha=0.12, zorder=2)

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(time_axis[0], time_axis[-1])
            _draw_gray_background_grid(
                ax=ax,
                x_lo=float(time_axis[0]),
                x_hi=float(time_axis[-1]),
                y_lo=float(y_lo),
                y_hi=float(y_hi),
            )
            ax.set_yticks([])
            ax.tick_params(axis="x", labelsize=5)

            if col == 0:
                raw_label = states_labels[si] if states_labels else f"state_{si}"
                label = STATE_LABEL_TO_MATH.get(raw_label, raw_label)
                ax.set_ylabel(label, fontsize=6, rotation=90, labelpad=9, va="center")
            else:
                ax.set_ylabel("")

            is_bottom = row == (n_rows - 1)
            sns.despine(
                ax=ax,
                left=True,
                bottom=not is_bottom,
                offset=2,
                trim=True,
            )
            if not is_bottom:
                ax.tick_params(labelbottom=False)
                ax.set_xticks([])
            else:
                ax.set_xlabel(r"$t$ [s]", fontsize=6, labelpad=2)

    # fig.supxlabel(r"$t$ [s]", fontsize=6, y=FULL_GRID_LAYOUT["supxlabel_y"])
    fig.subplots_adjust(
        hspace=FULL_GRID_LAYOUT["hspace"],
        wspace=FULL_GRID_LAYOUT["wspace"],
        top=FULL_GRID_LAYOUT["top"],
        bottom=FULL_GRID_LAYOUT["bottom"],
        left=FULL_GRID_LAYOUT["left"],
        right=FULL_GRID_LAYOUT["right"],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILENAME)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_subset_grid_from_payload(payload: dict, state_indices: List[int]) -> str:
    plt.rcParams.update(CORL_SETTINGS)

    states_labels = list(payload["states_labels"])
    gt_next_raw = np.asarray(payload["gt_next_raw"])
    time_axis = np.asarray(payload["time_axis"])
    plot_idxs = np.asarray(payload["plot_idxs"])
    results = payload["results"]

    n_rows = len(state_indices)
    n_cols = len(LOSS_FUNCTIONS)

    fig_w, fig_h = set_size(FIGURE_WIDTH_PT, subplots=(n_rows, n_cols))
    fig_h *= 0.95
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        # sharex=True,
        # sharey="row",
        squeeze=False,
    )

    colors = plt.cm.tab10(np.arange(len(plot_idxs)))
    y_limits = {}
    for si in state_indices:
        gt_all = gt_next_raw[plot_idxs, :, si]
        gt_min, gt_max = float(np.min(gt_all)), float(np.max(gt_all))
        span = max(gt_max - gt_min, 1e-8)
        margin = max(0.1 * span, 1e-3)
        y_limits[si] = (gt_min - margin, gt_max + margin)

    for col, loss_name in enumerate(LOSS_FUNCTIONS):
        mean_pred_raw = np.asarray(results[loss_name]["mean_pred_raw"])
        sample_pred_raw = np.asarray(results[loss_name]["sample_pred_raw"])

        col_title = LOSS_FUNCTION_TITLES.get(loss_name, loss_name)
        axes[0, col].set_title(
            col_title,
            fontsize=7,
            bbox=_title_bbox_for_loss(loss_name),
            pad=3,
        )

        for row, si in enumerate(state_indices):
            ax = axes[row, col]
            y_lo, y_hi = y_limits[si]

            for pi, ci in enumerate(plot_idxs):
                color = colors[pi]
                ax.plot(
                    time_axis,
                    gt_next_raw[ci, :, si],
                    "-",
                    color=color,
                    lw=0.8,
                    alpha=0.90,
                    zorder=4,
                )

                ax.plot(
                    time_axis,
                    mean_pred_raw[ci, :, si],
                    "--",
                    color=color,
                    lw=0.75,
                    alpha=0.85,
                    zorder=3,
                )

                samps_si = sample_pred_raw[ci, :, :, si]
                lo = np.nanpercentile(samps_si, 5, axis=0)
                hi = np.nanpercentile(samps_si, 95, axis=0)
                ax.fill_between(time_axis, lo, hi, color=color, alpha=0.12, zorder=2)

            ax.set_ylim(y_lo, y_hi)
            ax.set_xlim(time_axis[0], time_axis[-1])
            _draw_gray_background_grid(
                ax=ax,
                x_lo=float(time_axis[0]),
                x_hi=float(time_axis[-1]),
                y_lo=float(y_lo),
                y_hi=float(y_hi),
            )
            ax.set_yticks([])
            # ax.set_yticklabels([])
            ax.tick_params(axis="x", labelsize=5)

            if col == 0:
                raw_label = states_labels[si] if states_labels else f"state_{si}"
                label = STATE_LABEL_TO_MATH.get(raw_label, raw_label)
                ax.set_ylabel(label, fontsize=6, rotation=90, labelpad=6, va="center")
            else:
                ax.set_ylabel("")
                # ax.set_yticks([])

            is_bottom = row == (n_rows - 1)
            # ax.set_xticks([time_axis[::21]])
            sns.despine(
                ax=ax,
                left=True,
                bottom=not is_bottom,
                offset=2,
                trim=True,
            )
            if not is_bottom:
                ax.set_xticks([])
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(r"$t$ [s]", fontsize=6, labelpad=2)
            
    # fig.supxlabel(r"$t$ [s]", fontsize=6, y=SUBSET_GRID_LAYOUT["supxlabel_y"])
    fig.subplots_adjust(
        hspace=SUBSET_GRID_LAYOUT["hspace"],
        wspace=SUBSET_GRID_LAYOUT["wspace"],
        top=SUBSET_GRID_LAYOUT["top"],
        bottom=SUBSET_GRID_LAYOUT["bottom"],
        left=SUBSET_GRID_LAYOUT["left"],
        right=SUBSET_GRID_LAYOUT["right"],
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_FILENAME_SUBSET)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _compute_rollout_payload(model_paths: Dict[str, str], config: dict) -> dict:
    first_model_path = model_paths[LOSS_FUNCTIONS[0]]
    _, first_bundle = load_wheelbot_model_bundle(first_model_path, rng_seed=SEED)

    first_model_init = first_bundle["model_init"]
    first_metadata = first_bundle.get("metadata", {})
    state_dim = int(first_model_init["state_dim"])
    action_dim = int(first_model_init["action_dim"])

    trajectories, eval_states_labels, dt, source_desc = _load_trajectories()

    model_states_labels = list(first_metadata.get("states_labels", []))
    states_labels = (
        model_states_labels
        if model_states_labels
        else [f"state_{i}" for i in range(state_dim)]
    )
    dt = float(dt if dt is not None else first_metadata.get("dt", 0.006))

    traj_state_dim = int(np.asarray(trajectories[0]["states"]).shape[1])
    if traj_state_dim != state_dim:
        if STATE_INDEX_OVERRIDE is not None:
            state_indices = list(STATE_INDEX_OVERRIDE)
        elif model_states_labels and eval_states_labels:
            missing = [lab for lab in model_states_labels if lab not in eval_states_labels]
            if missing:
                raise ValueError(
                    "Cannot project trajectory states to model states by label; "
                    f"missing labels in eval trajectories: {missing}"
                )
            state_indices = [eval_states_labels.index(lab) for lab in model_states_labels]
        else:
            raise ValueError(
                f"Trajectory state dim ({traj_state_dim}) != model state dim ({state_dim}), "
                "and no label-based mapping is available. Set STATE_INDEX_OVERRIDE."
            )
        trajectories = _project_trajectories_states(trajectories, state_indices)
        print(
            f"Projected trajectory states from dim {traj_state_dim} to {state_dim} "
            f"using indices {state_indices}"
        )

    candidate_chunks = build_rollout_chunks_from_trajectories(
        trajectories=trajectories,
        rollout_length=ROLLOUT_LENGTH,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    if not candidate_chunks:
        raise ValueError(
            "No usable rollout chunks found "
            f"for rollout_length={ROLLOUT_LENGTH}"
        )

    n_eval = min(N_EVAL_CHUNKS, len(candidate_chunks))
    rng_eval = np.random.default_rng(SEED)
    selected_idx = rng_eval.choice(len(candidate_chunks), size=n_eval, replace=False)
    chunks = [candidate_chunks[i] for i in selected_idx]

    gt_next_raw = np.stack([c[2] for c in chunks]).astype(np.float32)
    time_axis = (np.arange(ROLLOUT_LENGTH) * dt).astype(np.float32)

    n_plot = min(NUM_PLOT_TRAJS, n_eval)
    rng_plot = np.random.default_rng(SEED)
    plot_idxs = rng_plot.choice(n_eval, size=n_plot, replace=False).astype(np.int32)

    results = {}
    for loss_name in LOSS_FUNCTIONS:
        model_path = model_paths[loss_name]
        model, bundle = load_wheelbot_model_bundle(model_path, rng_seed=SEED)
        model_init = bundle["model_init"]
        if int(model_init["state_dim"]) != state_dim or int(model_init["action_dim"]) != action_dim:
            raise ValueError(
                f"Dimension mismatch for {loss_name}: "
                f"({model_init['state_dim']}, {model_init['action_dim']}) vs "
                f"({state_dim}, {action_dim})"
            )

        norm = bundle["normalization"]
        state_mean = np.asarray(norm["state_mean"])
        state_std = np.asarray(norm["state_std"])
        action_mean = np.asarray(norm["action_mean"])
        action_std = np.asarray(norm["action_std"])

        mean_pred_raw, sample_pred_raw = _run_rollouts_for_model(
            model=model,
            chunks=chunks,
            state_mean=state_mean,
            state_std=state_std,
            action_mean=action_mean,
            action_std=action_std,
        )
        results[loss_name] = {
            "mean_pred_raw": np.asarray(mean_pred_raw, dtype=np.float32),
            "sample_pred_raw": np.asarray(sample_pred_raw, dtype=np.float32),
        }

    return {
        "cache_version": 1,
        "config": config,
        "source_desc": source_desc,
        "states_labels": list(states_labels),
        "time_axis": time_axis,
        "gt_next_raw": gt_next_raw,
        "plot_idxs": plot_idxs,
        "results": results,
        "n_eval": int(n_eval),
        "n_candidate_chunks": int(len(candidate_chunks)),
        "n_plot": int(n_plot),
        "n_trajectories": int(len(trajectories)),
    }


def main():
    model_paths = _resolve_model_paths()
    config = _cache_config(model_paths)

    payload = None
    if LOAD_ROLLOUT_CACHE_IF_AVAILABLE and not FORCE_RECOMPUTE_ROLLOUTS:
        payload = _load_rollout_cache(config)

    if payload is None:
        payload = _compute_rollout_payload(model_paths=model_paths, config=config)
        if SAVE_ROLLOUT_CACHE:
            _save_rollout_cache(payload)

    full_out_path = _plot_full_grid_from_payload(payload=payload)

    subset_indices = _resolve_subset_state_indices(list(payload["states_labels"]))
    subset_out_path = _plot_subset_grid_from_payload(
        payload=payload,
        state_indices=subset_indices,
    )

    print(f"Trajectory source: {payload['source_desc']}")
    print("Models:")
    for loss_name in LOSS_FUNCTIONS:
        print(f"  {loss_name}: {model_paths[loss_name]}")
    print(f"Loaded trajectories: {payload['n_trajectories']}")
    print(
        "Chunk sampling matches eval script with "
        f"rollout_length={ROLLOUT_LENGTH}, seed={SEED}"
    )
    print(
        f"Using {payload['n_eval']} chunks (from {payload['n_candidate_chunks']} candidates), "
        f"plotting {payload['n_plot']} trajectories"
    )
    print(f"Saved full grid: {full_out_path}")
    print(f"Saved subset grid: {subset_out_path}")


if __name__ == "__main__":
    main()
