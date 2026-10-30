"""VSLC paper figure: rollout metrics per loss as box plots.

Choose the baselines in BASELINES below (comment out a line to drop it; the
order is the column/legend order). Models whose training diverged (last 25% of
the loss history non-finite or > 1e6) are left out.

Usage:
    PATH=/Library/TeX/texbin:$PATH python plots/results_vslc.py
    # output: figures/<sweep_name>/vslc_paper_metrics_boxes.<FIG_FORMAT>
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plots.settings import (  # noqa: E402
    CORL_SETTINGS_FULL_WIDTH,
    FIG_FORMAT,
    SWEEP_NAME,
    figures_dir,
    set_size,
)
from src.models import VBLLDeltaModel  # noqa: E402
from src.models.vbll_model import VBLLModel  # noqa: E402
from src.training import compute_dtw_distance, compute_energy_distance, compute_mmd  # noqa: E402
from src.utils.model_management import get_activation_fn, load_model_and_config  # noqa: E402
from src.utils.normalization import NormalizedModelWrapper  # noqa: E402


plt.rcParams.update(CORL_SETTINGS_FULL_WIDTH)

# ---------------------------------------------------------------------------
# Baselines. Comment out a line to drop it; the order is the column/legend order.
#   (training loss = model folder name, display name, color)
# ---------------------------------------------------------------------------
BASELINES_SPEC = [
    # ("single_step_elbo",               "s-NLL (no HB)", "tab:pink"),
    ("single_step_elbo_hbatch",        "s-NLL",     "tab:red"),
    ("multi_step_elbo",                "m-NLL",     "tab:blue"),
    ("multi_step_factored_crps",       "m-CRPS",    "tab:orange"),
    ("multi_step_ar_elbo",             "m-arNLL",   "tab:gray"),
    ("multi_step_ar_crps",             "m-arCRPS",  "tab:purple"),
    ("multi_step_kde",                 "m-KDE",     "tab:green"),
]

NAME = SWEEP_NAME
DIVERGENCE_TAIL_FRACTION = 0.25
DIVERGENCE_THRESHOLD = 1e6

BASELINES = [name for _, name, _ in BASELINES_SPEC]
BASELINE_COLORS = {name: color for _, name, color in BASELINES_SPEC}
BASELINES_NAMES = {name: rf"\texttt{{{name}}}" for _, name, _ in BASELINES_SPEC}

METRICS = [
    ("cumulative_l2", r"cum. L2 ($\downarrow$)"),
    ("dtw", r"DTW ($\downarrow$)"),
    ("mmd2", r"MMD$^2$ ($\downarrow$)"),
    ("energy", r"ED ($\downarrow$)"),
    ("terminal_mmd2", r"term. MMD$^2$ ($\downarrow$)"),
    ("terminal_energy", r"term. ED ($\downarrow$)"),
]

TRAINING_LOSS_TO_LABEL = {loss: name for loss, name, _ in BASELINES_SPEC}

LABEL_ALIASES = {}

SWEEP_MODELS = [(loss, name) for loss, name, _ in BASELINES_SPEC]


def _set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _canonical_label(label: str) -> str:
    if label in LABEL_ALIASES:
        return LABEL_ALIASES[label]
    return label


def reconstruct_model(saved: dict):
    cfg = saved["config"]
    n_tr = saved["training_metadata"]["n_train_samples"] or 1

    state_dim = len(cfg.environment.initial_state_bounds)
    hidden_dims = tuple([cfg.model.model.hidden_dim] * cfg.model.model.num_hidden_layers)
    reg_weight = cfg.training.regularization_weight / n_tr
    activation = get_activation_fn(cfg.model.model.activation_fn)

    if "VBLLDeltaModel" in cfg.model.model._target_:
        model_class = VBLLDeltaModel
    else:
        model_class = VBLLModel

    model = model_class(
        input_dim=state_dim,
        output_dim=state_dim,
        hidden_dims=hidden_dims,
        feature_dim=cfg.model.model.feature_dim,
        activation_fn=activation,
        regularization_weight=reg_weight,
        prior_scale=cfg.model.model.prior_scale,
        wishart_scale=cfg.model.model.wishart_scale,
        rngs=nnx.Rngs(0),
    )
    nnx.update(model, saved["model_state"])
    return model


def _rollout_det(env, x0, n_steps: int) -> np.ndarray:
    if hasattr(env, "rollout_deterministic"):
        return env.rollout_deterministic(np.array(x0), n_steps)
    rng = np.random.RandomState(0)
    traj = [np.array(x0)]
    cur = np.array(x0)
    for _ in range(n_steps):
        cur = env.step(cur, rng)
        traj.append(cur.copy())
    return np.array(traj)


def _rollout_noisy(env, x0, n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    traj = [np.array(x0)]
    cur = np.array(x0)
    for _ in range(n_steps):
        cur = env.step(cur, rng)
        traj.append(cur.copy())
    return np.array(traj)


def _style_metric_axis(ax, title):
    ax.set_title("")
    ax.set_xlabel(title, labelpad=6)
    ax.set_ylim(bottom=0.0)
    ax.grid(axis="y", alpha=0.3)


def _create_metrics_figure():
    x_size, y_size = set_size(397, subplots=(1, 1))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(x_size, 0.35 * y_size), sharex=False)
    return fig, axes


def plot_metric_boxes(metric_data, save_path: Path, baselines_present: List[str]):
    fig, axes = _create_metrics_figure()
    rng = np.random.RandomState(123)

    for idx, (metric_key, metric_label) in enumerate(METRICS):
        ax = axes[idx]
        data = [metric_data[metric_key][b] for b in baselines_present]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.68,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.9},
            capprops={"linewidth": 0.9},
        )
        for patch, baseline in zip(bp["boxes"], baselines_present):
            patch.set_facecolor(BASELINE_COLORS[baseline])
            patch.set_alpha(0.55)
        for i, (ind_data, baseline) in enumerate(zip(data, baselines_present)):
            x = rng.normal(i + 1, 0.04, size=len(ind_data))
            ax.scatter(x, ind_data, alpha=0.3, s=5, color=BASELINE_COLORS[baseline], zorder=3)
        ax.set_xticks([])
        _style_metric_axis(ax, metric_label)
        if metric_key in ("energy", ):
            ax.set_ylim(0.0, 2.5)
        if metric_key in ("terminal_energy", ):
            ax.set_ylim(0.0, 0.5)

    legend_handles = [
        plt.Line2D([0], [0], color=BASELINE_COLORS[b], lw=6, label=BASELINES_NAMES[b]) for b in baselines_present
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(baselines_present),
        handlelength=0.8,
        frameon=False,
        columnspacing=0.7,
    )
    fig.subplots_adjust(wspace=0.5, left=0.04, right=0.99, top=0.85, bottom=0.16)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def resolve_model_specs(args) -> List[Tuple[Path, str]]:
    if args.model_paths:
        paths = [Path(p) for p in args.model_paths]
        if args.labels and len(args.labels) != len(paths):
            raise ValueError("--labels must match the number of --model-paths.")
        labels = args.labels if args.labels else [None] * len(paths)
        specs = []
        for p, lbl in zip(paths, labels):
            specs.append((p, _canonical_label(lbl) if lbl else None))
        return specs

    if not args.sweep_name:
        raise ValueError("Provide either --model-paths or --sweep-name.")

    specs = []
    for loss_name, label in SWEEP_MODELS:
        p = (
            Path(args.results_root)
            / args.sweep_name
            / "vbll_delta"
            / "variable_speed_limit_cycle"
            / loss_name
            / str(args.train_seed)
            / "vbll_model.pkl"
        )
        specs.append((p, label))
    return specs


def _loss_history_diverged(losses) -> bool:
    arr = np.asarray(losses, dtype=float)
    if arr.size == 0:
        return False
    tail = arr[int((1.0 - DIVERGENCE_TAIL_FRACTION) * arr.size):]
    return bool(np.any(~np.isfinite(tail)) or np.any(np.abs(tail) > DIVERGENCE_THRESHOLD))


def load_models(specs: List[Tuple[Path, str]]):
    loaded = []
    first_saved_cfg = None

    for i, (path, label) in enumerate(specs):
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        print(f"Loading model {i + 1}/{len(specs)}: {path}")
        saved = load_model_and_config(str(path))
        if _loss_history_diverged(saved.get("train_losses", [])) or _loss_history_diverged(
            saved.get("val_losses", [])
        ):
            print(f"  -> diverged during training, skipped: {path}")
            continue
        model = reconstruct_model(saved)

        inferred_label = TRAINING_LOSS_TO_LABEL.get(saved["training_metadata"]["training_loss"])
        final_label = _canonical_label(label or inferred_label or saved["training_metadata"]["training_loss"])

        in_norm = saved.get("input_normalization")
        out_norm = saved.get("output_normalization")
        if in_norm and in_norm.get("enabled"):
            model = NormalizedModelWrapper(model, in_norm, out_norm)

        if first_saved_cfg is None:
            first_saved_cfg = saved["config"]

        loaded.append(
            {
                "path": path,
                "label": final_label,
                "model": model,
                "saved": saved,
            }
        )
        print(f"  -> label: {final_label}")

    return loaded, first_saved_cfg


def build_envs(first_saved_cfg):
    env_oc = first_saved_cfg.environment.environment
    env_dict = OmegaConf.to_container(env_oc, resolve=True)
    noise_std = float(env_dict.get("noise_std", 0.01))

    env = hydra.utils.instantiate(env_dict)
    det_env = hydra.utils.instantiate({**env_dict, "noise_std": 0.0})
    noisy_env = hydra.utils.instantiate({**env_dict, "noise_std": noise_std})

    if not hasattr(env, "angular_velocity_field"):
        raise ValueError("Loaded environment is not VSLC-like (missing angular_velocity_field).")
    return env, det_env, noisy_env


def evaluate_models(
    models_data: List[dict],
    det_env,
    noisy_env,
    initial_state_bounds: np.ndarray,
    rollout_steps: int,
    n_test_states: int,
    n_noisy_samples: int,
    seed: int,
):
    rng = np.random.RandomState(seed + 9999)
    test_init_all = [
        jnp.array(rng.uniform(initial_state_bounds[:, 0], initial_state_bounds[:, 1]))
        for _ in range(n_test_states)
    ]

    gt_all = [_rollout_det(det_env, np.array(x0), rollout_steps) for x0 in test_init_all]

    noisy_gt_all = []
    for idx, x0 in enumerate(test_init_all):
        noisy_gt_all.append(
            [
                _rollout_noisy(noisy_env, np.array(x0), rollout_steps, seed + 5000 + idx * 1000 + si)
                for si in range(n_noisy_samples)
            ]
        )

    def _flatten_samples(samples_list):
        return np.array([s.flatten() for s in samples_list])

    gt_flat = [_flatten_samples(noisy_gt_all[i]) for i in range(n_test_states)]
    all_gt_flat = np.concatenate(gt_flat, axis=0)
    subsample_idx = rng.choice(len(all_gt_flat), size=min(500, len(all_gt_flat)), replace=False)
    gt_sub = all_gt_flat[subsample_idx]
    dists = pdist(gt_sub)
    shared_bw = float(np.median(dists[dists > 0]))

    gt_term_flat = np.concatenate([np.array([traj[-1] for traj in noisy_gt_all[i]]) for i in range(n_test_states)], axis=0)
    term_sub_idx = rng.choice(len(gt_term_flat), size=min(500, len(gt_term_flat)), replace=False)
    term_sub = gt_term_flat[term_sub_idx]
    term_dists = pdist(term_sub)
    terminal_bw = float(np.median(term_dists[term_dists > 0]))

    for mi, mentry in enumerate(models_data):
        print(f"Rolling out model {mi + 1}/{len(models_data)}: {mentry['label']}")
        model = mentry["model"]

        mean_trajs = [model.rollout_mean_trajectory(x0, rollout_steps) for x0 in test_init_all]

        samples_all = []
        for idx, x0 in enumerate(test_init_all):
            keys = jax.random.split(jax.random.PRNGKey(seed + (mi + 1) * 2000 + idx * 1000), n_noisy_samples)
            samps = jax.vmap(lambda k: model.simulate(x0, rollout_steps, k))(keys)
            samples_all.append([np.array(s) for s in samps])

        cumul_l2 = np.array(
            [np.sum(np.linalg.norm(np.array(m)[1:] - np.array(g)[1:], axis=1)) for m, g in zip(mean_trajs, gt_all)]
        )
        dtw_dist = np.array([compute_dtw_distance(np.array(m), np.array(g)) for m, g in zip(mean_trajs, gt_all)])

        m_flat = [_flatten_samples(samples_all[i]) for i in range(n_test_states)]
        mmd_vals = np.array([compute_mmd(m_flat[i], gt_flat[i], bandwidth=shared_bw) for i in range(n_test_states)])
        ed_vals = np.array([compute_energy_distance(m_flat[i], gt_flat[i]) for i in range(n_test_states)])

        terminal_mmd = np.array(
            [
                compute_mmd(
                    np.array([s[-1] for s in samples_all[i]]),
                    np.array([s[-1] for s in noisy_gt_all[i]]),
                    bandwidth=terminal_bw,
                )
                for i in range(n_test_states)
            ]
        )
        terminal_ed = np.array(
            [
                compute_energy_distance(
                    np.array([s[-1] for s in samples_all[i]]),
                    np.array([s[-1] for s in noisy_gt_all[i]]),
                )
                for i in range(n_test_states)
            ]
        )

        mentry.update(
            {
                "mean_trajs": mean_trajs,
                "samples_all": samples_all,
                "cumulative_l2": cumul_l2,
                "dtw": dtw_dist,
                "mmd2": mmd_vals,
                "energy": ed_vals,
                "terminal_mmd2": terminal_mmd,
                "terminal_energy": terminal_ed,
            }
        )

    return test_init_all


def build_metric_data(models_data: List[dict], baselines_present: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
    by_label = {m["label"]: m for m in models_data}
    metric_data = {}
    for key, _ in METRICS:
        metric_data[key] = {}
        for b in baselines_present:
            metric_data[key][b] = np.asarray(by_label[b][key])
    return metric_data


def parse_args():
    parser = argparse.ArgumentParser(description="Generate VSLC paper plots from trained model checkpoints.")

    parser.add_argument("--model-paths", nargs="+", default=None, help="Explicit model checkpoint paths.")
    parser.add_argument("--labels", nargs="+", default=None, help="Optional labels for --model-paths.")

    parser.add_argument("--sweep-name", type=str, default=NAME, help="Sweep name under results root.")
    parser.add_argument("--train-seed", type=int, default=42, help="Training seed used in checkpoint folder names.")
    parser.add_argument("--results-root", type=str, default="results", help="Root containing sweep directories.")

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save outputs (default: figures/<sweep_name>, see plots/settings.py).",
    )
    parser.add_argument("--prefix", type=str, default="vslc_paper", help="Output filename prefix.")

    parser.add_argument("--rollout-steps", type=int, default=60, help="Rollout horizon for metrics.")
    parser.add_argument("--n-test-states", type=int, default=100, help="Number of test initial states.")
    parser.add_argument("--n-noisy-samples", type=int, default=100, help="Samples per state for stochastic metrics.")
    parser.add_argument("--eval-seed", type=int, default=42, help="Evaluation random seed.")

    return parser.parse_args()


def main():
    args = parse_args()
    _set_eval_seed(args.eval_seed)

    specs = resolve_model_specs(args)
    models_data, first_saved_cfg = load_models(specs)

    _, det_env, noisy_env = build_envs(first_saved_cfg)
    bounds = np.array(first_saved_cfg.environment.initial_state_bounds)

    evaluate_models(
        models_data=models_data,
        det_env=det_env,
        noisy_env=noisy_env,
        initial_state_bounds=bounds,
        rollout_steps=args.rollout_steps,
        n_test_states=args.n_test_states,
        n_noisy_samples=args.n_noisy_samples,
        seed=args.eval_seed,
    )

    by_label = {m["label"]: m for m in models_data}
    baselines_present = [b for b in BASELINES if b in by_label]
    if not baselines_present:
        raise ValueError(
            "No recognized VSLC baseline labels found. Use labels from: "
            + ", ".join(BASELINES)
        )

    metric_data = build_metric_data(models_data, baselines_present)
    save_dir = Path(args.save_dir) if args.save_dir else figures_dir(args.sweep_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    boxes_path = save_dir / f"{args.prefix}_metrics_boxes.{FIG_FORMAT}"

    plot_metric_boxes(metric_data, boxes_path, baselines_present)

    print(f"Saved plot: {boxes_path}")


if __name__ == "__main__":
    main()
