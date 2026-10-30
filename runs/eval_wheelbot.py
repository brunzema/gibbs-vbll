"""Evaluate all Wheelbot models of a sweep: one-step and multi-step XY-odometry metrics.

For every run <results_root>/<sweep>/<env>/<loss>_<H>/<seed>/ this writes
  one_step_metrics_sweep_eval.json  MSE / NLL / CRPS on the normalized test transitions
  multistep_xy_metrics.json         final L2, cumulative L2, DTW (mean rollout) and energy
                                    distance (sample rollouts) of the XY odometry per horizon
and a manifest of the shared settings and rollout chunks next to the runs.

Rollouts start from the same test chunks for every model. They are generated once at
ROLLOUT_LENGTH and the metrics are reported on the sub-horizons HORIZONS. Data only, the
results table is made by plots/make_wheelbot_latex_table.py.

Usage:
    uv run python runs/eval_wheelbot.py <wb_sweep_name>
    uv run python runs/eval_wheelbot.py <wb_sweep_name> --methods multi_step_ar_crps --quick
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf
from wheelbot_dataset import Dataset

from src.training import compute_dtw_distance, compute_energy_distance
from src.training.utils import crps_gaussian
from src.utils.wheelbot_model_io import load_wheelbot_model_bundle
from src.wheelbot.racing import (
    FIELDS_ACTIONS,
    FIELDS_STATES,
    ensure_split_meta_files,
    load_prediction_split,
    make_cut_fn,
)

OmegaConf.register_new_resolver("half", lambda x: int(x) // 2, replace=True)

# Settings of the paper's Wheelbot table (same as the original evaluation script).
ROLLOUT_LENGTH = 1000
HORIZONS = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_EVAL_CHUNKS = 100  # overridden by loss.evaluation.n_eval_chunks of the first run
N_SAMPLES = 50  # sample rollouts per chunk for the energy distance
BATCH_SIZE = 16  # chunks per rollout batch
EVAL_SEED = 42  # chunk selection and sampling keys, fixed across training seeds
# Reduced setting for a quick local check (--quick); numbers are not comparable to the paper.
QUICK = dict(rollout_length=100, horizons=[50, 100], n_eval_chunks=10, n_samples=10)

WHEEL_RADIUS = 0.032  # driving wheel radius [m]
INITIAL_HEADING_DEG = 0.0
YAW_FIELD = "/q_yrp/yaw"
WHEEL_ANGLE_FIELD = "/q_DR/drive_wheel"
YAW_VEL_FIELD = "/dq_yrp/yaw_vel"
WHEEL_VEL_FIELD = "/dq_DR/drive_wheel"

MODEL_FILENAMES = ["wheelbot_racing_vbll_model.pkl", "wheelbot_racing_multistep_vbll_model.pkl"]
LOSS_TO_BASELINE = {
    "single_step_elbo": "s-NLL",
    "single_step_elbo_hbatch": "s-NLL-HB",
    "multi_step_elbo": "m-NLL",
    "multi_step_factored_crps": "m-CRPS",
    "multi_step_kde": "m-KDE",
    "multi_step_ar_elbo": "m-arNLL",
    "multi_step_ar_crps": "m-arCRPS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "wb_sweep_name",
        help="Wheelbot sweep under <results-root>/.",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results/wheelbot_dynamics"))
    parser.add_argument("--env-name", default="wheelbot_racing")
    parser.add_argument("--seed", type=int, default=42, help="Training seed, i.e. the run folder <seed>.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Only these runs: loss names (multi_step_ar_crps) or run folders (multi_step_ar_crps_40).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/wheelbot_racing"))
    parser.add_argument("--quick", action="store_true", help=f"Reduced setting for a local check: {QUICK}.")
    return parser.parse_args()


def _to_jsonable(obj: Any):
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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)


def discover_runs(env_dir: Path, seed: int, methods: List[str] | None) -> List[Dict[str, Any]]:
    """All <loss>_<H>/<seed>/ runs of the sweep that contain a model bundle."""
    runs = []
    for folder in sorted(d for d in env_dir.iterdir() if d.is_dir()):
        loss = folder.name.rsplit("_", 1)[0]
        if methods and folder.name not in methods and loss not in methods:
            continue
        run_dir = folder / str(seed)
        model_paths = [run_dir / name for name in MODEL_FILENAMES if (run_dir / name).exists()]
        if not model_paths:
            print(f"WARNING: skipping {run_dir}: no model bundle")
            continue
        runs.append({
            "run": folder.name,
            "loss": loss,
            "baseline": LOSS_TO_BASELINE.get(loss, loss),
            "run_dir": run_dir,
            "model_path": model_paths[0],
        })
    return runs


# ─── Statistics ──────────────────────────────────────────────────────────────

def summary_stats(values: np.ndarray, values_key: str | None = None) -> Dict[str, Any]:
    """count, mean, median and quantiles over the finite values, plus the raw values under values_key."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    arr = raw[np.isfinite(raw)]
    stats: Dict[str, Any] = {"count": int(arr.size)}
    for name, fn in [
        ("mean", np.mean), ("median", np.median),
        ("q05", lambda a: np.quantile(a, 0.05)), ("q25", lambda a: np.quantile(a, 0.25)),
        ("q75", lambda a: np.quantile(a, 0.75)), ("q95", lambda a: np.quantile(a, 0.95)),
    ]:
        stats[name] = float(fn(arr)) if arr.size else None
    if values_key:
        stats[values_key] = raw.tolist()
    return stats


def first_nonfinite_step(arr: np.ndarray) -> np.ndarray:
    """First time index (axis -2) with a non-finite entry, T+1 if there is none."""
    bad = ~np.isfinite(arr).all(axis=-1)
    return np.where(bad.any(axis=-1), np.argmax(bad, axis=-1), bad.shape[-1]).astype(int)


def divergence_summary(first_bad: np.ndarray, rollout_length: int) -> Dict[str, Any]:
    flat = np.asarray(first_bad, dtype=int).reshape(-1)
    diverged = flat <= rollout_length
    return {
        "n_total": int(flat.size),
        "n_diverged": int(diverged.sum()),
        "fraction_diverged": float(diverged.mean()) if flat.size else 0.0,
        "first_bad_step": summary_stats(flat[diverged]),
    }


# ─── Test data and odometry ──────────────────────────────────────────────────

def _to_2d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    return a[:, None] if a.ndim == 1 else a


def load_test_experiments(test_dir: Path, cut_fn, state_fields, action_fields) -> List[Dict[str, Any]]:
    """Full test experiments with states, actions, yaw and drive-wheel angle (finite steps only)."""
    ds = Dataset(str(test_dir)).map(cut_fn)
    experiments = []
    for group_name in sorted(ds.groups):
        for exp_idx, exp in enumerate(ds.groups[group_name].experiments):
            arrays = [
                _to_2d(exp.to_numpy(state_fields)),
                _to_2d(exp.to_numpy(action_fields)),
                _to_2d(exp.to_numpy([YAW_FIELD])),
                _to_2d(exp.to_numpy([WHEEL_ANGLE_FIELD])),
            ]
            min_len = min(a.shape[0] for a in arrays)
            if min_len < 2:
                continue
            arrays = [a[:min_len] for a in arrays]
            finite = np.all([np.isfinite(a).all(axis=1) for a in arrays], axis=0)
            states, actions, yaw, q_drive = [a[finite] for a in arrays]
            if len(yaw) < 2:
                continue
            experiments.append({
                "name": f"{group_name}/exp_{exp_idx}",
                "states": states,
                "actions": actions,
                "yaw": yaw[:, 0],
                "q_drive": q_drive[:, 0],
            })
    return experiments


def build_rollout_candidates(experiments: List[Dict[str, Any]], horizon: int) -> List[Dict[str, Any]]:
    """Every start index with `horizon` actions and horizon+1 ground-truth yaw / wheel angles."""
    candidates = []
    for exp in experiments:
        max_start = min(exp["actions"].shape[0] - horizon, exp["states"].shape[0] - 1)
        for start in range(max_start + 1):
            candidates.append({
                "name": exp["name"],
                "start": start,
                "init_state": exp["states"][start],
                "actions": exp["actions"][start:start + horizon],
                "yaw": exp["yaw"][start:start + horizon + 1],
                "q_drive": exp["q_drive"][start:start + horizon + 1],
            })
    return candidates


def integrate_angles(pred_states_raw: np.ndarray, idx: int, x0: np.ndarray, dt: float) -> np.ndarray:
    """Euler-integrate the velocity state[..., idx] from x0; (N, [S,] T+1, D) -> (N, [S,] T+1)."""
    inc = pred_states_raw[..., :-1, idx] * dt
    x0 = np.asarray(x0, dtype=float).reshape(x0.shape + (1,) * (inc.ndim - x0.ndim))
    return x0 + np.concatenate([np.zeros(inc.shape[:-1] + (1,)), np.cumsum(inc, axis=-1)], axis=-1)


def odometry_xy(yaw: np.ndarray, wheel_angle: np.ndarray) -> np.ndarray:
    """Planar odometry from yaw and drive-wheel angle along the last axis -> (..., T+1, 2)."""
    heading = np.unwrap(yaw, axis=-1) - yaw[..., :1] + math.radians(INITIAL_HEADING_DEG)
    displacement = np.diff(wheel_angle, axis=-1, prepend=wheel_angle[..., :1]) * WHEEL_RADIUS
    x = np.cumsum(np.cos(heading) * displacement, axis=-1)
    y = np.cumsum(np.sin(heading) * displacement, axis=-1)
    return np.stack([x, y], axis=-1)


# ─── Metrics ─────────────────────────────────────────────────────────────────

def one_step_metrics(model, X: jax.Array, Y: jax.Array, state_dim: int) -> Dict[str, Any]:
    """Per-sample MSE, NLL and Gaussian CRPS of the one-step predictive (normalized space)."""
    preds = np.asarray(jax.vmap(model)(X))
    mse = np.mean((preds - np.asarray(Y)) ** 2, axis=-1)
    nll = np.asarray(jax.vmap(model.val_loss_fn)(X, Y))

    def _pred_stats(xi):
        delta_dist = model.last_layer.predictive(model.feature_extractor(xi))
        return xi[:state_dim] + delta_dist.mean, delta_dist.scale

    means, scales = jax.vmap(_pred_stats)(X)
    crps = np.asarray(crps_gaussian(means, scales, Y))
    return {
        name: summary_stats(vals, "values_by_sample")
        for name, vals in [("mse_norm", mse), ("nll_norm", nll), ("crps_norm", crps)]
    }


def rollouts(model, init_states: np.ndarray, actions: np.ndarray, n_samples: int):
    """Mean rollouts (N, T+1, D) and sample rollouts (N, S, T+1, D) in normalized space."""
    n_chunks, horizon = actions.shape[:2]
    state_dim = init_states.shape[1]
    mean_all = np.zeros((n_chunks, horizon + 1, state_dim), dtype=np.float32)
    samples_all = np.zeros((n_chunks, n_samples, horizon + 1, state_dim), dtype=np.float32)

    mean_fn = jax.jit(jax.vmap(model.rollout_mean_trajectory, in_axes=(0, 0)))

    @jax.jit
    def samples_fn(x0s, acts, keys):
        return jax.vmap(lambda x0, a, ks: jax.vmap(lambda k: model.simulate(x0, a, k))(ks))(x0s, acts, keys)

    for start in range(0, n_chunks, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_chunks)
        x0_b, acts_b = jnp.array(init_states[start:end]), jnp.array(actions[start:end])
        mean_all[start:end] = np.asarray(mean_fn(x0_b, acts_b))
        keys = jax.random.split(jax.random.PRNGKey(EVAL_SEED + 10_000 + start), (end - start) * n_samples)
        samples_all[start:end] = np.asarray(samples_fn(x0_b, acts_b, keys.reshape(end - start, n_samples, 2)))
    return mean_all, samples_all


def horizon_metrics(
    xy_gt: np.ndarray,  # (N, T+1, 2)
    xy_mean: np.ndarray,  # (N, T+1, 2)
    xy_samples: np.ndarray,  # (N, S, T+1, 2)
    mean_first_bad: np.ndarray,  # (N,)
    sample_first_bad: np.ndarray,  # (N, S)
    horizons: List[int],
) -> Dict[str, Dict[str, Any]]:
    """XY metrics per horizon. A rollout counts at horizon h only if it is finite up to step h;
    the mean metrics of a diverged chunk are NaN, the energy distance uses the finite samples."""
    n_chunks, n_samples = xy_samples.shape[:2]
    out = {}
    for h in horizons:
        vals = {k: np.full(n_chunks, np.nan) for k in ("final_l2", "cum_l2", "dtw", "energy_dist")}
        for i in range(n_chunks):
            gt = xy_gt[i, :h + 1]
            if mean_first_bad[i] > h:
                mean = xy_mean[i, :h + 1]
                vals["final_l2"][i] = np.linalg.norm(mean[-1] - gt[-1])
                vals["cum_l2"][i] = np.sum(np.linalg.norm(mean[1:] - gt[1:], axis=-1))
                vals["dtw"][i] = compute_dtw_distance(mean, gt)
            valid = sample_first_bad[i] > h
            if valid.any():
                samples = xy_samples[i, valid, :h + 1].reshape(int(valid.sum()), -1)
                vals["energy_dist"][i] = compute_energy_distance(samples, gt.reshape(1, -1))

        n_mean = int(np.isfinite(vals["final_l2"]).sum())
        n_energy = int(np.isfinite(vals["energy_dist"]).sum())
        n_sample_used = int((sample_first_bad > h).sum())
        out[str(h)] = {
            "n_chunks": n_chunks,
            "n_chunks_used_mean_metrics": n_mean,
            "n_chunks_diverged_mean_by_horizon": n_chunks - n_mean,
            "n_chunks_used_energy": n_energy,
            "n_chunks_diverged_energy_by_horizon": n_chunks - n_energy,
            "n_sample_rollouts_total": n_chunks * n_samples,
            "n_sample_rollouts_used_energy": n_sample_used,
            "n_sample_rollouts_diverged_by_horizon": n_chunks * n_samples - n_sample_used,
            **{k: summary_stats(v, "values_by_chunk") for k, v in vals.items()},
        }
        print(f"    H={h}: mean metrics on {n_mean}/{n_chunks} chunks, "
              f"energy on {n_sample_used}/{n_chunks * n_samples} samples")
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    env_dir = args.results_root / args.wb_sweep_name / args.env_name
    if not env_dir.exists():
        raise FileNotFoundError(f"Sweep env directory not found: {env_dir}")
    runs = discover_runs(env_dir, args.seed, args.methods)
    if not runs:
        raise RuntimeError(f"No runs with a model bundle under {env_dir}/*/{args.seed}")

    # Data settings come from the first run (identical for all runs of a sweep).
    ref_cfg = OmegaConf.to_container(OmegaConf.load(runs[0]["run_dir"] / "resolved_config.yaml"), resolve=True)
    dt, trim_start, trim_end = (float(ref_cfg["env"][k]) for k in ("dt", "trim_start", "trim_end"))
    test_dir = args.data_root / ref_cfg["env"]["test_dir"]
    rollout_length, horizons, n_samples = ROLLOUT_LENGTH, HORIZONS, N_SAMPLES
    n_eval_chunks = int(ref_cfg.get("loss", {}).get("evaluation", {}).get("n_eval_chunks", N_EVAL_CHUNKS))
    if args.quick:
        rollout_length, horizons = QUICK["rollout_length"], QUICK["horizons"]
        n_eval_chunks, n_samples = QUICK["n_eval_chunks"], QUICK["n_samples"]

    ensure_split_meta_files(args.data_root)
    cut_fn = make_cut_fn(dt=dt, trim_start=trim_start, trim_end=trim_end)

    _, ref_bundle = load_wheelbot_model_bundle(str(runs[0]["model_path"]), rng_seed=EVAL_SEED)
    state_fields = list(ref_bundle.get("metadata", {}).get("states_labels", FIELDS_STATES))
    action_fields = list(ref_bundle.get("metadata", {}).get("actions_labels", FIELDS_ACTIONS))
    state_dim = len(state_fields)
    yaw_vel_idx, wheel_vel_idx = state_fields.index(YAW_VEL_FIELD), state_fields.index(WHEEL_VEL_FIELD)

    print(f"Sweep {args.wb_sweep_name} ({env_dir}), JAX devices: {jax.devices()}")
    print(f"rollout_length={rollout_length}, horizons={horizons}, n_eval_chunks={n_eval_chunks}, "
          f"n_samples={n_samples}, dt={dt}, trim={trim_start}/{trim_end}{' [QUICK]' if args.quick else ''}")
    for r in runs:
        print(f"  {r['run']} [{r['baseline']}]: {r['model_path']}")

    # One-step test transitions and shared rollout chunks, the same for all models.
    te_s, te_a, te_ns, _ = load_prediction_split(test_dir, cut_fn)
    te_s, te_a, te_ns = (np.asarray(a, dtype=float) for a in (te_s, te_a, te_ns))
    experiments = load_test_experiments(test_dir, cut_fn, state_fields, action_fields)
    candidates = build_rollout_candidates(experiments, rollout_length)
    if not candidates:
        raise RuntimeError(f"No test chunks of length {rollout_length} in {test_dir}")
    chosen = np.random.default_rng(EVAL_SEED).choice(
        len(candidates), size=min(n_eval_chunks, len(candidates)), replace=False
    )
    chunks = [candidates[i] for i in sorted(chosen.tolist())]
    print(f"Using {len(chunks)} rollout chunks (from {len(candidates)} candidates), "
          f"{te_s.shape[0]} one-step test transitions.")

    init_states_raw = np.stack([c["init_state"] for c in chunks])
    actions_raw = np.stack([c["actions"] for c in chunks])
    yaw0 = np.array([c["yaw"][0] for c in chunks])
    q_drive0 = np.array([c["q_drive"][0] for c in chunks])
    xy_gt = odometry_xy(np.stack([c["yaw"] for c in chunks]), np.stack([c["q_drive"] for c in chunks]))

    settings = {
        "rollout_length": rollout_length,
        "horizons": horizons,
        "n_eval_chunks": len(chunks),
        "n_samples": n_samples,
        "dt": dt,
        "trim_start": trim_start,
        "trim_end": trim_end,
        "wheel_radius": WHEEL_RADIUS,
        "initial_heading_deg": INITIAL_HEADING_DEG,
        "yaw_field": YAW_FIELD,
        "wheel_angle_field": WHEEL_ANGLE_FIELD,
        "yaw_vel_field": YAW_VEL_FIELD,
        "wheel_vel_field": WHEEL_VEL_FIELD,
        "seed": EVAL_SEED,
        "quick": args.quick,
    }
    manifest_models = []
    for r in runs:
        t0 = time.time()
        print(f"\nEvaluating {r['run']} [{r['baseline']}] ...")
        model, bundle = load_wheelbot_model_bundle(str(r["model_path"]), rng_seed=EVAL_SEED)
        norm = {k: np.asarray(v, dtype=float) for k, v in bundle["normalization"].items()}
        s_mean, s_std = norm["state_mean"], norm["state_std"]
        a_mean, a_std = norm["action_mean"], norm["action_std"]
        header = {"strategy": r["loss"], "baseline": r["baseline"], "model_path": r["model_path"]}

        X = jnp.concatenate([jnp.array((te_s - s_mean) / s_std), jnp.array((te_a - a_mean) / a_std)], axis=-1)
        Y = jnp.array((te_ns - s_mean) / s_std)
        _write_json(r["run_dir"] / "one_step_metrics_sweep_eval.json", {
            **header,
            "n_test_samples": te_s.shape[0],
            "metrics": one_step_metrics(model, X, Y, state_dim),
        })

        mean_norm, samples_norm = rollouts(
            model, (init_states_raw - s_mean) / s_std, (actions_raw - a_mean) / a_std, n_samples
        )
        mean_raw = mean_norm * s_std + s_mean
        samples_raw = samples_norm * s_std + s_mean
        xy_mean = odometry_xy(
            integrate_angles(mean_raw, yaw_vel_idx, yaw0, dt),
            integrate_angles(mean_raw, wheel_vel_idx, q_drive0, dt),
        )
        xy_samples = odometry_xy(
            integrate_angles(samples_raw, yaw_vel_idx, yaw0[:, None], dt),
            integrate_angles(samples_raw, wheel_vel_idx, q_drive0[:, None], dt),
        )
        # A rollout is valid up to its first non-finite step in state or XY space.
        mean_first_bad = np.minimum(first_nonfinite_step(mean_raw), first_nonfinite_step(xy_mean))
        sample_first_bad = np.minimum(first_nonfinite_step(samples_raw), first_nonfinite_step(xy_samples))
        divergence = {
            "mean_chunks": divergence_summary(mean_first_bad, rollout_length),
            "sample_rollouts": divergence_summary(sample_first_bad, rollout_length),
        }
        print(f"  diverged: mean {divergence['mean_chunks']['n_diverged']}/{len(chunks)}, "
              f"samples {divergence['sample_rollouts']['n_diverged']}/{len(chunks) * n_samples}")

        _write_json(r["run_dir"] / "multistep_xy_metrics.json", {
            **header,
            "settings": settings,
            "divergence": divergence,
            "horizon_metrics": horizon_metrics(
                xy_gt, xy_mean, xy_samples, mean_first_bad, sample_first_bad, horizons
            ),
        })
        print(f"  wrote metrics to {r['run_dir']} ({time.time() - t0:.0f} s)")
        manifest_models.append({**header, "run": r["run"], "run_dir": r["run_dir"], "divergence": divergence})

    _write_json(env_dir / "sweep_metrics_xy_manifest.json", {
        "sweep_name": args.wb_sweep_name,
        "env_name": args.env_name,
        "test_split_dir": test_dir,
        "train_seed": args.seed,
        "settings": settings,
        "shared_rollouts": {
            "candidates_total": len(candidates),
            "entries": [{"name": c["name"], "start": c["start"]} for c in chunks],
        },
        "models": manifest_models,
    })
    print(f"\nManifest -> {env_dir / 'sweep_metrics_xy_manifest.json'}")


if __name__ == "__main__":
    main()
