"""
Post-training evaluation: quick metrics logged to the run summary.

Called at the end of runs/train.py so key trajectory metrics are visible
without having to run the full evaluation script.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.distance import pdist

from src.training.metrics import (
    compute_dtw_distance,
    compute_mmd,
    compute_energy_distance,
)


# ── Environment rollouts ──────────────────────────────────────────────────────

def _rollout_det(env, x0: np.ndarray, n_steps: int) -> np.ndarray:
    if hasattr(env, 'rollout_deterministic'):
        return env.rollout_deterministic(x0, n_steps)
    rng = np.random.RandomState(0)
    traj = [x0.copy()]
    cur = x0.copy()
    for _ in range(n_steps):
        cur = env.step(cur, rng)
        traj.append(cur.copy())
    return np.array(traj)


def _rollout_noisy(env, x0: np.ndarray, n_steps: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    traj = [x0.copy()]
    cur = x0.copy()
    for _ in range(n_steps):
        cur = env.step(cur, rng)
        traj.append(cur.copy())
    return np.array(traj)


# ── Coverage ──────────────────────────────────────────────────────────────────

def _empirical_coverage(samples_all, noisy_gt_all, nominal: float = 0.90) -> float:
    """Marginal empirical coverage averaged over test states and dimensions."""
    alpha_lo = (1.0 - nominal) / 2.0 * 100.0
    alpha_hi = (1.0 + nominal) / 2.0 * 100.0
    coverages = []
    for model_arr, gt_arr in zip(samples_all, noisy_gt_all):
        model_arr = np.array(model_arr)   # (n_samples, T+1, D)
        gt_arr    = np.array(gt_arr)      # (n_samples, T+1, D)
        lo = np.percentile(model_arr, alpha_lo, axis=0)
        hi = np.percentile(model_arr, alpha_hi, axis=0)
        inside = (gt_arr >= lo[None]) & (gt_arr <= hi[None])
        # mean over GT samples, time steps (exclude t=0), dimensions
        coverages.append(inside[:, 1:, :].mean())
    return float(np.mean(coverages))


def _median_pairwise_distance(samples: np.ndarray) -> float:
    """Median nonzero Euclidean distance without a quadratic 3-D broadcast."""
    distances = pdist(np.asarray(samples), metric="euclidean")
    positive = distances[distances > 0.0]
    return float(np.median(positive)) if positive.size else 1.0


# ── Main entry point ──────────────────────────────────────────────────────────

def run_post_train_eval(
    model,
    det_env,
    noisy_env,
    initial_state_bounds: np.ndarray,
    logger,
    rollout_steps: int = 40,
    n_test_states: int = 30,
    n_noisy_samples: int = 30,
    seed: int = 42,
) -> dict:
    """
    Roll out *model* on randomly sampled initial states, compute trajectory
    metrics, and log them to the run summary via *logger*.

    Returns a dict of {metric_name: mean_value} for optional printing.
    """
    print("\n" + "=" * 80)
    print("Post-training evaluation")
    print("=" * 80)
    print(f"  n_test_states={n_test_states}, n_noisy_samples={n_noisy_samples}, "
          f"rollout_steps={rollout_steps}")

    rng = np.random.RandomState(seed + 7777)
    test_inits = [
        jnp.array(rng.uniform(initial_state_bounds[:, 0], initial_state_bounds[:, 1]))
        for _ in range(n_test_states)
    ]

    # Ground truth
    gt_all = [_rollout_det(det_env, np.array(x0), rollout_steps) for x0 in test_inits]

    noisy_gt_all = []
    for idx, x0 in enumerate(test_inits):
        noisy_gt_all.append([
            _rollout_noisy(noisy_env, np.array(x0), rollout_steps,
                           seed + idx * 1000 + si)
            for si in range(n_noisy_samples)
        ])

    # Model rollouts
    mean_trajs, samples_all = [], []
    for idx, x0 in enumerate(test_inits):
        mean_trajs.append(model.rollout_mean_trajectory(x0, rollout_steps))
        keys = jax.random.split(
            jax.random.PRNGKey(seed + idx * 1000), n_noisy_samples
        )
        samps = jax.vmap(lambda k: model.simulate(x0, rollout_steps, k))(keys)
        samples_all.append([np.array(s) for s in samps])

    # ── Metrics ───────────────────────────────────────────────────────────────
    gt_final = np.stack([g[-1] for g in gt_all])
    m_final  = np.stack([np.array(m)[-1] for m in mean_trajs])
    final_l2 = np.linalg.norm(m_final - gt_final, axis=1)

    cumul_l2 = np.array([
        np.sum(np.linalg.norm(np.array(m)[1:] - np.array(g)[1:], axis=1))
        for m, g in zip(mean_trajs, gt_all)
    ])

    dtw_dist = np.array([
        compute_dtw_distance(np.array(m), np.array(g))
        for m, g in zip(mean_trajs, gt_all)
    ])

    def _flatten(samps):
        return np.array([s.flatten() for s in samps])

    m_flat  = [_flatten(samples_all[i]) for i in range(n_test_states)]
    gt_flat = [_flatten(noisy_gt_all[i]) for i in range(n_test_states)]

    all_gt_flat = np.concatenate(gt_flat, axis=0)
    sub_idx = np.random.choice(len(all_gt_flat),
                               size=min(2000, len(all_gt_flat)), replace=False)
    gt_sub  = all_gt_flat[sub_idx]
    shared_bw = _median_pairwise_distance(gt_sub)

    mmd_vals = np.array([
        compute_mmd(m_flat[i], gt_flat[i], bandwidth=shared_bw)
        for i in range(n_test_states)
    ])
    ed_vals = np.array([
        compute_energy_distance(m_flat[i], gt_flat[i])
        for i in range(n_test_states)
    ])

    cov_90 = _empirical_coverage(samples_all, noisy_gt_all, nominal=0.90)
    cov_50 = _empirical_coverage(samples_all, noisy_gt_all, nominal=0.50)

    summary = {
        "eval/final_l2_mean":    float(np.mean(final_l2)),
        "eval/final_l2_median":  float(np.median(final_l2)),
        "eval/cumul_l2_mean":    float(np.mean(cumul_l2)),
        "eval/dtw_mean":         float(np.mean(dtw_dist)),
        "eval/mmd_mean":         float(np.mean(mmd_vals)),
        "eval/energy_dist_mean": float(np.mean(ed_vals)),
        "eval/coverage_90":      cov_90,
        "eval/coverage_50":      cov_50,
    }

    logger.log_summary(summary)

    print("  Results:")
    for k, v in summary.items():
        print(f"    {k}: {v:.4f}")

    return summary
