from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from src.models.vbll_model import VBLLModel
from src.training.utils import (
    precompute_true_state_quantities,
    crps_gaussian,
    rollout_diagnostics,
)


def multi_step_elbo_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,         # (d,)
    target_trajectory: jax.Array,     # (H, d)
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,  # unused — kept for API compatibility
    n_samples: int = 16,              # unused — kept for API compatibility
) -> jax.Array:
    """Closed-form multi-step ELBO (no Monte Carlo over W).
 
    Under teacher forcing the per-step NLL is quadratic in W:
 
        ℓ_h(W) = -log N(δ_h; W^T φ_h, Σ_ν)
 
    so the expectation under q(W) = N(μ_W, Σ_W) evaluates analytically:
 
        E_q[ℓ_h] = (d/2)log(2π) + (1/2) Σ_j [ log σ_j²
                    + (1/σ_j²)(  (δ_{h,j} - μ_{w_j}^T φ_h)²    ← squared bias
                               +  φ_h^T Σ_{w_j} φ_h  )  ]       ← epistemic variance
 
    This is mathematically identical to the sampled version
    ``multi_step_elbo_loss_fn`` but has zero estimator variance.
 
    Args:
        model:              VBLLDeltaModel with Gaussian posterior over W.
        initial_state:      x*_0 of shape (d,).
        target_trajectory:  x*_{1:H} of shape (H, d).
        gamma:              Horizon discount γ ∈ (0, 1].
        key:                Unused (present for drop-in compatibility).
        n_samples:          Unused (present for drop-in compatibility).
 
    Returns:
        Scalar loss (negative closed-form ELBO).
    """
    H, d = target_trajectory.shape
 
    # ── Precompute features φ_h at true states (no rollout) ───
    _, target_deltas, all_features = precompute_true_state_quantities(
        model=model, initial_state=initial_state, target_trajectory=target_trajectory,
    )
    # target_deltas: (H, d)   — δ_h = x*_{h+1} - x*_h
    # all_features:  (H, f)   — φ_h = φ_θ(x*_h)
 
    # ── Process noise variance σ_j² ───────────────────────────
    proc_var = jnp.exp(2.0 * model.last_layer.process_noise_log_std.value)  # (d,)
 
    # ── Horizon discount weights ──────────────────────────────
    horizon_weights = gamma ** jnp.arange(H)  # (H,)
 
    # ── Predictive statistics at each true state ──────────────
    # last_layer.predictive(φ) returns N(μ_W^T φ, diag(φ^T Σ_{w_j} φ + σ_j²))
    def predictive_stats(features_h):
        pred = model.last_layer.predictive(features_h)
        return pred.mean, pred.scale ** 2          # (d,), (d,)
 
    pred_means, pred_vars = jax.vmap(predictive_stats)(all_features)
    # pred_means: (H, d)  — E_q[W^T φ_h]  =  μ_W^T φ_h
    # pred_vars:  (H, d)  — φ_h^T Σ_{w_j} φ_h + σ_j²
 
    # ── Decompose into bias² and epistemic variance ───────────
    epistemic_vars = jnp.maximum(pred_vars - proc_var[None, :], 0.0)  # (H, d)
    sq_errors = (target_deltas - pred_means) ** 2                     # (H, d)
 
    # ── Closed-form  E_q[ -log N(δ_h; W^T φ_h, Σ_ν) ]  ──────
    #   = 0.5 * log(2π σ_j²)  +  0.5/σ_j² * (sq_error + epistemic_var)
    expected_nll_hj = (
        0.5 * jnp.log(2.0 * jnp.pi * proc_var[None, :])
        + 0.5 / proc_var[None, :] * (sq_errors + epistemic_vars)
    )  # (H, d)
 
    # ── Weighted sum over horizon and dimensions ──────────────
    expected_nll = jnp.sum(horizon_weights[:, None] * expected_nll_hj)
    return expected_nll
 
 
def multi_step_elbo_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation loss: closed-form ELBO + stochastic rollout diagnostics.
 
    The primary loss (for model selection) is the closed-form negative ELBO
    — identical to training but without gradient noise.  A stochastic rollout
    is performed to compute RMSE / spread diagnostics for monitoring.
 
    Args:
        model:              VBLLDeltaModel.
        initial_state:      x*_0 of shape (d,).
        target_trajectory:  x*_{1:H} of shape (H, d).
        n_samples:          Number of W samples for rollout diagnostics.
        gamma:              Horizon discount γ ∈ (0, 1].
        key:                JAX PRNG key (needed for rollout diagnostics).
 
    Returns:
        (elbo_loss, diagnostics) where elbo_loss is the scalar to minimise.
    """
    if key is None:
        raise ValueError("key required for rollout diagnostics")
 
    H, d = target_trajectory.shape
 
    # ── Closed-form negative ELBO (primary loss) ──────────────
    elbo_loss = multi_step_elbo_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        gamma=gamma,
    )
 
    # ── Decompose for logging ─────────────────────────────────
    _, target_deltas, all_features = precompute_true_state_quantities(
        model=model, initial_state=initial_state, target_trajectory=target_trajectory,
    )
    proc_var = jnp.exp(2.0 * model.last_layer.process_noise_log_std.value)
    horizon_weights = gamma ** jnp.arange(H)
 
    def predictive_stats(features_h):
        pred = model.last_layer.predictive(features_h)
        return pred.mean, pred.scale ** 2
 
    pred_means, pred_vars = jax.vmap(predictive_stats)(all_features)
    epistemic_vars = jnp.maximum(pred_vars - proc_var[None, :], 0.0)
    sq_errors = (target_deltas - pred_means) ** 2
 
    expected_nll_hj = (
        0.5 * jnp.log(2.0 * jnp.pi * proc_var[None, :])
        + 0.5 / proc_var[None, :] * (sq_errors + epistemic_vars)
    )
    expected_nll = jnp.sum(horizon_weights[:, None] * expected_nll_hj)
    kl = model.last_layer._gaussian_kl()
 
    # ── Stochastic rollout for RMSE / spread diagnostics ──────
    W_dist = model.last_layer.W()
    sigma = model.last_layer.process_noise().scale
 
    key_w, key_nu = jax.random.split(key)
    keys_w = jax.random.split(key_w, n_samples)
    keys_nu = jax.random.split(key_nu, n_samples)
 
    def stochastic_rollout(key_w_i, key_nu_i):
        W_sample = W_dist.rsample(key_w_i)
        noise_keys = jax.random.split(key_nu_i, H)
 
        def step_fn(curr_state, nu_key):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            nu = sigma * jax.random.normal(nu_key, curr_state.shape)
            return curr_state + delta + nu, curr_state + delta + nu
 
        _, trajectory = jax.lax.scan(step_fn, initial_state, noise_keys)
        return trajectory
 
    sampled_trajectories = jax.vmap(stochastic_rollout)(keys_w, keys_nu)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)
 
    diagnostics = {
        "val_loss": elbo_loss,
        "elbo": -elbo_loss,
        "expected_log_likelihood": -expected_nll,
        "kl": kl,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return elbo_loss, diagnostics

# ==================================================================
# Multi-step Factored CRPS
# ==================================================================

def multi_step_factored_crps_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,         # (d,)
    target_trajectory: jax.Array,     # (H, d)
    n_samples: int = 16,
    gamma: float = 1.0,
    key: jax.Array = None,
) -> jax.Array:
    """Multi-step loss with CRPS as the loss.

    Factored Markov form — conditions on true observed states x*_h,
    exactly like multi_step_elbo_loss_fn, but replaces the Gaussian
    log-likelihood with the Continuous Ranked Probability Score.

    Args:
        initial_state: x*_0 of shape (d,)
        target_trajectory: x*_{1:H} of shape (H, d)
        n_samples: Number of W samples (S)
        gamma: Horizon discount gamma ∈ (0, 1]
        key: JAX PRNG key

    Returns:
        Scalar loss.
    """
    if key is None:
        raise ValueError("key required for CRPS sampling")

    H, d = target_trajectory.shape

    # ── Precompute at true states ─────────────────────────────
    _, target_deltas, all_features = precompute_true_state_quantities(
        model=model, target_trajectory=target_trajectory, initial_state=initial_state,
    )

    # ── Process noise std (learnable) ─────────────────────────
    proc_scale = jnp.exp(
        model.last_layer.process_noise_log_std.value,
    )  # (d,)

    # ── Horizon discount weights ──────────────────────────────
    horizon_weights = gamma ** jnp.arange(H)  # (H,)

    # ── Sample W and compute E_q[ Σ_h γ^h CRPS_h ] ──────────
    W_dist = model.last_layer.W()
    keys_w = jax.random.split(key, n_samples)

    def crps_for_sample(key_i):
        W_s = W_dist.rsample(key_i)                        # (d, p)

        # Predicted deltas at true states: (H, p) @ (p, d) → (H, d)
        pred_deltas = all_features @ W_s.T                 # (H, d)

        # Per-step CRPS of N(pred, process_var) evaluated at δ*_h
        crps_per_h = crps_gaussian(
            pred_deltas,
            jnp.broadcast_to(proc_scale, pred_deltas.shape),
            target_deltas,
        )  # (H,)

        return jnp.sum(horizon_weights * crps_per_h)

    crps_vals = jax.vmap(crps_for_sample)(keys_w)          # (S,)
    expected_crps = jnp.mean(crps_vals)
    return expected_crps

def multi_step_factored_crps_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: jax.Array = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for GVI-CRPS: loss + rollout diagnostics.

    Primary metric: factored CRPS loss (same as training objective).
    Also runs stochastic rollouts for RMSE / spread / calibration.

    Args:
        initial_state: x*_0 of shape (d,)
        target_trajectory: x*_{1:H} of shape (H, d)
        n_samples: Number of W samples (S)
        gamma: Horizon discount gamma ∈ (0, 1]
        key: JAX PRNG key
        alpha: Rényi divergence order

    Returns:
        (crps_loss, diagnostics)
    """
    if key is None:
        raise ValueError("key required for CRPS val sampling")

    H, d = target_trajectory.shape
    key_crps, key_rollout = jax.random.split(key)

    # ── Primary loss ──────────────────────────────────────────
    crps_loss = multi_step_factored_crps_loss_fn(
        model=model, initial_state=initial_state, target_trajectory=target_trajectory,
        n_samples=n_samples, gamma=gamma, key=key_crps
    )

    # ── Per-horizon CRPS breakdown ────────────────────────────
    _, target_deltas, all_features = precompute_true_state_quantities(
        model=model, initial_state=initial_state, target_trajectory=target_trajectory,
    )
    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()

    keys_diag = jax.random.split(key_crps, n_samples)

    def crps_per_step(key_i):
        W_s = W_dist.rsample(key_i)
        pred_deltas = all_features @ W_s.T
        return crps_gaussian(
            pred_deltas,
            jnp.broadcast_to(proc_scale, pred_deltas.shape),
            target_deltas,
        )  # (H,)

    crps_all = jax.vmap(crps_per_step)(keys_diag)            # (S, H)
    crps_per_horizon = jnp.mean(crps_all, axis=0)            # (H,)

    # ── Stochastic rollouts for RMSE / spread ─────────────────
    sigma = model.last_layer.process_noise().scale
    key_w_r, key_nu_r = jax.random.split(key_rollout)
    keys_w_r = jax.random.split(key_w_r, n_samples)
    keys_nu_r = jax.random.split(key_nu_r, n_samples)

    def stochastic_rollout(key_w_i, key_nu_i):
        W_sample = W_dist.rsample(key_w_i)
        noise_keys = jax.random.split(key_nu_i, H)

        def step_fn(curr_state, nu_key):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            nu = sigma * jax.random.normal(nu_key, curr_state.shape)
            next_state = curr_state + delta + nu
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, noise_keys)
        return trajectory

    sampled_trajectories = jax.vmap(stochastic_rollout)(
        keys_w_r, keys_nu_r,
    )  # (S, H, d)
    
    # ── Ensemble RMSE and spread ──────────────────────────
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)
    divergence = model.last_layer._gaussian_kl()

    total_loss = crps_loss + model.last_layer.regularization_weight * divergence

    diagnostics = {
        "val_loss": total_loss,
        "crps_loss": crps_loss,
        "crps": jnp.mean(crps_per_horizon),
        "crps_per_horizon": crps_per_horizon,
        "divergence": divergence,
        "process_noise_std": proc_scale,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }

    return total_loss, diagnostics


# ==================================================================
# Multi-step KDE
# ==================================================================

def multi_step_kde_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,         # (d,)
    target_trajectory: jax.Array,     # (H, d)
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Multi-step KDE log-likelihood loss (Ivanovic, Harrison, Pavone 2023).

    Evaluates the predictive density at the ground-truth trajectory via
    kernel density estimation over S stochastic rollouts, with an
    accumulated-covariance bandwidth that grows linearly with horizon.

    where:
        V_h = h · Σ_ν           (accumulated process-noise covariance)
        x̂_h^{(s)} is the h-th state of the s-th stochastic rollout
        x*_h is the h-th ground-truth state

    The log-sum-exp trick is used for numerical stability, as
    recommended by the paper.

    Args:
        initial_state: x*_0 of shape (d,)
        target_trajectory: x*_{1:H} of shape (H, d)
        n_samples: Number of trajectory samples S
        gamma: Horizon discount γ ∈ (0, 1] applied as weight per step
        key: JAX PRNG key

    Returns:
        Scalar loss (negative KDE log-likelihood + KL).
    """
    if key is None:
        raise ValueError("key required for KDE sampling")

    H, d = target_trajectory.shape
    W_dist = model.last_layer.W()
    # Process noise std: σ_j for j = 1, ..., d
    sigma = model.last_layer.process_noise().scale          # (d,)
    proc_var = sigma ** 2                                  # (d,)  = Σ_ν diagonal

    # ── Stochastic rollouts ───────────────────────────────────────
    key_w, key_nu = jax.random.split(key)
    keys_w = jax.random.split(key_w, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(carry, _):
            curr_state = carry
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(
            step_fn,
            initial_state,
            xs=jnp.arange(H),  # optional but often cleaner
        )
        return trajectory
    
    # sampled_trajectories: (S, H, d)
    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    # ── KDE log-likelihood at each horizon step ───────────────────
    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)   # (H,) : 1, ..., H

    # Residuals: (S, H, d)
    residuals = sampled_trajectories - target_trajectory[None, :, :]

    # Mahalanobis per (s, h): Σ_j r²_{s,h,j} / σ_j²
    mahal = jnp.sum(residuals ** 2 / proc_var[None, None, :], axis=-1)  # (S, H)

    # log-det of V_h = Σ_j [log h + log σ_j²] = d·log(h) + Σ_j log σ_j²
    sum_log_var = jnp.sum(jnp.log(proc_var))              # scalar
    logdet_Vh = d * jnp.log(h_indices) + sum_log_var       # (H,)

    # Per-sample, per-step log Gaussian density
    log_gauss = -0.5 * (
        d * jnp.log(2.0 * jnp.pi)
        + logdet_Vh[None, :]                               # (1, H)
        + mahal / h_indices[None, :]                       # (S, H) — divide by h
    )  # (S, H)

    # log-sum-exp over samples → log (1/S) Σ_s N(·)
    # = logsumexp_s(log N) - log(S)
    log_S = jnp.log(jnp.float32(n_samples))
    log_kde_per_h = jax.scipy.special.logsumexp(log_gauss, axis=0) - log_S  # (H,)

    # ── Horizon weighting ─────────────────────────────────────────
    horizon_weights = gamma ** jnp.arange(H)               # (H,)

    # Weighted negative log-likelihood
    kde_loss = -jnp.mean(horizon_weights * log_kde_per_h) # / jnp.sum(horizon_weights)

    # ── divergence regularisation ─────────────────────────────────────────
    # divergence = model.last_layer._gaussian_kl()

    return kde_loss # + model.last_layer.regularization_weight * divergence


def multi_step_kde_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion to multi_step_kde_loss_fn.

    Returns (kde_loss, diagnostics) with rollout RMSE / spread / etc.
    """
    if key is None:
        raise ValueError("key required for KDE val sampling")

    H, d = target_trajectory.shape
    key_kde, key_rollout = jax.random.split(key)

    # ── Primary loss (same as training) ───────────────────────────
    kde_loss = multi_step_kde_loss_fn(
        model=model, 
        initial_state=initial_state, 
        target_trajectory=target_trajectory, 
        n_samples=n_samples, 
        gamma=gamma,
        key=key_kde,
    )

    # ── Rollout diagnostics ───────────────────────────────────────
    W_dist = model.last_layer.W()
    key_w_r, key_nu_r = jax.random.split(key_rollout)
    keys_w_r = jax.random.split(key_w_r, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(carry, _):
            curr_state = carry
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(
            step_fn,
            initial_state,
            xs=jnp.arange(H),  # optional but often cleaner
        )
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w_r)

    # ── Ensemble RMSE and spread ──────────────────────────
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": kde_loss,
        "kde_nll": kde_loss,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return kde_loss, diagnostics


# ==================================================================
# Multi-step Autoregressive CRPS (arCRPS — closed-form Gaussian)
# ==================================================================

def gaussian_log_score(
    mean: jax.Array,
    scale: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Gaussian log score as a minimisation loss: negative log likelihood."""
    scale = jnp.maximum(scale, 1e-6)
    var = scale ** 2
    nll_per_dim = 0.5 * (
        jnp.log(2.0 * jnp.pi * var) + (target - mean) ** 2 / var
    )
    return jnp.sum(nll_per_dim, axis=-1)


def multi_step_ar_crps_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,         # (d,)
    target_trajectory: jax.Array,     # (H, d)
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Multi-step autoregressive CRPS with accumulated-variance bandwidth.

    For each W^(s) ~ q(W), runs a single deterministic rollout (no
    process noise) and scores each step h against x*_h using the
    closed-form Gaussian CRPS with scale sqrt(h) · σ_j.

    The linear-variance approximation models the accumulated aleatoric
    uncertainty after h consecutive autoregressive steps as h · Σ_ν,
    giving per-dimension scale sqrt(h) · σ_j.  This is exact when
    φ is locally linear and a good approximation when σ is small
    relative to the dynamics scale.

    This is a valid Gibbs VI loss: ℓ(W) is a deterministic function of
    a single weight draw — no inner Monte Carlo loop is needed.  Cost
    is S deterministic rollouts.

    Args:
        initial_state: x*_0 of shape (d,)
        target_trajectory: x*_{1:H} of shape (H, d)
        n_samples: Number of W samples (S)
        gamma: Horizon discount γ ∈ (0, 1]
        key: JAX PRNG key

    Returns:
        Scalar loss (expected arCRPS + KL).
    """
    if key is None:
        raise ValueError("key required for arCRPS sampling")

    H, d = target_trajectory.shape
    W_dist = model.last_layer.W()

    # Process noise std: σ_j for j = 1, ..., d
    proc_scale = jnp.exp(
        model.last_layer.process_noise_log_std.value,
    )  # (d,)

    # ── Deterministic rollouts (no process noise) ─────────────
    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, _):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, jnp.arange(H))
        return trajectory  # (H, d)

    # sampled_trajectories: (S, H, d)
    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    # ── Accumulated-variance CRPS at each horizon step ────────
    # Per-sample marginal variance at step h is h · σ_j²,
    # so the scale (std) is sqrt(h) · σ_j.
    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)    # (H,) : 1, ..., H
    # Accumulated scale: (H, d)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def crps_for_sample(traj_s):
        """CRPS for a single deterministic rollout against ground truth."""
        crps_per_h = crps_gaussian(
            traj_s,                   # mu: (H, d)
            accumulated_scale,        # scale: (H, d)
            target_trajectory,        # obs: (H, d)
        )  # (H,)
        return crps_per_h

    # (S, H)
    crps_all = jax.vmap(crps_for_sample)(sampled_trajectories)

    # ── Horizon-weighted sum, averaged over samples ───────────
    horizon_weights = gamma ** jnp.arange(H)                # (H,)
    # E_q[Σ_h γ^h CRPS_h]
    expected_crps = jnp.mean(
        jnp.sum(crps_all * horizon_weights[None, :], axis=-1)
    )

    # # ── KL regularisation ─────────────────────────────────────
    # divergence = model.last_layer._gaussian_kl()

    return expected_crps #  + model.last_layer.regularization_weight * divergence


def multi_step_ar_crps_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for arCRPS: loss + rollout diagnostics.

    Primary metric: autoregressive CRPS loss (same as training).
    Also computes per-horizon CRPS breakdown, RMSE, and spread.

    Args:
        initial_state: x*_0 of shape (d,)
        target_trajectory: x*_{1:H} of shape (H, d)
        n_samples: Number of W samples (S)
        gamma: Horizon discount γ ∈ (0, 1]
        key: JAX PRNG key

    Returns:
        (ar_crps_loss, diagnostics)
    """
    if key is None:
        raise ValueError("key required for arCRPS val sampling")

    H, d = target_trajectory.shape
    key_loss, key_rollout = jax.random.split(key)

    # ── Primary loss (same as training) ───────────────────────
    ar_crps_loss = multi_step_ar_crps_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        n_samples=n_samples,
        gamma=gamma,
        key=key_loss,
    )

    # ── Per-horizon CRPS breakdown (from deterministic rollouts) ──
    W_dist = model.last_layer.W()
    proc_scale = jnp.exp(
        model.last_layer.process_noise_log_std.value,
    )
    keys_w = jax.random.split(key_loss, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, _):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, jnp.arange(H))
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)  # (S, H, d)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def crps_per_step(traj_s):
        return crps_gaussian(traj_s, accumulated_scale, target_trajectory)

    crps_all = jax.vmap(crps_per_step)(sampled_trajectories)  # (S, H)
    crps_per_horizon = jnp.mean(crps_all, axis=0)            # (H,)

    # ── Ensemble RMSE and spread (from deterministic rollouts) ─
    rmse_per_h, spread_per_h = rollout_diagnostics(
        sampled_trajectories, target_trajectory,
    )

    divergence = model.last_layer._gaussian_kl()

    total_loss = ar_crps_loss + model.last_layer.regularization_weight * divergence

    diagnostics = {
        "val_loss": total_loss,
        "ar_crps_loss": ar_crps_loss,
        "crps": jnp.mean(crps_per_horizon),
        "crps_per_horizon": crps_per_horizon,
        "divergence": divergence,
        "process_noise_std": proc_scale,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }

    return total_loss, diagnostics


# ==================================================================
# Multi-step Autoregressive ELBO (arELBO — accumulated Gaussian log score)
# ==================================================================

def multi_step_ar_elbo_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Multi-step autoregressive ELBO using the arCRPS predictive family.

    This is the log-score equivalent of ``multi_step_ar_crps_loss_fn``:
    for each sampled W, roll out the deterministic trajectory and score
    x*_h under N(mu_h(W), h Sigma_nu).
    """
    if key is None:
        raise ValueError("key required for arELBO sampling")

    H, _ = target_trajectory.shape
    W_dist = model.last_layer.W()
    proc_scale = jnp.exp(model.last_layer.process_noise_log_std.value)
    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, _):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, jnp.arange(H))
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def log_score_for_sample(traj_s):
        return gaussian_log_score(traj_s, accumulated_scale, target_trajectory)

    log_scores_all = jax.vmap(log_score_for_sample)(sampled_trajectories)
    horizon_weights = gamma ** jnp.arange(H)
    expected_nll = jnp.mean(
        jnp.sum(log_scores_all * horizon_weights[None, :], axis=-1)
    )
    return expected_nll


def multi_step_ar_elbo_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for arELBO: loss + rollout diagnostics."""
    if key is None:
        raise ValueError("key required for arELBO val sampling")

    H, _ = target_trajectory.shape
    key_loss, key_diag = jax.random.split(key)

    ar_elbo_loss = multi_step_ar_elbo_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        n_samples=n_samples,
        gamma=gamma,
        key=key_loss,
    )

    W_dist = model.last_layer.W()
    proc_scale = jnp.exp(model.last_layer.process_noise_log_std.value)
    keys_w = jax.random.split(key_diag, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, _):
            features = model.feature_extractor(curr_state)
            delta = (W_sample @ features[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, jnp.arange(H))
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def log_score_per_step(traj_s):
        return gaussian_log_score(traj_s, accumulated_scale, target_trajectory)

    log_scores_all = jax.vmap(log_score_per_step)(sampled_trajectories)
    log_score_per_horizon = jnp.mean(log_scores_all, axis=0)
    rmse_per_h, spread_per_h = rollout_diagnostics(
        sampled_trajectories, target_trajectory,
    )

    divergence = model.last_layer._gaussian_kl()
    total_loss = ar_elbo_loss + model.last_layer.regularization_weight * divergence

    diagnostics = {
        "val_loss": total_loss,
        "ar_elbo_loss": ar_elbo_loss,
        "elbo": -ar_elbo_loss,
        "log_score": jnp.mean(log_score_per_horizon),
        "log_score_per_horizon": log_score_per_horizon,
        "divergence": divergence,
        "process_noise_std": proc_scale,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }

    return total_loss, diagnostics
