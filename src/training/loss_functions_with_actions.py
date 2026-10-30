from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from src.models.vbll_model import VBLLModel
from src.training.utils import rollout_diagnostics, crps_gaussian
from src.training.loss_functions import gaussian_log_score


def _precompute_true_state_action_quantities(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Precompute teacher-forced inputs, deltas, and features for state-action models."""
    input_states = jnp.concatenate([initial_state[None], target_trajectory[:-1]], axis=0)
    target_deltas = target_trajectory - input_states
    state_actions = jnp.concatenate([input_states, action_sequence], axis=-1)
    all_features = jax.vmap(model.feature_extractor)(state_actions)
    return input_states, target_deltas, all_features


def multi_step_factored_crps_with_actions_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Teacher-forced factorized CRPS for action-conditioned dynamics."""
    if key is None:
        raise ValueError("key required for CRPS sampling")

    H = target_trajectory.shape[0]
    _, target_deltas, all_features = _precompute_true_state_action_quantities(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
    )

    proc_scale = model.process_noise_stdv
    horizon_weights = gamma ** jnp.arange(H)
    W_dist = model.last_layer.W()
    keys_w = jax.random.split(key, n_samples)

    def crps_for_sample(k):
        W_s = W_dist.rsample(k)
        pred_deltas = all_features @ W_s.T
        crps_per_h = crps_gaussian(
            pred_deltas,
            jnp.broadcast_to(proc_scale, pred_deltas.shape),
            target_deltas,
        )
        return jnp.sum(horizon_weights * crps_per_h)

    return jnp.mean(jax.vmap(crps_for_sample)(keys_w))


def multi_step_factored_crps_with_actions_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for action-conditioned factorized CRPS."""
    if key is None:
        raise ValueError("key required for CRPS validation sampling")

    H = target_trajectory.shape[0]
    key_crps, key_rollout = jax.random.split(key)

    crps_loss = multi_step_factored_crps_with_actions_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
        n_samples=n_samples,
        gamma=gamma,
        key=key_crps,
    )

    _, target_deltas, all_features = _precompute_true_state_action_quantities(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
    )

    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()
    keys_diag = jax.random.split(key_crps, n_samples)

    def crps_per_step(k):
        W_s = W_dist.rsample(k)
        pred_deltas = all_features @ W_s.T
        return crps_gaussian(
            pred_deltas,
            jnp.broadcast_to(proc_scale, pred_deltas.shape),
            target_deltas,
        )

    crps_all = jax.vmap(crps_per_step)(keys_diag)
    crps_per_horizon = jnp.mean(crps_all, axis=0)

    key_w_r, key_nu_r = jax.random.split(key_rollout)
    keys_w_r = jax.random.split(key_w_r, n_samples)
    keys_nu_r = jax.random.split(key_nu_r, n_samples)
    sigma = model.last_layer.process_noise().scale

    def stochastic_rollout(key_w_i, key_nu_i):
        W_sample = W_dist.rsample(key_w_i)
        noise_keys = jax.random.split(key_nu_i, H)

        def step_fn(curr_state, xs):
            action_h, nu_key = xs
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            nu = sigma * jax.random.normal(nu_key, curr_state.shape)
            nxt = curr_state + delta + nu
            return nxt, nxt

        _, trajectory = jax.lax.scan(step_fn, initial_state, (action_sequence, noise_keys))
        return trajectory

    sampled_trajectories = jax.vmap(stochastic_rollout)(keys_w_r, keys_nu_r)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": crps_loss,
        "crps": jnp.mean(crps_per_horizon),
        "crps_per_horizon": crps_per_horizon,
        "divergence": model.last_layer._gaussian_kl(),
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return crps_loss, diagnostics


def multi_step_ar_crps_with_actions_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Autoregressive CRPS with accumulated process-noise variance for action-conditioned dynamics."""
    if key is None:
        raise ValueError("key required for arCRPS sampling")

    H = target_trajectory.shape[0]
    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()
    horizon_weights = gamma ** jnp.arange(H)
    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)  # (S, H, d)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def crps_for_sample(traj_s):
        return crps_gaussian(traj_s, accumulated_scale, target_trajectory)

    crps_all = jax.vmap(crps_for_sample)(sampled_trajectories)  # (S, H)
    return jnp.mean(jnp.sum(crps_all * horizon_weights[None, :], axis=-1))


def multi_step_ar_crps_with_actions_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for action-conditioned autoregressive CRPS."""
    if key is None:
        raise ValueError("key required for arCRPS validation sampling")

    H = target_trajectory.shape[0]
    key_loss, key_diag = jax.random.split(key)

    ar_crps_loss = multi_step_ar_crps_with_actions_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
        n_samples=n_samples,
        gamma=gamma,
        key=key_loss,
    )

    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()
    keys_w = jax.random.split(key_diag, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)  # (S, H, d)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def crps_per_step(traj_s):
        return crps_gaussian(traj_s, accumulated_scale, target_trajectory)

    crps_all = jax.vmap(crps_per_step)(sampled_trajectories)
    crps_per_horizon = jnp.mean(crps_all, axis=0)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": ar_crps_loss,
        "crps": jnp.mean(crps_per_horizon),
        "crps_per_horizon": crps_per_horizon,
        "divergence": model.last_layer._gaussian_kl(),
        "process_noise_std": proc_scale,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return ar_crps_loss, diagnostics


def multi_step_ar_elbo_with_actions_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Autoregressive ELBO with accumulated process-noise variance."""
    if key is None:
        raise ValueError("key required for arELBO sampling")

    H = target_trajectory.shape[0]
    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()
    horizon_weights = gamma ** jnp.arange(H)
    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def log_score_for_sample(traj_s):
        return gaussian_log_score(traj_s, accumulated_scale, target_trajectory)

    log_scores_all = jax.vmap(log_score_for_sample)(sampled_trajectories)
    return jnp.mean(jnp.sum(log_scores_all * horizon_weights[None, :], axis=-1))


def multi_step_ar_elbo_with_actions_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for action-conditioned autoregressive ELBO."""
    if key is None:
        raise ValueError("key required for arELBO validation sampling")

    H = target_trajectory.shape[0]
    key_loss, key_diag = jax.random.split(key)

    ar_elbo_loss = multi_step_ar_elbo_with_actions_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
        n_samples=n_samples,
        gamma=gamma,
        key=key_loss,
    )

    proc_scale = model.process_noise_stdv
    W_dist = model.last_layer.W()
    keys_w = jax.random.split(key_diag, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    accumulated_scale = jnp.sqrt(h_indices[:, None]) * proc_scale[None, :]

    def log_score_per_step(traj_s):
        return gaussian_log_score(traj_s, accumulated_scale, target_trajectory)

    log_scores_all = jax.vmap(log_score_per_step)(sampled_trajectories)
    log_score_per_horizon = jnp.mean(log_scores_all, axis=0)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": ar_elbo_loss,
        "ar_elbo_loss": ar_elbo_loss,
        "elbo": -ar_elbo_loss,
        "log_score": jnp.mean(log_score_per_horizon),
        "log_score_per_horizon": log_score_per_horizon,
        "divergence": model.last_layer._gaussian_kl(),
        "process_noise_std": proc_scale,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return ar_elbo_loss, diagnostics


def multi_step_elbo_with_actions_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
    n_samples: int = 16,
) -> jax.Array:
    """Closed-form action-conditioned multi-step ELBO (exact expectation under q(W))."""
    del key, n_samples

    H = target_trajectory.shape[0]
    _, target_deltas, all_features = _precompute_true_state_action_quantities(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
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
    return jnp.sum(horizon_weights[:, None] * expected_nll_hj)


def multi_step_elbo_with_actions_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for exact action-conditioned multi-step ELBO."""
    if key is None:
        raise ValueError("key required for rollout diagnostics")

    H = target_trajectory.shape[0]
    elbo_loss = multi_step_elbo_with_actions_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
        gamma=gamma,
    )

    _, target_deltas, all_features = _precompute_true_state_action_quantities(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
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

    W_dist = model.last_layer.W()
    sigma = model.last_layer.process_noise().scale
    key_w, key_nu = jax.random.split(key)
    keys_w = jax.random.split(key_w, n_samples)
    keys_nu = jax.random.split(key_nu, n_samples)

    def stochastic_rollout(key_w_i, key_nu_i):
        W_sample = W_dist.rsample(key_w_i)
        noise_keys = jax.random.split(key_nu_i, H)

        def step_fn(curr_state, xs):
            action_h, nu_key = xs
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            nu = sigma * jax.random.normal(nu_key, curr_state.shape)
            next_state = curr_state + delta + nu
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, (action_sequence, noise_keys))
        return trajectory

    sampled_trajectories = jax.vmap(stochastic_rollout)(keys_w, keys_nu)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": elbo_loss,
        "elbo": -elbo_loss,
        "expected_log_likelihood": -expected_nll,
        "kl": model.last_layer._gaussian_kl(),
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return elbo_loss, diagnostics


def multi_step_kde_with_actions_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> jax.Array:
    """Action-conditioned multi-step KDE log-likelihood objective."""
    if key is None:
        raise ValueError("key required for KDE sampling")

    H, d = target_trajectory.shape
    W_dist = model.last_layer.W()
    sigma = model.last_layer.process_noise().scale
    proc_var = sigma ** 2

    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)

    h_indices = jnp.arange(1, H + 1, dtype=jnp.float32)
    residuals = sampled_trajectories - target_trajectory[None, :, :]
    mahal = jnp.sum(residuals ** 2 / proc_var[None, None, :], axis=-1)

    sum_log_var = jnp.sum(jnp.log(proc_var))
    logdet_Vh = d * jnp.log(h_indices) + sum_log_var

    log_gauss = -0.5 * (
        d * jnp.log(2.0 * jnp.pi)
        + logdet_Vh[None, :]
        + mahal / h_indices[None, :]
    )

    log_S = jnp.log(jnp.float32(n_samples))
    log_kde_per_h = jax.scipy.special.logsumexp(log_gauss, axis=0) - log_S
    horizon_weights = gamma ** jnp.arange(H)
    return -jnp.mean(horizon_weights * log_kde_per_h)


def multi_step_kde_with_actions_val_loss_fn(
    model: VBLLModel,
    initial_state: jax.Array,
    target_trajectory: jax.Array,
    action_sequence: jax.Array,
    n_samples: int = 16,
    gamma: float = 1.0,
    key: Optional[jax.Array] = None,
) -> Tuple[jax.Array, dict]:
    """Validation companion for action-conditioned KDE."""
    if key is None:
        raise ValueError("key required for KDE validation sampling")

    H = target_trajectory.shape[0]
    kde_loss = multi_step_kde_with_actions_loss_fn(
        model=model,
        initial_state=initial_state,
        target_trajectory=target_trajectory,
        action_sequence=action_sequence,
        n_samples=n_samples,
        gamma=gamma,
        key=key,
    )

    W_dist = model.last_layer.W()
    keys_w = jax.random.split(key, n_samples)

    def deterministic_rollout(key_w_i):
        W_sample = W_dist.rsample(key_w_i)

        def step_fn(curr_state, action_h):
            x_in = jnp.concatenate([curr_state, action_h], axis=-1)
            feat = model.feature_extractor(x_in)
            delta = (W_sample @ feat[..., None]).squeeze(-1)
            next_state = curr_state + delta
            return next_state, next_state

        _, trajectory = jax.lax.scan(step_fn, initial_state, action_sequence)
        return trajectory

    sampled_trajectories = jax.vmap(deterministic_rollout)(keys_w)
    rmse_per_h, spread_per_h = rollout_diagnostics(sampled_trajectories, target_trajectory)

    diagnostics = {
        "val_loss": kde_loss,
        "kde_nll": kde_loss,
        "rmse_per_horizon": rmse_per_h,
        "spread_per_horizon": spread_per_h,
        "spread_skill_ratio": spread_per_h / (rmse_per_h + 1e-8),
    }
    return kde_loss, diagnostics
