"""Logging helpers shared across training loops."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from typing import Any


def _state_copy(state):
    return jax.tree.map(lambda value: jnp.array(value), state)


def _state_norm(state) -> float:
    leaves = jax.tree.leaves(state)
    return float(jnp.sqrt(sum(jnp.sum(value**2) for value in leaves)))


def _state_delta_norm(current, previous) -> float:
    delta = jax.tree.map(lambda new, old: new - old, current, previous)
    return _state_norm(delta)


def _relative_change(change: float, reference_norm: float) -> float:
    return float(change / max(reference_norm, 1e-12))


def _effective_rank(features: np.ndarray) -> tuple[float, float]:
    centered = features - np.mean(features, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    energy = singular_values**2
    total = float(np.sum(energy))
    if total <= 0.0:
        return 0.0, 0.0
    probabilities = energy / total
    positive = probabilities[probabilities > 0.0]
    rank = float(np.exp(-np.sum(positive * np.log(positive))))
    max_rank = min(centered.shape[0] - 1, centered.shape[1])
    normalized = rank / max(max_rank, 1)
    return rank, float(normalized)


class OptimizationDynamicsTracker:
    """Track compact parameter, posterior, and fixed-probe feature dynamics."""

    def __init__(self, model, probe_inputs) -> None:
        self.probe_inputs = probe_inputs
        self.initial_backbone = _state_copy(
            nnx.state(model.feature_extractor, nnx.Param)
        )
        self.previous_backbone = _state_copy(self.initial_backbone)
        self.initial_last_layer = _state_copy(nnx.state(model.last_layer, nnx.Param))
        self.previous_last_layer = _state_copy(self.initial_last_layer)
        self.initial_w_mean = jnp.array(model.last_layer.W_mean.value)
        self.previous_w_mean = jnp.array(self.initial_w_mean)
        self.initial_covariance_params = (
            jnp.array(model.last_layer.W_logdiag.value),
            jnp.array(model.last_layer.W_offdiag.value),
        )
        self.previous_covariance_params = self.initial_covariance_params
        self.initial_process_noise = jnp.array(
            model.last_layer.process_noise_log_std.value
        )
        self.previous_process_noise = jnp.array(self.initial_process_noise)
        self.initial_features = self._features(model)
        self.previous_features = np.array(self.initial_features, copy=True)

    def _features(self, model) -> np.ndarray:
        return np.asarray(
            jax.vmap(model.feature_extractor)(self.probe_inputs), dtype=float
        )

    @staticmethod
    def _array_change(current, previous) -> float:
        return float(jnp.linalg.norm(current - previous))

    def log(self, logger: Any, model, epoch: int) -> None:
        backbone = nnx.state(model.feature_extractor, nnx.Param)
        last_layer = nnx.state(model.last_layer, nnx.Param)
        backbone_norm = _state_norm(backbone)
        last_layer_norm = _state_norm(last_layer)
        previous_backbone_norm = _state_norm(self.previous_backbone)
        previous_last_layer_norm = _state_norm(self.previous_last_layer)
        backbone_update = _state_delta_norm(backbone, self.previous_backbone)
        last_layer_update = _state_delta_norm(last_layer, self.previous_last_layer)

        logger.add_scalar("Parameters/backbone_norm", backbone_norm, epoch)
        logger.add_scalar("Parameters/backbone_update_norm", backbone_update, epoch)
        logger.add_scalar(
            "Parameters/backbone_relative_update",
            _relative_change(backbone_update, previous_backbone_norm),
            epoch,
        )
        logger.add_scalar("Parameters/last_layer_norm", last_layer_norm, epoch)
        logger.add_scalar("Parameters/last_layer_update_norm", last_layer_update, epoch)
        logger.add_scalar(
            "Parameters/last_layer_relative_update",
            _relative_change(last_layer_update, previous_last_layer_norm),
            epoch,
        )

        current_w_mean = model.last_layer.W_mean.value
        w_mean_update = self._array_change(current_w_mean, self.previous_w_mean)
        w_mean_from_init = self._array_change(current_w_mean, self.initial_w_mean)
        logger.add_scalar("Posterior/W_mean_update_norm", w_mean_update, epoch)
        logger.add_scalar(
            "Posterior/W_mean_relative_update",
            _relative_change(
                w_mean_update, float(jnp.linalg.norm(self.previous_w_mean))
            ),
            epoch,
        )
        logger.add_scalar("Posterior/W_mean_drift_from_init", w_mean_from_init, epoch)

        current_covariance_params = (
            model.last_layer.W_logdiag.value,
            model.last_layer.W_offdiag.value,
        )
        covariance_update = float(
            jnp.sqrt(
                sum(
                    jnp.sum((current - previous) ** 2)
                    for current, previous in zip(
                        current_covariance_params,
                        self.previous_covariance_params,
                    )
                )
            )
        )
        covariance_from_init = float(
            jnp.sqrt(
                sum(
                    jnp.sum((current - initial) ** 2)
                    for current, initial in zip(
                        current_covariance_params,
                        self.initial_covariance_params,
                    )
                )
            )
        )
        logger.add_scalar(
            "Posterior/covariance_parameter_update_norm", covariance_update, epoch
        )
        previous_covariance_norm = float(
            jnp.sqrt(
                sum(jnp.sum(value**2) for value in self.previous_covariance_params)
            )
        )
        logger.add_scalar(
            "Posterior/covariance_parameter_relative_update",
            _relative_change(covariance_update, previous_covariance_norm),
            epoch,
        )
        logger.add_scalar(
            "Posterior/covariance_parameter_drift_from_init",
            covariance_from_init,
            epoch,
        )

        current_process_noise = model.last_layer.process_noise_log_std.value
        process_noise_update = self._array_change(
            current_process_noise, self.previous_process_noise
        )
        process_noise_from_init = self._array_change(
            current_process_noise, self.initial_process_noise
        )
        logger.add_scalar(
            "Posterior/process_noise_parameter_update_norm",
            process_noise_update,
            epoch,
        )
        logger.add_scalar(
            "Posterior/process_noise_parameter_drift_from_init",
            process_noise_from_init,
            epoch,
        )

        features = self._features(model)
        feature_norm = float(np.linalg.norm(features))
        previous_feature_norm = float(np.linalg.norm(self.previous_features))
        feature_update = float(np.linalg.norm(features - self.previous_features))
        feature_from_init = float(np.linalg.norm(features - self.initial_features))
        effective_rank, normalized_rank = _effective_rank(features)
        logger.add_scalar("Representation/probe_feature_norm", feature_norm, epoch)
        logger.add_scalar("Representation/probe_feature_mean", float(np.mean(features)), epoch)
        logger.add_scalar("Representation/probe_feature_std", float(np.std(features)), epoch)
        logger.add_scalar("Representation/probe_feature_effective_rank", effective_rank, epoch)
        logger.add_scalar(
            "Representation/probe_feature_effective_rank_normalized",
            normalized_rank,
            epoch,
        )
        logger.add_scalar("Representation/probe_feature_update_norm", feature_update, epoch)
        logger.add_scalar(
            "Representation/probe_feature_relative_update",
            _relative_change(feature_update, previous_feature_norm),
            epoch,
        )
        logger.add_scalar(
            "Representation/probe_feature_drift_from_init", feature_from_init, epoch
        )

        self.previous_backbone = _state_copy(backbone)
        self.previous_last_layer = _state_copy(last_layer)
        self.previous_w_mean = jnp.array(current_w_mean)
        self.previous_covariance_params = tuple(
            jnp.array(value) for value in current_covariance_params
        )
        self.previous_process_noise = jnp.array(current_process_noise)
        self.previous_features = np.array(features, copy=True)


def log_model_stats(
    logger: Any,
    model,
    epoch: int,
    print_every: int,
    log_kl: bool = True,
    kl_in_objective: bool = True,
):
    """Log process noise, weight posterior statistics, and W_mean histogram.

    Args:
        logger: Scalar metric logger.
        model: VBLL model with a last_layer attribute.
        epoch: Current training epoch.
        print_every: Histogram is logged every this many epochs.
        log_kl: Whether to log KL divergence. Set False for GVI which uses
            Renyi divergence instead.
        kl_in_objective: Whether the current objective includes the KL term.

    Returns:
        W_dist: The weight distribution object (re-used by callers that need it).
    """
    if hasattr(model, 'process_noise_stdv'):
        for i, pn_val in enumerate(model.process_noise_stdv):
            logger.add_scalar(f'Noise/process_stddev_dim_{i}', float(pn_val), epoch)

    W_dist = model.last_layer.W()
    logger.add_scalar('Weights/W_mean_norm', float(jnp.linalg.norm(model.last_layer.W_mean.value)), epoch)
    trace_cov, logdet_cov = W_dist.trace_covariance, W_dist.logdet_covariance
    for i in range(len(trace_cov)):
        logger.add_scalar(f'Weights/trace_cov_dim_{i}', float(trace_cov[i]), epoch)
        logger.add_scalar(f'Weights/logdet_cov_dim_{i}', float(logdet_cov[i]), epoch)
    if log_kl:
        kl = model.last_layer._gaussian_kl()
        logger.add_scalar('Weights/kl_divergence', float(kl), epoch)
        logger.add_scalar(
            'Weights/scaled_kl_contribution',
            float(model.last_layer.regularization_weight * kl)
            if kl_in_objective
            else 0.0,
            epoch,
        )

    if epoch % print_every == 0:
        logger.add_histogram('Histograms/W_mean', model.last_layer.W_mean.value, epoch)

    return W_dist


def log_per_horizon_diagnostics(
    logger: Any,
    val_diagnostics: dict,
    epoch: int,
) -> None:
    arr = jnp.array
    rmse = val_diagnostics["rmse_per_horizon"]
    spread = val_diagnostics["spread_per_horizon"]
    ssr = val_diagnostics["spread_skill_ratio"]
    logger.add_scalar("Diagnostics/rmse_mean", float(jnp.mean(arr(rmse))), epoch)
    logger.add_scalar("Diagnostics/spread_mean", float(jnp.mean(arr(spread))), epoch)
    logger.add_scalar("Diagnostics/ssr_mean", float(jnp.mean(arr(ssr))), epoch)

    if hasattr(logger, "add_horizon_diagnostics"):
        logger.add_horizon_diagnostics(epoch, rmse, spread, ssr)
    else:
        for horizon, (rmse_h, spread_h, ssr_h) in enumerate(
            zip(rmse, spread, ssr)
        ):
            logger.add_scalar(
                f"Diagnostics/rmse_by_horizon/{horizon}", float(rmse_h), epoch
            )
            logger.add_scalar(
                f"Diagnostics/spread_by_horizon/{horizon}",
                float(spread_h),
                epoch,
            )
            logger.add_scalar(
                f"Diagnostics/ssr_by_horizon/{horizon}", float(ssr_h), epoch
            )
