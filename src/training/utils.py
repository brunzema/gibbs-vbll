from typing import Tuple
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx

from src.models.vbll_model import VBLLModel


@jax.jit
def crps_gaussian(
    mean: jax.Array,
    scale: jax.Array,
    target: jax.Array,
) -> jax.Array:
    """Closed-form CRPS for a diagonal Gaussian predictive.

    Args:
        mean:   Predictive mean,  shape (..., d)
        scale:  Predictive std,   shape (..., d)  — must be > 0
        target: Observed value,   shape (..., d)

    Returns:
        CRPS summed over the d dimensions, shape (...).
    """
    scale = jnp.maximum(scale, 1e-6)

    z = (target - mean) / scale

    phi_z = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2.0 * jnp.pi)
    Phi_z = 0.5 * (1.0 + jax.scipy.special.erf(z / jnp.sqrt(2.0)))

    crps_per_dim = scale * (
        z * (2.0 * Phi_z - 1.0)
        + 2.0 * phi_z
        - 1.0 / jnp.sqrt(jnp.pi)
    )

    return jnp.sum(crps_per_dim, axis=-1)


def precompute_true_state_quantities(
    model: VBLLModel,
    initial_state: jax.Array,     # (d,)
    target_trajectory: jax.Array,  # (H, d)
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Precompute features and deltas at true observed states.

    Returns:
        input_states: input_states of shape (H, d)
        target_deltas: target_deltas of shape (H, d)
        all_features:  all_features of shape (H, p)
    """
    input_states = jnp.concatenate(
        [initial_state[None], target_trajectory[:-1]], axis=0
    )  # (H, d)
    target_deltas = target_trajectory - input_states  # (H, d)
    all_features = model.feature_extractor(input_states)  # (H, p)
    return input_states, target_deltas, all_features


@jax.jit
def rollout_diagnostics(sampled_trajectories, target_trajectory):
    ensemble_mean = jnp.mean(sampled_trajectories, axis=0)
    rmse_per_h = jnp.sqrt(
        jnp.mean((ensemble_mean - target_trajectory) ** 2, axis=-1)
    )
    spread_per_h = jnp.mean(
        jnp.sqrt(jnp.mean(
            (sampled_trajectories - ensemble_mean[None]) ** 2, axis=-1
        )),
        axis=0,
    )
    return rmse_per_h, spread_per_h