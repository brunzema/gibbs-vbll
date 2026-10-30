"""
Delta (Residual) VBLL Model with Process Noise.

Builds on the delta/residual formulation:

    x_{t+1} = x_t + W^T phi(x_t) + ν,  ν ~ N(0, Σ_process)

The model uses:
- Variational Bayesian inference over last-layer weights W
- Standard VBLL observation and process noise
- Energy Score for multi-step training (strictly proper scoring rule)

Single-step training uses the standard VBLL ELBO on deltas (y - x).
Multi-step rollouts combine weight uncertainty and process noise.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from src.models.vbll_model import (
    Normal,
    SampledVBLLModel,
    VBLLModel,
)


# Re-export base classes so downstream code can import everything from this module.
__all__ = [
    "SampledVBLLModel",
    "SampledVBLLDeltaModel",
    "VBLLDeltaModel",
]


class SampledVBLLDeltaModel(SampledVBLLModel):
    """Deterministic realization of the delta model with weights sampled from posterior.

    Like SampledVBLLModel but includes the residual skip connection:
        x_{t+1} = x_t + W * phi(x_t)
    """

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: predict next state via skip connection.

        Args:
            x: Input of shape (input_dim,)

        Returns:
            Next state x + delta of shape (output_dim,)
        """
        features = self.feature_extractor(x)
        delta = (self.W.value @ features[..., None]).squeeze(-1)
        return x + delta


class VBLLDeltaModel(VBLLModel):
    """VBLL delta model with process noise for multi-step prediction.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dims: Tuple[int, ...] = (64, 64),
        feature_dim: int = 32,
        activation_fn=nnx.gelu,
        regularization_weight: float = 1.0,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
        noise_init: float = 0.01,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            input_dim: Dimension of input space
            output_dim: Dimension of output space (defaults to input_dim)
            hidden_dims: Hidden layer dimensions for feature extractor
            feature_dim: Dimension of feature space
            activation_fn: Activation function for feature extractor
            regularization_weight: Weight on KL (typically 1/dataset_size)
            prior_scale: Prior scale on last layer weights
            wishart_scale: Scale for noise covariance
            dof: Degrees of freedom for Wishart prior
            rngs: Random number generators
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            feature_dim=feature_dim,
            activation_fn=activation_fn,
            regularization_weight=regularization_weight,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            dof=dof,
            noise_init=noise_init,
            rngs=rngs,
        )

    # ------------------------------------------------------------------
    # Core forward methods (skip connection)
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array) -> jax.Array:
        """Mean prediction with skip connection: x + E[delta]."""
        features = self.feature_extractor(x)
        pred_dist = self.last_layer.predictive(features)
        return x + pred_dist.mean

    def predictive(self, x: jax.Array) -> Normal:
        """Predictive distribution over x_{t+1} = x_t + delta.

        The mean is shifted by x; the scale (uncertainty) is unchanged.
        """
        features = self.feature_extractor(x)
        delta_dist = self.last_layer.predictive(features)
        return Normal(x + delta_dist.mean, delta_dist.scale)

    # ------------------------------------------------------------------
    # Single-step losses (supervise on delta = y - x)
    # ------------------------------------------------------------------

    def train_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Single-step ELBO: target is the delta y - x."""
        features = self.feature_extractor(x)
        return self.last_layer.train_loss_fn(features, y - x)

    def val_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Single-step NLL: evaluated on delta y - x."""
        features = self.feature_extractor(x)
        return self.last_layer.val_loss_fn(features, y - x)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------

    def sample_predict(self, x: jax.Array, key: jax.Array) -> jax.Array:
        """Sample prediction with skip connection."""
        features = self.feature_extractor(x)
        W_dist = self.last_layer.W()
        noise_dist = self.last_layer.noise()

        key1, key2 = jax.random.split(key)
        W_sample = W_dist.rsample(key1)

        delta = (W_sample @ features[..., None]).squeeze(-1)
        noise_sample = noise_dist.rsample(key2)
        return x + delta + noise_sample

    def sample_weight_function(self, key: jax.Array) -> SampledVBLLDeltaModel:
        """Sample W from posterior and return a deterministic delta MLP."""
        W_dist = self.last_layer.W()
        W_sample = W_dist.rsample(key)
        return SampledVBLLDeltaModel(self.feature_extractor, W_sample)

    # ------------------------------------------------------------------
    # Rollout methods
    # ------------------------------------------------------------------

    def rollout_mean_trajectory(
        self,
        initial_state: jax.Array,
        n_steps: int,
    ) -> jax.Array:
        """Rollout using mean delta predictions.

        predictive() already returns Normal(x + delta_mean, ...), so
        pred_dist.mean gives the next state directly.
        """
        def step_fn(current_state, _):
            pred_dist = self.predictive(current_state)
            next_state = pred_dist.mean
            return next_state, next_state
        
        _, trajectory = jax.lax.scan(step_fn, initial_state, None, length=n_steps)
        # Prepend initial state
        full_trajectory = jnp.concatenate([initial_state[None], trajectory], axis=0)
        return full_trajectory

    def rollout_sampled_trajectory(
        self,
        initial_state: jax.Array,
        n_steps: int,
        key: jax.Array,
        add_noise: bool = False,
    ) -> jax.Array:
        """Rollout using a single W sample with delta/skip connection.

        Samples w once from the posterior. If add_noise=True, applies
        process noise at each step: ν_h ~ N(0, σ²I).
        """
        key_w, key_noise = jax.random.split(key)

        W_dist = self.last_layer.W()
        W_sample = W_dist.rsample(key_w)
        sigma = self.last_layer.process_noise().scale

        if add_noise:
            noise_keys = jax.random.split(key_noise, n_steps)
            
            def step_fn(current_state, noise_key):
                features = self.feature_extractor(current_state)
                delta = (W_sample @ features[..., None]).squeeze(-1)
                noise = sigma * jax.random.normal(noise_key, current_state.shape)
                next_state = current_state + delta + noise
                return next_state, next_state
            
            _, trajectory = jax.lax.scan(step_fn, initial_state, noise_keys)
        else:
            def step_fn(current_state, _):
                features = self.feature_extractor(current_state)
                delta = (W_sample @ features[..., None]).squeeze(-1)
                next_state = current_state + delta
                return next_state, next_state
            
            _, trajectory = jax.lax.scan(step_fn, initial_state, None, length=n_steps)
        
        # Prepend initial state
        full_trajectory = jnp.concatenate([initial_state[None], trajectory], axis=0)
        return full_trajectory

    def simulate(
        self,
        initial_state: jax.Array,
        n_steps: int,
        key: jax.Array,
    ) -> jax.Array:
        return self.rollout_sampled_trajectory(initial_state, n_steps, key, add_noise=True)