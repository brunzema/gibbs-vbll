import numpy as np
import jax.numpy as jnp


def compute_normalization_params(data: np.ndarray) -> dict:
    """Compute per-dimension mean/std normalization params from an (N, dim) array.

    Args:
        data: Array of shape (N, dim) — e.g. training inputs or training targets.

    Returns:
        dict with keys 'mean' (np.ndarray), 'std' (np.ndarray), 'enabled' (True).
    """
    data = np.array(data)
    mean = data.mean(axis=0)
    std  = data.std(axis=0).clip(min=1e-8)
    return {'mean': mean, 'std': std, 'enabled': True}


class NormalizedModelWrapper:
    """Transparent normalization wrapper around a trained model.

    External callers always pass/receive states in the *original* (un-normalized)
    scale.  Internally the wrapper applies input normalization before forwarding
    to the underlying model and denormalizes model outputs before returning them.

    Storing input and output normalization separately future-proofs the design:
    when actions are concatenated to the state as model inputs, `input_norm` will
    cover the full (state, action) vector while `output_norm` remains state-only.

    Args:
        model:       Trained VBLLModel / VBLLDeltaModel instance.
        input_norm:  Dict with 'mean', 'std' covering the model's input dimensions.
        output_norm: Dict with 'mean', 'std' covering the model's output dimensions.
    """

    def __init__(self, model, input_norm: dict, output_norm: dict):
        self._model   = model
        self.in_mean  = jnp.array(input_norm['mean'],  dtype=jnp.float32)
        self.in_std   = jnp.array(input_norm['std'],   dtype=jnp.float32)
        self.out_mean = jnp.array(output_norm['mean'], dtype=jnp.float32)
        self.out_std  = jnp.array(output_norm['std'],  dtype=jnp.float32)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _norm_in(self, x):
        return (jnp.array(x, dtype=jnp.float32) - self.in_mean) / self.in_std

    def _denorm_out(self, x):
        return jnp.array(x, dtype=jnp.float32) * self.out_std + self.out_mean

    # ── public API (mirrors VBLLDeltaModel) ──────────────────────────────────

    def rollout_mean_trajectory(self, x0, n_steps):
        """Deterministic mean rollout in original scale.

        Returns array of shape (n_steps+1, state_dim) in original units.
        """
        traj_norm = self._model.rollout_mean_trajectory(self._norm_in(x0), n_steps)
        return self._denorm_out(traj_norm)

    def simulate(self, x0, n_steps, key):
        """Single stochastic sample trajectory in original scale.

        Returns array of shape (n_steps+1, state_dim) in original units.
        """
        traj_norm = self._model.simulate(self._norm_in(x0), n_steps, key)
        return self._denorm_out(traj_norm)
