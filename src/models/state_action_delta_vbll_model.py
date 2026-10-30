import jax
import jax.numpy as jnp

from src.models.vbll_model import Normal
from src.models.delta_vbll_model import VBLLDeltaModel


class StateActionVBLLDeltaModel(VBLLDeltaModel):
    """VBLLDeltaModel adapted for action-conditioned WheelBot dynamics.

    Feature extractor sees [state, action] concatenated.
    Skip connection applies only to the state part:
        next_state = state + W * phi([state, action])
    """

    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self._state_dim = state_dim
        super().__init__(
            input_dim=state_dim + action_dim,
            output_dim=state_dim,
            **kwargs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        state = x[: self._state_dim]
        features = self.feature_extractor(x)
        pred_dist = self.last_layer.predictive(features)
        return state + pred_dist.mean

    def predictive(self, x: jax.Array) -> Normal:
        state = x[: self._state_dim]
        features = self.feature_extractor(x)
        delta_dist = self.last_layer.predictive(features)
        return Normal(state + delta_dist.mean, delta_dist.scale)

    def train_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        state = x[: self._state_dim]
        features = self.feature_extractor(x)
        return self.last_layer.train_loss_fn(features, y - state)

    def val_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        state = x[: self._state_dim]
        features = self.feature_extractor(x)
        return self.last_layer.val_loss_fn(features, y - state)

    def sample_predict(self, x: jax.Array, key: jax.Array) -> jax.Array:
        state = x[: self._state_dim]
        features = self.feature_extractor(x)
        w_dist = self.last_layer.W()
        noise_dist = self.last_layer.noise()
        key1, key2 = jax.random.split(key)
        w_sample = w_dist.rsample(key1)
        delta = (w_sample @ features[..., None]).squeeze(-1)
        noise_sample = noise_dist.rsample(key2)
        return state + delta + noise_sample

    def rollout_mean_trajectory(
        self,
        initial_state: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        def step_fn(state, action):
            x = jnp.concatenate([state, action])
            next_state = self.__call__(x)
            return next_state, next_state

        _, traj = jax.lax.scan(step_fn, initial_state, actions)
        return jnp.concatenate([initial_state[None], traj], axis=0)

    def rollout_sampled_trajectory(
        self,
        initial_state: jax.Array,
        actions: jax.Array,
        key: jax.Array,
        add_noise: bool = True,
    ) -> jax.Array:
        key_w, key_noise = jax.random.split(key)
        w_dist = self.last_layer.W()
        w_sample = w_dist.rsample(key_w)
        sigma = self.last_layer.process_noise().scale
        n_steps = actions.shape[0]

        if add_noise:
            noise_keys = jax.random.split(key_noise, n_steps)

            def step_fn(state, inputs):
                action, nk = inputs
                x = jnp.concatenate([state, action])
                features = self.feature_extractor(x)
                delta = (w_sample @ features[..., None]).squeeze(-1)
                noise = sigma * jax.random.normal(nk, state.shape)
                next_state = state + delta + noise
                return next_state, next_state

            _, traj = jax.lax.scan(step_fn, initial_state, (actions, noise_keys))
        else:
            def step_fn(state, action):
                x = jnp.concatenate([state, action])
                features = self.feature_extractor(x)
                delta = (w_sample @ features[..., None]).squeeze(-1)
                next_state = state + delta
                return next_state, next_state

            _, traj = jax.lax.scan(step_fn, initial_state, actions)

        return jnp.concatenate([initial_state[None], traj], axis=0)

    def simulate(
        self,
        initial_state: jax.Array,
        actions: jax.Array,
        key: jax.Array,
    ) -> jax.Array:
        return self.rollout_sampled_trajectory(initial_state, actions, key, add_noise=True)
