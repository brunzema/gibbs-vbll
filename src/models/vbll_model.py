"""
Variational Bayesian Last Layer Model with Neural Network Backbone using Flax NNX.

This implements a regression model f(x) -> y where:
- A neural network backbone extracts features: phi_theta(x)
- A variational Bayesian last layer predicts: y = W * phi_theta(x)
- The weights W have a variational posterior (mean-field Gaussian with dense covariance)
- The noise has a Gaussian distribution with learned variance

Based on the VBLL paper and implementation from:
https://github.com/cambridge-mlg/vbll
"""

import jax
import jax.numpy as jnp
from flax import nnx
from typing import Tuple, Optional
import numpy as np

from src.utils.model_management import get_activation_fn


_EPS = 1e-6


class DenseNormal:
    """Multivariate normal distribution with dense covariance parameterized by Cholesky factor.
    
    This corresponds to the DenseNormal distribution in the torch implementation.
    The covariance is parameterized as Sigma = L @ L.T where L is lower triangular.
    """
    
    def __init__(self, mean: jax.Array, chol_covariance: jax.Array):
        """
        Args:
            mean: Mean of distribution, shape (..., dim)
            chol_covariance: Lower triangular Cholesky factor, shape (..., dim, dim)
        """
        self.mean = mean
        self.chol_covariance = chol_covariance
    
    @property
    def covariance(self) -> jax.Array:
        """Compute covariance matrix: Sigma = L @ L.T"""
        return self.chol_covariance @ jnp.swapaxes(self.chol_covariance, -2, -1)
    
    @property
    def variance(self) -> jax.Array:
        """Compute diagonal variance (useful for scalar outputs)"""
        # For dense covariance, variance is diagonal of Sigma = L @ L.T
        return (self.chol_covariance ** 2).sum(-1)
    
    @property
    def logdet_covariance(self) -> jax.Array:
        """Log determinant of covariance matrix via Cholesky decomposition"""
        # log|Sigma| = 2 * sum(log(diag(L)))
        return 2.0 * jnp.log(jnp.diagonal(self.chol_covariance, axis1=-2, axis2=-1)).sum(-1)
    
    @property
    def trace_covariance(self) -> jax.Array:
        """Trace of covariance matrix"""
        # tr(L @ L.T) = ||L||_F^2 (Frobenius norm squared)
        return (self.chol_covariance ** 2).sum(-1).sum(-1)
    
    def covariance_weighted_inner_prod(self, b: jax.Array) -> jax.Array:
        """Compute b.T @ Sigma @ b efficiently using Cholesky factor.
        
        Args:
            b: Vector of shape (..., dim, 1)
            
        Returns:
            Scalar result b.T @ Sigma @ b
        """
        # b.T @ Sigma @ b = b.T @ L @ L.T @ b = ||L.T @ b||^2
        Lt_b = jnp.swapaxes(self.chol_covariance, -2, -1) @ b
        result = (Lt_b ** 2).sum(-2)
        return result.squeeze(-1)
    
    def log_prob(self, y: jax.Array) -> jax.Array:
        """Compute log probability of observation y.
        
        Args:
            y: Observation of shape (..., dim)
            
        Returns:
            Log probability
        """
        dim = self.mean.shape[-1]
        error = y - self.mean
        
        # Solve L @ x = error for x, then compute ||x||^2
        # This is equivalent to error.T @ Sigma^{-1} @ error
        x = jax.scipy.linalg.solve_triangular(
            self.chol_covariance, error[..., None], lower=True
        )
        mahal = (x ** 2).sum()
        
        logdet = self.logdet_covariance
        const = dim * jnp.log(2 * jnp.pi)
        
        return -0.5 * (mahal + logdet + const)
    
    def rsample(self, key: jax.Array) -> jax.Array:
        """Sample from the distribution using reparameterization trick.
        
        Args:
            key: JAX random key
            
        Returns:
            Sample of shape (..., dim)
        """
        dim = self.mean.shape[-1]
        eps = jax.random.normal(key, self.mean.shape)
        # Sample = mean + L @ eps
        return self.mean + (self.chol_covariance @ eps[..., None]).squeeze(-1)
    
    def __matmul__(self, x: jax.Array):
        """Compute W @ x where W ~ N(mean, Sigma).
        
        Returns a scalar Normal distribution (diagonal covariance).
        
        Args:
            x: Input of shape (..., feature_dim, 1)
            
        Returns:
            Normal distribution with mean and scale
        """
        # Mean: mean @ x
        new_mean = (self.mean @ x).squeeze(-1)
        
        # Variance: x.T @ Sigma @ x
        new_var = self.covariance_weighted_inner_prod(x)
        new_scale = jnp.sqrt(jnp.clip(new_var, a_min=1e-12))
        
        return Normal(new_mean, new_scale)


class Normal:
    """Diagonal Gaussian distribution (for convenience)."""
    
    def __init__(self, mean: jax.Array, scale: jax.Array):
        """
        Args:
            mean: Mean of distribution, shape (...)
            scale: Standard deviation, shape (...)
        """
        self.mean = mean
        self.scale = scale
    
    @property
    def variance(self) -> jax.Array:
        return self.scale ** 2
    
    def log_prob(self, y: jax.Array) -> jax.Array:
        """Compute log probability of observation y.
        
        Args:
            y: Observation of same shape as mean
            
        Returns:
            Log probability (sum over dimensions)
        """
        # Log prob for diagonal Gaussian
        const = jnp.log(2 * jnp.pi)
        logdet = 2 * jnp.log(self.scale)
        error = y - self.mean
        mahal = (error ** 2) / (self.scale ** 2)
        return -0.5 * (mahal + logdet + const).sum(-1)
    
    def rsample(self, key: jax.Array) -> jax.Array:
        """Sample using reparameterization trick."""
        eps = jax.random.normal(key, self.mean.shape)
        return self.mean + self.scale * eps


class VariationalBayesianLastLayer(nnx.Module):
    """Variational Bayesian last layer with dense covariance for regression.
    
    This implements the VBLL regression layer from the paper, with dense covariance
    parameterization only. The model learns:
    - W: weight matrix with variational posterior q(W) = N(mu_W, Sigma_W)
    - noise: observation noise with fixed or learned variance
    """
    
    def __init__(
        self,
        feature_dim: int,
        output_dim: int,
        regularization_weight: float = 1.0,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
        noise_init: float = 0.01,
        divergence: str = "kl",
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            feature_dim: Dimension of input features
            output_dim: Dimension of output
            regularization_weight: Weight on KL regularization term (typically 1/dataset_size)
            prior_scale: Scale of prior covariance on weights
            wishart_scale: Scale parameter for noise covariance initialization
            dof: Degrees of freedom for Wishart prior on noise covariance
            divergence: Divergence type to use for regularization (kl or renyi)
            rngs: Random number generators
        """
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.regularization_weight = regularization_weight
        self.prior_scale = prior_scale * (1.0 / feature_dim)
        self.divergence = divergence
        self.wishart_scale = wishart_scale
        self.dof = (dof + output_dim + 1.0) / 2.0
        self.alpha = 0.5  # Renyi alpha parameter (can be tuned)
        
        # Weight distribution parameters: q(W) = N(mu_W, Sigma_W)
        # W has shape (output_dim, feature_dim)
        # Each row has its own dense covariance
        self.W_mean = nnx.Param(
            jax.random.normal(rngs.params(), (output_dim, feature_dim)) * 
            jnp.sqrt(2.0 / feature_dim)
        )
        
        # Cholesky factors for covariance: one per output dimension
        # Shape: (output_dim, feature_dim, feature_dim)
        # Initialize with small values to start with low uncertainty
        self.W_logdiag = nnx.Param(
            1e-3 * jax.random.normal(rngs.params(), (output_dim, feature_dim)) - 
            0.5 * jnp.log(feature_dim)
        )
        self.W_offdiag = nnx.Param(
            1e-3 * jax.random.normal(rngs.params(), (output_dim, feature_dim, feature_dim)) /
            feature_dim
        )
            
        self.process_noise_mean = jnp.zeros(output_dim)
        if isinstance(noise_init, float) and noise_init > 0.0:
            self.process_noise_log_std = nnx.Param(
                jnp.ones((output_dim,))
                * jnp.log(noise_init)
            )
        else:
            self.process_noise_log_std = nnx.Param(
                jax.random.normal(rngs.params(), (output_dim,))
                * jnp.log(wishart_scale)
            )
    
    def W(self) -> DenseNormal:
        """Get weight distribution q(W)."""
        # Construct lower triangular Cholesky factor
        cov_diag = jnp.exp(self.W_logdiag.value)
        # Make lower triangular + diagonal
        tril = jnp.tril(self.W_offdiag.value, k=-1) + jnp.vectorize(
            jnp.diag, signature='(n)->(n,n)'
        )(cov_diag)
        return DenseNormal(self.W_mean.value, tril)
    
    def process_noise(self) -> Normal:
        """Get process noise distribution."""
        return Normal(self.process_noise_mean, jnp.exp(self.process_noise_log_std.value))
    
    def predictive(self, x: jax.Array) -> Normal:
        """Compute predictive distribution p(y|x) = N(W @ x, Sigma).
        
        Args:
            x: Input features of shape (feature_dim,)
            
        Returns:
            Normal distribution over outputs
        """
        W_dist = self.W()
        # noise_dist = self.noise()
        process_noise_dist = self.process_noise()
        
        # Predictive mean: E[W] @ x
        pred_mean = (W_dist.mean @ x[..., None]).squeeze(-1)
        
        # Predictive variance: noise + x.T @ Sigma_W @ x
        epistemic_var = W_dist.covariance_weighted_inner_prod(x[..., None])
        pred_scale = jnp.sqrt((process_noise_dist.variance) + epistemic_var)
        return Normal(pred_mean, pred_scale)
    
    def train_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """One-step ELBO loss."""
        W_dist = self.W()
        proc_var = jnp.exp(2.0 * self.process_noise_log_std.value)
        total_var = proc_var 
        precision = 1.0 / total_var
        
        # 1. Likelihood
        pred_mean = (W_dist.mean @ x[..., None]).squeeze(-1)
        error = y - pred_mean
        mahal = (error**2 * precision).sum()
        logdet = jnp.sum(jnp.log(total_var))
        
        # 2. Trace Term (Epistemic Uncertainty penalty)
        epistemic_term = (W_dist.covariance_weighted_inner_prod(x[..., None]) * precision).sum()
        
        # Total Neg Log Likelihood (Expected)
        nll = 0.5 * (mahal + logdet + epistemic_term + self.output_dim * jnp.log(2 * jnp.pi))
        return nll
    
    def val_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute validation loss (negative log likelihood under predictive).
        
        Args:
            x: Input features of shape (feature_dim,)
            y: Target of shape (output_dim,)
            
        Returns:
            Negative log probability
        """
        pred_dist = self.predictive(x)
        return -pred_dist.log_prob(y)
    
    def _gaussian_kl(self) -> jax.Array:
        """Compute KL(q || N(0, prior_scale * I)).
        
        Args:
            q: Posterior distribution
            prior_scale: Variance of prior
            
        Returns:
            KL divergence (scalar or per output dimension, summed)
        """
        q = self.W()
        prior_scale = self.prior_scale
        
        feat_dim = q.mean.shape[-1]
        mse_term = (q.mean ** 2).sum(-1).sum(-1) / prior_scale
        trace_term = (q.trace_covariance / prior_scale).sum(-1)
        logdet_term = (feat_dim * jnp.log(prior_scale) - q.logdet_covariance).sum(-1)
        const_term  = -float(feat_dim) * float(q.mean.shape[0])
        return 0.5 * (mse_term + trace_term + logdet_term + const_term)

    def _renyi_divergence(self) -> jax.Array:
        """Renyi divergence D_alpha(q || p), p = N(0, prior_scale * I).

        Fixes:
        1. assume_a='gen' instead of 'pos': M = I + c*Sigma_q is not
            guaranteed positive definite when alpha > 1.
        2. NaN guard now covers the full result, not just logdet_M,
            so a non-p.d. M produces NaN everywhere consistently.
        """
        q = self.W()
        prior_scale = self.prior_scale
        alpha = self.alpha

        if abs(alpha - 1.0) < _EPS:
            return self._gaussian_kl()

        d, p = q.mean.shape
        L  = q.chol_covariance          # (d, p, p)
        mu = q.mean                     # (d, p)

        c = (1.0 - alpha) / (alpha * prior_scale)

        Sigma_q = L @ jnp.swapaxes(L, -2, -1)          # (d, p, p)
        M = jnp.eye(p) + c * Sigma_q                    # (d, p, p)

        # Log-det of M — check sign for each output dim
        sign, logdet_M = jnp.linalg.slogdet(M)          # (d,), (d,)
        valid = sign > 0.0                               # (d,)

        M_inv_mu = jax.scipy.linalg.solve(
            M, mu[..., None], assume_a='gen'
        )                                                # (d, p, 1)
        mu_Minv_mu = jnp.sum(mu[..., None] * M_inv_mu, axis=(-2, -1))  # (d,)
        quad_term = mu_Minv_mu / (2.0 * prior_scale)    # (d,)

        logdet_cov_q = q.logdet_covariance              # (d,)
        logdet_prior = p * jnp.log(prior_scale)

        log_term = (
            (1.0 - alpha) * logdet_cov_q
            + alpha * logdet_prior
            - (p * jnp.log(alpha * prior_scale) + logdet_M)
        )                                                # (d,)

        renyi_div_per_dim = (
            log_term / (2.0 * (alpha - 1.0)) + quad_term
        )                                                # (d,)

        renyi_div_per_dim = jnp.where(valid, renyi_div_per_dim, jnp.nan)
        return jnp.sum(renyi_div_per_dim)

    def _wishart_regulariser(self) -> jax.Array:
        """log p(precision) under the Wishart prior (up to constants)."""
        proc_var = jnp.exp(2.0 * self.process_noise_log_std.value)
        logdet_proc = 2.0 * jnp.sum(self.process_noise_log_std.value)
        return (
            self.dof * (-logdet_proc)
            - 0.5 * self.wishart_scale * (1.0 / proc_var).sum()
        )

class FeatureExtractor(nnx.Module):
    """Neural network feature extractor (deterministic backbone)."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...],
        feature_dim: int,
        activation_fn: callable = jax.nn.elu,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            input_dim: Dimension of input state
            hidden_dims: Tuple of hidden layer dimensions
            feature_dim: Dimension of output features
            activation_fn: Activation function to use
            rngs: Random number generators
        """
        # Build the network architecture
        layers_list = []
        
        # Input layer
        layers_list.append(nnx.Linear(input_dim, hidden_dims[0], rngs=rngs))
        layers_list.append(lambda x: activation_fn(x))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers_list.append(nnx.Linear(hidden_dims[i], hidden_dims[i+1], rngs=rngs))
            layers_list.append(lambda x: activation_fn(x))
        
        # Output layer (features)
        layers_list.append(nnx.Linear(hidden_dims[-1], feature_dim, rngs=rngs))
        layers_list.append(lambda x: activation_fn(x))
        
        # Use Sequential to combine layers
        self.network = nnx.Sequential(*layers_list)
        
    def __call__(self, x: jax.Array) -> jax.Array:
        """Extract features from input state."""
        return self.network(x)


class SampledVBLLModel(nnx.Module):
    """A deterministic MLP realization with weights sampled from VBLL posterior.
    
    This represents one hypothesis/sample from the posterior, acting as a
    standard feedforward neural network with fixed weights.
    """
    
    def __init__(self, feature_extractor: FeatureExtractor, W_sample: jax.Array):
        """Initialize with feature extractor and sampled weights.
        
        Args:
            feature_extractor: The feature extraction network
            W_sample: Sampled weight matrix of shape (output_dim, feature_dim)
        """
        self.feature_extractor = feature_extractor
        # Store sampled weights as a non-trainable parameter
        self.W = nnx.Param(W_sample)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with sampled weights.
        
        Args:
            x: Input of shape (input_dim,)
            
        Returns:
            Prediction of shape (output_dim,)
        """
        features = self.feature_extractor(x)
        return (self.W.value @ features[..., None]).squeeze(-1)


class VBLLModel(nnx.Module):
    """Complete Variational Bayesian Last Layer regression model.

    This combines a deterministic feature extractor with a variational
    Bayesian last layer for uncertainty-aware predictions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: Optional[int] = None,
        hidden_dims: Tuple[int, ...] = (64, 64),
        activation_fn = nnx.elu,
        feature_dim: int = 32,
        regularization_weight: float = 1.0,
        prior_scale: float = 1.0,
        wishart_scale: float = 0.1,
        dof: float = 1.0,
        noise_init: float = 0.01,
        divergence: str = "kl",
        *,
        rngs: nnx.Rngs,
    ):
        """
        Args:
            input_dim: Dimension of input space
            output_dim: Dimension of output space (defaults to input_dim if None)
            hidden_dims: Hidden layer dimensions for feature extractor
            feature_dim: Dimension of feature space
            regularization_weight: Weight on KL regularization (typically 1/dataset_size)
            prior_scale: Prior scale on last layer weights
            wishart_scale: Scale for noise covariance
            dof: Degrees of freedom for Wishart prior
            rngs: Random number generators
        """
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        # Feature extractor (deterministic neural network)
        self.feature_extractor = FeatureExtractor(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            feature_dim=feature_dim,
            activation_fn=activation_fn,
            rngs=rngs,
        )

        # Variational Bayesian last layer
        self.last_layer = VariationalBayesianLastLayer(
            feature_dim=feature_dim,
            output_dim=self.output_dim,
            regularization_weight=regularization_weight,
            prior_scale=prior_scale,
            wishart_scale=wishart_scale,
            dof=dof,
            noise_init=noise_init,
            divergence=divergence,
            rngs=rngs,
        )

    @classmethod
    def from_cfg(cls, cfg, state_dim: int, rngs: nnx.Rngs):
        """Create a VBLLModel from Hydra configuration.

        Args:
            cfg: Hydra configuration object (can be full cfg or just cfg.model)
            state_dim: State dimension (input/output dimension)
            rngs: Random number generators

        Returns:
            VBLLModel instance
        """
        # Handle both full cfg and cfg.model
        if hasattr(cfg, 'model'):
            model_cfg = cfg.model.model
        else:
            model_cfg = cfg.model if hasattr(cfg, 'model') else cfg

        # Get activation function
        activation_fn = get_activation_fn(model_cfg.activation_fn)

        # Build hidden_dims tuple from hidden_dim and num_hidden_layers
        hidden_dims = tuple([model_cfg.hidden_dim] * model_cfg.num_hidden_layers)

        # Compute regularization weight if not specified
        reg_weight = model_cfg.regularization_weight

        # Create and return model
        return cls(
            input_dim=state_dim,
            output_dim=state_dim,
            hidden_dims=hidden_dims,
            feature_dim=model_cfg.feature_dim,
            activation_fn=activation_fn,
            regularization_weight=reg_weight,
            prior_scale=model_cfg.prior_scale,
            wishart_scale=model_cfg.wishart_scale,
            rngs=rngs,
        )
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass: predict output (mean prediction).
        
        Args:
            x: Input of shape (input_dim,)
            
        Returns:
            Mean prediction of shape (output_dim,)
        """
        features = self.feature_extractor(x)
        pred_dist = self.last_layer.predictive(features)
        return pred_dist.mean
    
    # @property
    # def noise_stdv(self) -> jax.Array:
    #     """Get current observation noise standard deviation."""
    #     return self.last_layer.noise().scale

    @property
    def process_noise_stdv(self) -> jax.Array:
        """Get current process noise standard deviation."""
        return self.last_layer.process_noise().scale

    def predictive(self, x: jax.Array) -> Normal:
        """Get full predictive distribution.
        
        Args:
            x: Input of shape (input_dim,)
            
        Returns:
            Normal distribution over outputs
        """
        features = self.feature_extractor(x)
        return self.last_layer.predictive(features)
    
    def train_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute training loss (negative ELBO).
        
        Args:
            x: Input of shape (input_dim,)
            y: Target of shape (output_dim,)
            
        Returns:
            Training loss (negative ELBO)
        """
        features = self.feature_extractor(x)
        return self.last_layer.train_loss_fn(features, y)
    
    def val_loss_fn(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Compute validation loss.
        
        Args:
            x: Input of shape (input_dim,)
            y: Target of shape (output_dim,)
            
        Returns:
            Negative log probability
        """
        features = self.feature_extractor(x)
        return self.last_layer.val_loss_fn(features, y)
    
    def sample_predict(self, x: jax.Array, key: jax.Array) -> jax.Array:
        """Sample prediction from posterior.
        
        Args:
            x: Input of shape (input_dim,)
            key: JAX random key
            
        Returns:
            Sampled prediction of shape (output_dim,)
        """
        features = self.feature_extractor(x)
        W_dist = self.last_layer.W()
        
        # Sample weights
        key1, key2 = jax.random.split(key)
        W_sample = W_dist.rsample(key1)
        
        # Predict with sampled weights + noise
        mean_pred = (W_sample @ features[..., None]).squeeze(-1)
        return mean_pred
    
    def sample_weight_function(self, key: jax.Array) -> SampledVBLLModel:
        """Sample weights W from posterior and return a deterministic MLP.
        
        This is useful for consistent rollouts where the same weight realization
        should be used throughout the trajectory. Returns a standard NNX module
        that acts as a feedforward neural network with the sampled weights.
        
        Args:
            key: JAX random key for sampling W
            
        Returns:
            SampledVBLLModel: An NNX module with sampled weights acting as a deterministic MLP
        """
        W_dist = self.last_layer.W()
        W_sample = W_dist.rsample(key)
        return SampledVBLLModel(self.feature_extractor, W_sample)
    
    def rollout_mean_trajectory(
        self,
        initial_state: jax.Array,
        n_steps: int,
    ) -> np.ndarray:
        """Rollout trajectory using mean predictions (no sampling).
        
        Args:
            initial_state: Initial state of shape (input_dim,)
            n_steps: Number of steps to rollout
            
        Returns:
            Trajectory of shape (n_steps+1, output_dim) including initial state
        """
        states = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps):
            pred_dist = self.predictive(current_state)
            next_state = pred_dist.mean
            states.append(next_state)
            current_state = next_state
        
        return np.array(states)
    
    def rollout_sampled_trajectory(
        self,
        initial_state: jax.Array,
        n_steps: int,
        key: jax.Array,
        add_noise: bool = False,
    ) -> np.ndarray:
        """Rollout trajectory using a single sample of W.

        Samples W once and uses it throughout the rollout.

        Args:
            initial_state: Initial state of shape (input_dim,)
            n_steps: Number of steps to rollout
            key: Random key for sampling
            add_noise: Whether to add process noise to state transitions

        Returns:
            Trajectory of shape (n_steps+1, output_dim)
        """
        key_w, key_noise = jax.random.split(key)

        # Sample W once
        W_dist = self.last_layer.W()
        W_sample = W_dist.rsample(key_w)
        process_noise_dist = self.last_layer.process_noise()

        states = [initial_state]
        current_state = initial_state

        for step in range(n_steps):
            features = self.feature_extractor(current_state)
            pred_mean = (W_sample @ features[..., None]).squeeze(-1)

            if add_noise:
                key_noise, subkey = jax.random.split(key_noise)
                proc_noise = process_noise_dist.rsample(subkey)
                next_state = pred_mean + proc_noise
            else:
                next_state = pred_mean

            states.append(next_state)
            current_state = next_state

        return np.array(states)
