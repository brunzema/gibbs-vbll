"""
Training utilities for single-step and multi-step VBLL models.

This module provides training functions that can be used with any VBLL model
for both single-step and multi-step trajectory prediction.
"""

import time
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from typing import Any, Callable, Optional, Tuple, List

from src.models.vbll_model import VBLLModel
from src.training.optimizers import get_optimizer
from src.training.training_logger import (
    OptimizationDynamicsTracker,
    log_model_stats,
    log_per_horizon_diagnostics,
)

from src.training.loss_functions import (
    # likelihood-based losses
    multi_step_elbo_loss_fn,
    multi_step_elbo_val_loss_fn,
    #
    multi_step_kde_loss_fn,
    multi_step_kde_val_loss_fn,
    multi_step_factored_crps_loss_fn,
    multi_step_factored_crps_val_loss_fn,
    #
    multi_step_ar_crps_loss_fn,
    multi_step_ar_crps_val_loss_fn,
    multi_step_ar_elbo_loss_fn,
    multi_step_ar_elbo_val_loss_fn,
)
from src.training.loss_functions_with_actions import (
    multi_step_elbo_with_actions_loss_fn,
    multi_step_elbo_with_actions_val_loss_fn,
    multi_step_kde_with_actions_loss_fn,
    multi_step_kde_with_actions_val_loss_fn,
    multi_step_factored_crps_with_actions_loss_fn,
    multi_step_factored_crps_with_actions_val_loss_fn,
    multi_step_ar_crps_with_actions_loss_fn,
    multi_step_ar_crps_with_actions_val_loss_fn,
    multi_step_ar_elbo_with_actions_loss_fn,
    multi_step_ar_elbo_with_actions_val_loss_fn,
)


def _tree_l2_norm(tree) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))


def _gradient_norms(grads) -> tuple[jax.Array, ...]:
    """Return pre-clipping norms for the backbone and posterior components."""
    last_layer_grads = grads["last_layer"]
    covariance_norm = jnp.sqrt(
        _tree_l2_norm(last_layer_grads["W_logdiag"]) ** 2
        + _tree_l2_norm(last_layer_grads["W_offdiag"]) ** 2
    )
    return (
        _tree_l2_norm(grads),
        _tree_l2_norm(grads["feature_extractor"]),
        _tree_l2_norm(last_layer_grads),
        _tree_l2_norm(last_layer_grads["W_mean"]),
        covariance_norm,
        _tree_l2_norm(last_layer_grads["process_noise_log_std"]),
    )


def _log_gradient_stats(
    logger: Any,
    epoch: int,
    losses: list,
    global_norms: list,
    backbone_norms: list,
    last_layer_norms: list,
    posterior_mean_norms: list,
    posterior_covariance_norms: list,
    process_noise_norms: list,
    clip_grad_norm: Optional[float],
) -> dict[str, float]:
    """Log epoch aggregates useful for diagnosing BPTT optimization."""
    loss_values = np.asarray([float(value) for value in losses], dtype=float)
    global_values = np.asarray(
        [float(value) for value in global_norms], dtype=float
    )
    backbone_values = np.asarray(
        [float(value) for value in backbone_norms], dtype=float
    )
    last_layer_values = np.asarray(
        [float(value) for value in last_layer_norms], dtype=float
    )
    posterior_mean_values = np.asarray(
        [float(value) for value in posterior_mean_norms], dtype=float
    )
    posterior_covariance_values = np.asarray(
        [float(value) for value in posterior_covariance_norms], dtype=float
    )
    process_noise_values = np.asarray(
        [float(value) for value in process_noise_norms], dtype=float
    )

    def finite_mean(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else float("nan")

    def finite_max(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.max(finite)) if finite.size else float("nan")

    def finite_p95(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(np.percentile(finite, 95)) if finite.size else float("nan")

    stats = {
        "global_mean": finite_mean(global_values),
        "global_p95": finite_p95(global_values),
        "global_max": finite_max(global_values),
        "backbone_mean": finite_mean(backbone_values),
        "backbone_p95": finite_p95(backbone_values),
        "backbone_max": finite_max(backbone_values),
        "last_layer_mean": finite_mean(last_layer_values),
        "last_layer_p95": finite_p95(last_layer_values),
        "last_layer_max": finite_max(last_layer_values),
        "posterior_mean_mean": finite_mean(posterior_mean_values),
        "posterior_mean_p95": finite_p95(posterior_mean_values),
        "posterior_covariance_mean": finite_mean(posterior_covariance_values),
        "posterior_covariance_p95": finite_p95(posterior_covariance_values),
        "process_noise_mean": finite_mean(process_noise_values),
        "process_noise_p95": finite_p95(process_noise_values),
        "nonfinite_batch_count": float(
            np.count_nonzero(~np.isfinite(loss_values) | ~np.isfinite(global_values))
        ),
    }
    if clip_grad_norm is None:
        stats["clip_fraction"] = 0.0
    else:
        stats["clip_fraction"] = float(
            np.mean((global_values > float(clip_grad_norm)) | ~np.isfinite(global_values))
        )

    logger.add_scalar("Gradients/preclip_global_norm_mean", stats["global_mean"], epoch)
    logger.add_scalar("Gradients/preclip_global_norm_p95", stats["global_p95"], epoch)
    logger.add_scalar("Gradients/preclip_global_norm_max", stats["global_max"], epoch)
    logger.add_scalar("Gradients/preclip_backbone_norm_mean", stats["backbone_mean"], epoch)
    logger.add_scalar("Gradients/preclip_backbone_norm_p95", stats["backbone_p95"], epoch)
    logger.add_scalar("Gradients/preclip_backbone_norm_max", stats["backbone_max"], epoch)
    logger.add_scalar("Gradients/preclip_last_layer_norm_mean", stats["last_layer_mean"], epoch)
    logger.add_scalar("Gradients/preclip_last_layer_norm_p95", stats["last_layer_p95"], epoch)
    logger.add_scalar("Gradients/preclip_last_layer_norm_max", stats["last_layer_max"], epoch)
    logger.add_scalar(
        "Gradients/preclip_posterior_mean_norm_mean",
        stats["posterior_mean_mean"],
        epoch,
    )
    logger.add_scalar(
        "Gradients/preclip_posterior_mean_norm_p95",
        stats["posterior_mean_p95"],
        epoch,
    )
    logger.add_scalar(
        "Gradients/preclip_posterior_covariance_norm_mean",
        stats["posterior_covariance_mean"],
        epoch,
    )
    logger.add_scalar(
        "Gradients/preclip_posterior_covariance_norm_p95",
        stats["posterior_covariance_p95"],
        epoch,
    )
    logger.add_scalar(
        "Gradients/preclip_process_noise_norm_mean",
        stats["process_noise_mean"],
        epoch,
    )
    logger.add_scalar(
        "Gradients/preclip_process_noise_norm_p95",
        stats["process_noise_p95"],
        epoch,
    )
    logger.add_scalar("Gradients/clip_fraction", stats["clip_fraction"], epoch)
    logger.add_scalar(
        "Optimization/nonfinite_batch_count", stats["nonfinite_batch_count"], epoch
    )
    if clip_grad_norm is not None:
        logger.add_scalar("Gradients/clip_threshold", float(clip_grad_norm), epoch)
    return stats

def _get_multi_train_and_val_funcs(strategy: str):
    if strategy in {"multi_step_elbo", "single_step_elbo"}:
        return multi_step_elbo_loss_fn, multi_step_elbo_val_loss_fn
    elif strategy == "multi_step_kde":
        return multi_step_kde_loss_fn, multi_step_kde_val_loss_fn
    elif strategy == "multi_step_factored_crps":
        return multi_step_factored_crps_loss_fn, multi_step_factored_crps_val_loss_fn
    elif strategy == "multi_step_ar_crps":
        return multi_step_ar_crps_loss_fn, multi_step_ar_crps_val_loss_fn
    elif strategy == "multi_step_ar_elbo":
        return multi_step_ar_elbo_loss_fn, multi_step_ar_elbo_val_loss_fn
    else:
        raise ValueError(f"Unsupported training strategy: {strategy}")


def _get_multi_train_and_val_funcs_with_actions(strategy: str):
    if strategy in {"multi_step_elbo", "single_step_elbo"}:
        return multi_step_elbo_with_actions_loss_fn, multi_step_elbo_with_actions_val_loss_fn
    if strategy == "multi_step_kde":
        return multi_step_kde_with_actions_loss_fn, multi_step_kde_with_actions_val_loss_fn
    if strategy == "multi_step_factored_crps":
        return multi_step_factored_crps_with_actions_loss_fn, multi_step_factored_crps_with_actions_val_loss_fn
    if strategy == "multi_step_ar_crps":
        return multi_step_ar_crps_with_actions_loss_fn, multi_step_ar_crps_with_actions_val_loss_fn
    if strategy == "multi_step_ar_elbo":
        return multi_step_ar_elbo_with_actions_loss_fn, multi_step_ar_elbo_with_actions_val_loss_fn
    raise ValueError(f"Unsupported action-conditioned training strategy: {strategy}")


def train_single_step_model(
    model: VBLLModel,
    inputs: jax.Array,
    targets: jax.Array,
    val_inputs: Optional[jax.Array] = None,
    val_targets: Optional[jax.Array] = None,
    n_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    verbose: bool = True,
    print_every: int = 10,
    logger: Optional[Any] = None,
    return_history: bool = False,
) -> Tuple[List[float], List[float]]:
    """
    Train a VBLL model using single-step predictions.

    This trains the model to predict the next state given the current state:
    x_{t+1} = f(x_t)

    Args:
        model: VBLLModel instance to train
        inputs: Training inputs of shape (N, input_dim)
        targets: Training targets of shape (N, output_dim)
        val_inputs: Optional validation inputs
        val_targets: Optional validation targets
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for mini-batch training
        weight_decay: Weight decay for feature extractor
        clip_grad_norm: Gradient clipping threshold
        verbose: Whether to print training progress
        print_every: Print frequency (in epochs)
        logger: Optional scalar metric logger

    Returns:
        Tuple of (train_losses, val_losses) by default.
        If return_history=True, returns (train_losses, val_losses, history_dict).
    """
    optimizer = get_optimizer(
        name="vbll",
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm
    )
    
    @nnx.jit
    def train_step(model, optimizer, batch_inputs, batch_targets):
        def loss_fn(model):
            exp_nll = jnp.mean(
                jax.vmap(model.train_loss_fn)(batch_inputs, batch_targets)
            )
            kl = model.last_layer._gaussian_kl()
            wishart = 0. # model.last_layer._wishart_regulariser()
            reg = model.last_layer.regularization_weight * (kl - wishart)
            return exp_nll + reg, (exp_nll, reg)

        (loss, (data_fit_loss, regularization_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        (
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
        ) = _gradient_norms(grads)
        optimizer.update(model, grads)
        return (
            loss,
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
            data_fit_loss,
            regularization_loss,
        )

    @nnx.jit
    def train_step_fixed_batch(model, optimizer, inputs_shuffled, targets_shuffled, batch_idx):
        start_idx = batch_idx * batch_size
        batch_inputs = jax.lax.dynamic_slice_in_dim(inputs_shuffled, start_idx, batch_size, axis=0)
        batch_targets = jax.lax.dynamic_slice_in_dim(targets_shuffled, start_idx, batch_size, axis=0)
        return train_step(model, optimizer, batch_inputs, batch_targets)

    @nnx.jit
    def eval_step(model, batch_inputs, batch_targets):
        return jnp.mean(
            jax.vmap(model.val_loss_fn)(batch_inputs, batch_targets)
        )
        
    # @nnx.jit
    # def diagnostic_step(model, batch_inputs, batch_targets):
    #     def per_sample(x, y):
    #         state = x[:model._state_dim]
    #         features = model.feature_extractor(x)
    #         delta = y - state
    #         nll = model.last_layer.train_loss_fn(features, delta)
    #         kl = model.last_layer._gaussian_kl()
    #         return nll, kl
    #     nlls, kls = jax.vmap(per_sample)(batch_inputs, batch_targets)
    #     return jnp.mean(nlls), jnp.mean(kls)
            
    train_losses = []
    val_losses = []
    n_samples = inputs.shape[0]
    n_full_batches = n_samples // batch_size
    tail_size = n_samples - n_full_batches * batch_size
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1
    last_val_loss = float('inf')
    history = {
        "epoch": [],
        "epoch_time_sec": [],
        "train_loss": [],
        "data_fit_loss": [],
        "regularization_loss": [],
        "val_loss": [],
        "val_loss_eval_only": [],
        "did_eval": [],
        "grad_norm": [],
        "grad_norm_max": [],
        "backbone_grad_norm": [],
        "last_layer_grad_norm": [],
        "posterior_mean_grad_norm": [],
        "posterior_covariance_grad_norm": [],
        "process_noise_grad_norm": [],
        "clip_fraction": [],
        "nonfinite_batch_count": [],
        "process_noise_std": [],
    }
    dynamics_tracker = (
        OptimizationDynamicsTracker(model, inputs[: min(batch_size, n_samples)])
        if logger
        else None
    )
    training_t0 = time.perf_counter()
    
    # print(model.last_layer.regularization_weight)

    for epoch in range(n_epochs):
        epoch_t0 = time.perf_counter()
        # Shuffle data
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_samples)
        inputs_shuffled = inputs[perm]
        targets_shuffled = targets[perm]

        epoch_losses = []
        epoch_grad_norms = []
        epoch_backbone_grad_norms = []
        epoch_last_layer_grad_norms = []
        epoch_posterior_mean_grad_norms = []
        epoch_posterior_covariance_grad_norms = []
        epoch_process_noise_grad_norms = []
        epoch_data_fit_losses = []
        epoch_regularization_losses = []

        # Train on mini-batches
        for batch_idx in range(n_full_batches):
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step_fixed_batch(
                model, optimizer, inputs_shuffled, targets_shuffled, batch_idx,
            )
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        if tail_size > 0:
            batch_inputs = inputs_shuffled[-tail_size:]
            batch_targets = targets_shuffled[-tail_size:]
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step(model, optimizer, batch_inputs, batch_targets)
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        # print(epoch_losses)
        avg_train_loss = float(jnp.mean(jnp.array(epoch_losses)))
        avg_grad_norm = float(jnp.mean(jnp.array(epoch_grad_norms)))
        avg_data_fit_loss = float(jnp.mean(jnp.array(epoch_data_fit_losses)))
        avg_regularization_loss = float(
            jnp.mean(jnp.array(epoch_regularization_losses))
        )
        train_losses.append(avg_train_loss)

        # Log training metrics
        if logger:
            logger.add_scalar('Loss/train', avg_train_loss, epoch)
            logger.add_scalar('Loss/data_fit', avg_data_fit_loss, epoch)
            logger.add_scalar(
                'Loss/regularization', avg_regularization_loss, epoch
            )

            logger.add_scalar('Gradients/grad_norm', avg_grad_norm, epoch)
            grad_stats = _log_gradient_stats(
                logger,
                epoch,
                epoch_losses,
                epoch_grad_norms,
                epoch_backbone_grad_norms,
                epoch_last_layer_grad_norms,
                epoch_posterior_mean_grad_norms,
                epoch_posterior_covariance_grad_norms,
                epoch_process_noise_grad_norms,
                clip_grad_norm,
            )
            log_model_stats(
                logger,
                model,
                epoch,
                print_every,
                kl_in_objective=True,
            )
            dynamics_tracker.log(logger, model, epoch)
        else:
            grad_stats = {
                "global_max": float(np.max(np.asarray(epoch_grad_norms))),
                "backbone_mean": float(np.mean(np.asarray(epoch_backbone_grad_norms))),
                "last_layer_mean": float(np.mean(np.asarray(epoch_last_layer_grad_norms))),
                "posterior_mean_mean": float(
                    np.mean(np.asarray(epoch_posterior_mean_grad_norms))
                ),
                "posterior_covariance_mean": float(
                    np.mean(np.asarray(epoch_posterior_covariance_grad_norms))
                ),
                "process_noise_mean": float(
                    np.mean(np.asarray(epoch_process_noise_grad_norms))
                ),
                "clip_fraction": float(
                    np.mean(np.asarray(epoch_grad_norms) > clip_grad_norm)
                ) if clip_grad_norm is not None else 0.0,
                "nonfinite_batch_count": float(
                    np.count_nonzero(~np.isfinite(np.asarray(epoch_losses)))
                ),
            }

        val_loss_epoch = last_val_loss
        did_eval = False

        # Compute validation loss
        if epoch % print_every == 0:
            if val_inputs is not None and val_targets is not None:
                val_loss_epoch = float(eval_step(model, val_inputs, val_targets))
                did_eval = True
                last_val_loss = val_loss_epoch
                if val_loss_epoch < best_val_loss:
                    best_val_loss = val_loss_epoch
                    _, best_model_state = nnx.split(model)
                    best_epoch = epoch
                    
            # --- diagnostic ---
            # nll, kl = diagnostic_step(model, inputs[:batch_size], targets[:batch_size])
            # nll, kl = float(nll), float(kl)
            # if logger:
            #     logger.add_scalar('Loss/nll', nll, epoch)
            #     logger.add_scalar('Loss/kl', kl, epoch)

            if verbose:
                msg = f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_train_loss:.4f}"
                if val_loss_epoch < np.inf:
                    msg += f", Val Loss: {val_loss_epoch:.4f}"
                # msg += f", E[NLL]: {nll:.4f}, KL: {kl:.4f}"  # <-- add this
                
                if hasattr(model, 'process_noise_stdv'):
                    process_noise = model.process_noise_stdv
                    pn_str = ", ".join([f"{float(n):.4f}" for n in process_noise])
                    msg += f", Process Noise: [{pn_str}]"
                print(msg)

        if val_loss_epoch < np.inf:
            val_losses.append(float(val_loss_epoch))
        else:
            val_losses.append(np.nan)

        # Log validation loss
        if logger and did_eval and val_loss_epoch < np.inf:
            logger.add_scalar('Loss/validation', float(val_loss_epoch), epoch)

        epoch_time = float(time.perf_counter() - epoch_t0)
        if logger:
            logger.add_scalar("Timing/epoch_seconds", epoch_time, epoch)
            logger.add_scalar(
                "Timing/cumulative_training_seconds",
                float(time.perf_counter() - training_t0),
                epoch,
            )

        # Flush this epoch's logs
        if logger:
            logger.flush(epoch)

        process_noise = []
        if hasattr(model, "process_noise_stdv"):
            process_noise = [float(v) for v in model.process_noise_stdv]
        history["epoch"].append(int(epoch))
        history["epoch_time_sec"].append(epoch_time)
        history["train_loss"].append(float(avg_train_loss))
        history["data_fit_loss"].append(float(avg_data_fit_loss))
        history["regularization_loss"].append(float(avg_regularization_loss))
        history["val_loss"].append(float(val_loss_epoch) if val_loss_epoch < np.inf else np.nan)
        history["val_loss_eval_only"].append(float(val_loss_epoch) if did_eval else np.nan)
        history["did_eval"].append(bool(did_eval))
        history["grad_norm"].append(float(avg_grad_norm))
        history["grad_norm_max"].append(float(grad_stats["global_max"]))
        history["backbone_grad_norm"].append(float(grad_stats["backbone_mean"]))
        history["last_layer_grad_norm"].append(float(grad_stats["last_layer_mean"]))
        history["posterior_mean_grad_norm"].append(
            float(grad_stats["posterior_mean_mean"])
        )
        history["posterior_covariance_grad_norm"].append(
            float(grad_stats["posterior_covariance_mean"])
        )
        history["process_noise_grad_norm"].append(
            float(grad_stats["process_noise_mean"])
        )
        history["clip_fraction"].append(float(grad_stats["clip_fraction"]))
        history["nonfinite_batch_count"].append(
            int(grad_stats["nonfinite_batch_count"])
        )
        history["process_noise_std"].append(process_noise)

    # Restore best model if validation was used
    if best_model_state is not None:
        if verbose:
            print(f"Restoring best model state with Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        nnx.update(model, best_model_state)

    if return_history:
        return train_losses, val_losses, history
    return train_losses, val_losses


def train_multi_step_model(
    model: VBLLModel,
    strategy: str,
    initial_states: jax.Array,
    target_trajectories: jax.Array,
    val_initial_states: Optional[jax.Array] = None,
    val_target_trajectories: Optional[jax.Array] = None,
    n_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    n_samples: int = 1,
    gamma_scheduler: Optional[Callable[[int], float]] = None,
    n_val_samples: int = 10,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    print_every: int = 10,
    logger: Optional[Any] = None,
) -> Tuple[List[float], List[float], List[dict]]:
    """
    Train a VBLL model using multi-step predictions with tempering.

    This trains the model to predict H-step ahead trajectories:
    [x_{t+1}, x_{t+2}, ..., x_{t+H}] = f(x_t)

    Uses tempering (discount factor gamma) to handle compounding errors.

    Args:
        model: VBLLModel instance to train
        initial_states: Initial states of shape (N, input_dim)
        target_trajectories: Target trajectories of shape (N, H, output_dim)
        val_initial_states: Optional validation initial states
        val_target_trajectories: Optional validation target trajectories
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for mini-batch training
        weight_decay: Weight decay for feature extractor
        clip_grad_norm: Gradient clipping threshold
        n_samples: Number of samples for training loss estimation
        gamma_scheduler: Function mapping epoch -> gamma (discount factor)
        n_val_samples: Number of samples for validation loss estimation
        verbose: Whether to print training progress
        print_every: Print frequency (in epochs)
        logger: Optional scalar metric logger

    Returns:
        Tuple of (train_losses, val_losses, val_diagnostics_history)
    """

    train_func, val_func = _get_multi_train_and_val_funcs(strategy)
    
    if "kde" in strategy.lower():
        include_divergence = False
    else:
        include_divergence = True

    # Default gamma scheduler
    if gamma_scheduler is None:
        gamma_scheduler = lambda epoch: 1.
    
    optimizer = get_optimizer(
        name="vbll",
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm
    )
    
    @nnx.jit
    def train_step(model, optimizer, batch_initial, batch_targets, step_gamma, key):
        """Train step for multi-step loss. Returns (loss, grad_norm)."""
        def loss_fn(model):
            keys = jax.random.split(key, batch_initial.shape[0])

            # NLL/score term only — vmap over batch
            losses = jax.vmap(
                lambda x0, traj, k: train_func(
                    model, x0, traj, n_samples=n_samples, gamma=step_gamma, key=k
                )
            )(batch_initial, batch_targets, keys)
            avg_losses = jnp.mean(losses)

            if include_divergence:
                kl = model.last_layer._gaussian_kl()
                reg = model.last_layer.regularization_weight * kl
                return avg_losses + reg, (avg_losses, reg)

            return avg_losses, (avg_losses, jnp.array(0.0))

        (loss, (data_fit_loss, regularization_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        (
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
        ) = _gradient_norms(grads)
        optimizer.update(model, grads)
        return (
            loss,
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
            data_fit_loss,
            regularization_loss,
        )

    @nnx.jit
    def train_step_fixed_batch(
        model,
        optimizer,
        initial_shuffled,
        targets_shuffled,
        step_gamma,
        batch_idx,
        key,
    ):
        start_idx = batch_idx * batch_size
        batch_initial = jax.lax.dynamic_slice_in_dim(initial_shuffled, start_idx, batch_size, axis=0)
        batch_targets = jax.lax.dynamic_slice_in_dim(targets_shuffled, start_idx, batch_size, axis=0)
        return train_step(model, optimizer, batch_initial, batch_targets, step_gamma, key)
    
    @nnx.jit    
    def eval_step(model, batch_initial, batch_targets, step_gamma, key):
        """Validation step using standard multi-step loss."""
        keys = jax.random.split(key, batch_initial.shape[0])
        results = jax.vmap(
            lambda x0, traj, k: val_func(
                model, x0, traj, n_samples=n_val_samples, gamma=step_gamma, key=k
            )
        )(batch_initial, batch_targets, keys)
        # results is (losses, diagnostics_dict)
        losses, diagnostics = results
        # Average losses and diagnostics across batch
        avg_loss = jnp.mean(losses)
        avg_diagnostics = {
            k: jnp.mean(v, axis=0) for k, v in diagnostics.items()
        }
        return avg_loss, avg_diagnostics
 
    train_losses = []
    val_losses = []
    val_diagnostics_history = []  # Store diagnostics over epochs
    n_trajectories = initial_states.shape[0]
    horizon = target_trajectories.shape[1]
    n_full_batches = n_trajectories // batch_size
    tail_size = n_trajectories - n_full_batches * batch_size
    best_model_state = None
    best_val_loss = float('inf')
    best_epoch = -1
    last_val_loss = float('inf')
    evals_without_improvement = 0
    dynamics_tracker = (
        OptimizationDynamicsTracker(
            model, initial_states[: min(batch_size, n_trajectories)]
        )
        if logger
        else None
    )
    training_t0 = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_t0 = time.perf_counter()
        # Shuffle data
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_trajectories)
        initial_shuffled = initial_states[perm]
        targets_shuffled = target_trajectories[perm]
        step_gamma = gamma_scheduler(epoch)
        gamma_float = float(step_gamma)
        effective_horizon_float = (1-gamma_float**horizon)/(1 - gamma_float) if gamma_float < 1 else horizon
        effective_horizon = int(effective_horizon_float)
        epoch_losses = []
        normalized_losses = []
        epoch_grad_norms = []
        epoch_backbone_grad_norms = []
        epoch_last_layer_grad_norms = []
        epoch_posterior_mean_grad_norms = []
        epoch_posterior_covariance_grad_norms = []
        epoch_process_noise_grad_norms = []
        epoch_data_fit_losses = []
        epoch_regularization_losses = []

        # Train on mini-batches
        for batch_idx in range(n_full_batches):
            key, subkey = jax.random.split(key)
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step_fixed_batch(
                model,
                optimizer,
                initial_shuffled,
                targets_shuffled,
                step_gamma,
                batch_idx,
                subkey,
            )
            epoch_losses.append(loss)
            normalized_losses.append(loss / effective_horizon_float)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        if tail_size > 0:
            batch_initial = initial_shuffled[-tail_size:]
            batch_targets = targets_shuffled[-tail_size:]
            key, subkey = jax.random.split(key)
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step(
                model,
                optimizer,
                batch_initial,
                batch_targets,
                step_gamma,
                subkey,
            )
            epoch_losses.append(loss)
            normalized_losses.append(loss / effective_horizon_float)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        avg_train_loss = float(jnp.mean(jnp.array(epoch_losses)))
        avg_grad_norm = float(jnp.mean(jnp.array(epoch_grad_norms)))
        avg_data_fit_loss = float(jnp.mean(jnp.array(epoch_data_fit_losses)))
        avg_regularization_loss = float(
            jnp.mean(jnp.array(epoch_regularization_losses))
        )
        train_losses.append(avg_train_loss)

        # Log training metrics
        if logger:
            # Loss and schedule
            logger.add_scalar('Loss/train_loss', avg_train_loss, epoch)
            logger.add_scalar('Loss/data_fit', avg_data_fit_loss, epoch)
            logger.add_scalar(
                'Loss/regularization', avg_regularization_loss, epoch
            )
            logger.add_scalar('Training/gamma', float(step_gamma), epoch)
            logger.add_scalar('Training/effective_horizon', effective_horizon, epoch)
            logger.add_scalar('Training/effective_horizon_float', effective_horizon_float, epoch)
            logger.add_scalar('Gradients/grad_norm', avg_grad_norm, epoch)
            _log_gradient_stats(
                logger,
                epoch,
                epoch_losses,
                epoch_grad_norms,
                epoch_backbone_grad_norms,
                epoch_last_layer_grad_norms,
                epoch_posterior_mean_grad_norms,
                epoch_posterior_covariance_grad_norms,
                epoch_process_noise_grad_norms,
                clip_grad_norm,
            )
            log_model_stats(
                logger,
                model,
                epoch,
                print_every,
                kl_in_objective=include_divergence,
            )
            dynamics_tracker.log(logger, model, epoch)

        # Compute validation loss
        val_loss_epoch = last_val_loss
        did_eval = False
        val_diagnostics = None
        if epoch % print_every == 0:
            if val_initial_states is not None and val_target_trajectories is not None:
                key, subkey = jax.random.split(key)
                val_loss, val_diagnostics = eval_step(
                    model, val_initial_states, val_target_trajectories, step_gamma, subkey
                )
                val_loss_epoch = float(val_loss)
                last_val_loss = val_loss_epoch
                did_eval = True

                if val_loss_epoch < (best_val_loss - float(early_stopping_min_delta)):
                    best_val_loss = val_loss_epoch
                    _, best_model_state = nnx.split(model)
                    best_epoch = epoch
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

            if verbose:
                msg = f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_train_loss:.4f}"
                if val_loss_epoch < np.inf:
                    msg += f", Val Loss: {val_loss_epoch:.4f}"
                    if val_diagnostics is not None:
                        mean_ssr = float(jnp.mean(val_diagnostics['spread_skill_ratio']))
                        msg += f", Spread/Skill: {mean_ssr:.3f}"
                gamma_float = float(step_gamma)
                msg += f", Gamma: {gamma_float:.4f} (Eff. H: {effective_horizon})"
                if hasattr(model, 'process_noise_stdv'):
                    process_noise = model.process_noise_stdv
                    pn_str = ", ".join([f"{float(n):.4f}" for n in process_noise])
                    msg += f", Process Noise: [{pn_str}]"
                print(msg)

        if val_loss_epoch < np.inf:
            val_losses.append(float(val_loss_epoch))
        else:
            val_losses.append(np.nan)

        # Store diagnostics
        if val_diagnostics is not None:
            val_diagnostics_history.append(val_diagnostics)

        # Log validation loss and diagnostics
        if logger and did_eval and val_loss_epoch < np.inf:
            logger.add_scalar('Loss/val_loss', float(val_loss_epoch), epoch)
            if val_diagnostics is not None:
                # logger.add_scalar('Diagnostics/accuracy', float(val_diagnostics['accuracy']), epoch)
                # logger.add_scalar('Diagnostics/sharpness', float(val_diagnostics['sharpness']), epoch)
                # logger.add_scalar('Diagnostics/latent_mse', float(val_diagnostics['latent_mse']), epoch)
                mean_ssr = float(jnp.mean(val_diagnostics['spread_skill_ratio']))
                logger.add_scalar('Diagnostics/mean_spread_skill_ratio', mean_ssr, epoch)

                log_per_horizon_diagnostics(logger, val_diagnostics, epoch)

        epoch_time = float(time.perf_counter() - epoch_t0)
        if logger:
            logger.add_scalar("Timing/epoch_seconds", epoch_time, epoch)
            logger.add_scalar(
                "Timing/cumulative_training_seconds",
                float(time.perf_counter() - training_t0),
                epoch,
            )

        # Flush this epoch's logs
        if logger:
            logger.flush(epoch)

        if (
            early_stopping_patience is not None
            and did_eval
            and evals_without_improvement >= int(early_stopping_patience)
        ):
            if verbose:
                print(
                    "Early stopping triggered: "
                    f"no validation improvement for {evals_without_improvement} eval checks."
                )
            break

    # Restore best model if validation was used
    if best_model_state is not None:
        if verbose:
            print(f"Restoring best model state with Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        nnx.update(model, best_model_state)

    return train_losses, val_losses, val_diagnostics_history


def train_multi_step_model_with_actions(
    model: VBLLModel,
    strategy: str,
    initial_states: jax.Array,
    target_trajectories: jax.Array,
    action_sequences: jax.Array,
    val_initial_states: Optional[jax.Array] = None,
    val_target_trajectories: Optional[jax.Array] = None,
    val_action_sequences: Optional[jax.Array] = None,
    n_epochs: int = 1000,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    clip_grad_norm: float = 1.0,
    n_samples: int = 16,
    gamma_scheduler: Optional[Callable[[int], float]] = None,
    n_val_samples: int = 16,
    early_stopping_patience: Optional[int] = None,
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    print_every: int = 10,
    logger: Optional[Any] = None,
    return_history: bool = False,
) -> Tuple[List[float], List[float], List[dict]]:
    """Train a state-action VBLL model with action-conditioned multi-step losses."""
    train_func, val_func = _get_multi_train_and_val_funcs_with_actions(strategy)
    include_divergence = "kde" not in strategy.lower()

    if gamma_scheduler is None:
        gamma_scheduler = lambda _epoch: 1.0

    optimizer = get_optimizer(
        name="vbll",
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        clip_grad_norm=clip_grad_norm,
    )

    @nnx.jit
    def train_step(model, optimizer, batch_initial, batch_targets, batch_actions, step_gamma, key):
        def loss_fn(model):
            keys = jax.random.split(key, batch_initial.shape[0])
            losses = jax.vmap(
                lambda x0, traj, acts, k: train_func(
                    model,
                    x0,
                    traj,
                    acts,
                    n_samples=n_samples,
                    gamma=step_gamma,
                    key=k,
                )
            )(batch_initial, batch_targets, batch_actions, keys)
            avg_losses = jnp.mean(losses)
            if include_divergence:
                kl = model.last_layer._gaussian_kl()
                reg = model.last_layer.regularization_weight * kl
                return avg_losses + reg, (avg_losses, reg)
            return avg_losses, (avg_losses, jnp.array(0.0))

        (loss, (data_fit_loss, regularization_loss)), grads = nnx.value_and_grad(
            loss_fn, has_aux=True
        )(model)
        (
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
        ) = _gradient_norms(grads)
        optimizer.update(model, grads)
        return (
            loss,
            grad_norm,
            backbone_grad_norm,
            last_layer_grad_norm,
            posterior_mean_grad_norm,
            posterior_covariance_grad_norm,
            process_noise_grad_norm,
            data_fit_loss,
            regularization_loss,
        )

    @nnx.jit
    def train_step_fixed_batch(
        model,
        optimizer,
        initial_shuffled,
        targets_shuffled,
        actions_shuffled,
        step_gamma,
        batch_idx,
        key,
    ):
        start_idx = batch_idx * batch_size
        batch_initial = jax.lax.dynamic_slice_in_dim(initial_shuffled, start_idx, batch_size, axis=0)
        batch_targets = jax.lax.dynamic_slice_in_dim(targets_shuffled, start_idx, batch_size, axis=0)
        batch_actions = jax.lax.dynamic_slice_in_dim(actions_shuffled, start_idx, batch_size, axis=0)
        return train_step(model, optimizer, batch_initial, batch_targets, batch_actions, step_gamma, key)

    @nnx.jit
    def eval_step(model, batch_initial, batch_targets, batch_actions, step_gamma, key):
        keys = jax.random.split(key, batch_initial.shape[0])
        losses, diagnostics = jax.vmap(
            lambda x0, traj, acts, k: val_func(
                model,
                x0,
                traj,
                acts,
                n_samples=n_val_samples,
                gamma=step_gamma,
                key=k,
            )
        )(batch_initial, batch_targets, batch_actions, keys)
        avg_loss = jnp.mean(losses)
        avg_diagnostics = {k: jnp.mean(v, axis=0) for k, v in diagnostics.items()}
        return avg_loss, avg_diagnostics

    train_losses = []
    val_losses = []
    val_diagnostics_history = []
    n_trajectories = initial_states.shape[0]
    horizon = target_trajectories.shape[1]
    n_full_batches = n_trajectories // batch_size
    tail_size = n_trajectories - n_full_batches * batch_size
    best_model_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    last_val_loss = float("inf")
    evals_without_improvement = 0
    history = {
        "epoch": [],
        "epoch_time_sec": [],
        "train_loss": [],
        "data_fit_loss": [],
        "regularization_loss": [],
        "val_loss": [],
        "val_loss_eval_only": [],
        "did_eval": [],
        "grad_norm": [],
        "grad_norm_max": [],
        "backbone_grad_norm": [],
        "last_layer_grad_norm": [],
        "posterior_mean_grad_norm": [],
        "posterior_covariance_grad_norm": [],
        "process_noise_grad_norm": [],
        "clip_fraction": [],
        "nonfinite_batch_count": [],
        "process_noise_std": [],
        "gamma": [],
        "effective_horizon": [],
    }
    probe_count = min(batch_size, n_trajectories)
    dynamics_tracker = (
        OptimizationDynamicsTracker(
            model,
            jnp.concatenate(
                [
                    initial_states[:probe_count],
                    action_sequences[:probe_count, 0],
                ],
                axis=-1,
            ),
        )
        if logger
        else None
    )
    training_t0 = time.perf_counter()

    for epoch in range(n_epochs):
        epoch_t0 = time.perf_counter()
        key = jax.random.PRNGKey(epoch)
        perm = jax.random.permutation(key, n_trajectories)
        initial_shuffled = initial_states[perm]
        targets_shuffled = target_trajectories[perm]
        actions_shuffled = action_sequences[perm]

        step_gamma = gamma_scheduler(epoch)
        gamma_float = float(step_gamma)
        effective_horizon_float = (1 - gamma_float**horizon) / (1 - gamma_float) if gamma_float < 1 else horizon
        effective_horizon = int(effective_horizon_float)

        epoch_losses = []
        epoch_grad_norms = []
        epoch_backbone_grad_norms = []
        epoch_last_layer_grad_norms = []
        epoch_posterior_mean_grad_norms = []
        epoch_posterior_covariance_grad_norms = []
        epoch_process_noise_grad_norms = []
        epoch_data_fit_losses = []
        epoch_regularization_losses = []
        for batch_idx in range(n_full_batches):
            key, subkey = jax.random.split(key)
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step_fixed_batch(
                model,
                optimizer,
                initial_shuffled,
                targets_shuffled,
                actions_shuffled,
                step_gamma,
                batch_idx,
                subkey,
            )
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        if tail_size > 0:
            batch_initial = initial_shuffled[-tail_size:]
            batch_targets = targets_shuffled[-tail_size:]
            batch_actions = actions_shuffled[-tail_size:]
            key, subkey = jax.random.split(key)
            (
                loss,
                grad_norm,
                backbone_grad_norm,
                last_layer_grad_norm,
                posterior_mean_grad_norm,
                posterior_covariance_grad_norm,
                process_noise_grad_norm,
                data_fit_loss,
                regularization_loss,
            ) = train_step(
                model,
                optimizer,
                batch_initial,
                batch_targets,
                batch_actions,
                step_gamma,
                subkey,
            )
            epoch_losses.append(loss)
            epoch_grad_norms.append(grad_norm)
            epoch_backbone_grad_norms.append(backbone_grad_norm)
            epoch_last_layer_grad_norms.append(last_layer_grad_norm)
            epoch_posterior_mean_grad_norms.append(posterior_mean_grad_norm)
            epoch_posterior_covariance_grad_norms.append(
                posterior_covariance_grad_norm
            )
            epoch_process_noise_grad_norms.append(process_noise_grad_norm)
            epoch_data_fit_losses.append(data_fit_loss)
            epoch_regularization_losses.append(regularization_loss)

        avg_train_loss = float(jnp.mean(jnp.array(epoch_losses)))
        avg_grad_norm = float(jnp.mean(jnp.array(epoch_grad_norms)))
        avg_data_fit_loss = float(jnp.mean(jnp.array(epoch_data_fit_losses)))
        avg_regularization_loss = float(
            jnp.mean(jnp.array(epoch_regularization_losses))
        )
        train_losses.append(avg_train_loss)

        if logger:
            logger.add_scalar("Loss/train_loss", avg_train_loss, epoch)
            logger.add_scalar("Loss/data_fit", avg_data_fit_loss, epoch)
            logger.add_scalar(
                "Loss/regularization", avg_regularization_loss, epoch
            )
            logger.add_scalar("Training/gamma", float(step_gamma), epoch)
            logger.add_scalar("Training/effective_horizon", effective_horizon, epoch)
            logger.add_scalar("Training/effective_horizon_float", effective_horizon_float, epoch)
            logger.add_scalar("Gradients/grad_norm", avg_grad_norm, epoch)
            grad_stats = _log_gradient_stats(
                logger,
                epoch,
                epoch_losses,
                epoch_grad_norms,
                epoch_backbone_grad_norms,
                epoch_last_layer_grad_norms,
                epoch_posterior_mean_grad_norms,
                epoch_posterior_covariance_grad_norms,
                epoch_process_noise_grad_norms,
                clip_grad_norm,
            )
            log_model_stats(
                logger,
                model,
                epoch,
                print_every,
                kl_in_objective=include_divergence,
            )
            dynamics_tracker.log(logger, model, epoch)
        else:
            grad_stats = {
                "global_max": float(np.max(np.asarray(epoch_grad_norms))),
                "backbone_mean": float(np.mean(np.asarray(epoch_backbone_grad_norms))),
                "last_layer_mean": float(np.mean(np.asarray(epoch_last_layer_grad_norms))),
                "posterior_mean_mean": float(
                    np.mean(np.asarray(epoch_posterior_mean_grad_norms))
                ),
                "posterior_covariance_mean": float(
                    np.mean(np.asarray(epoch_posterior_covariance_grad_norms))
                ),
                "process_noise_mean": float(
                    np.mean(np.asarray(epoch_process_noise_grad_norms))
                ),
                "clip_fraction": float(
                    np.mean(np.asarray(epoch_grad_norms) > clip_grad_norm)
                ) if clip_grad_norm is not None else 0.0,
                "nonfinite_batch_count": float(
                    np.count_nonzero(~np.isfinite(np.asarray(epoch_losses)))
                ),
            }

        val_loss_epoch = last_val_loss
        did_eval = False
        val_diagnostics = None
        if epoch % print_every == 0 and val_initial_states is not None and val_target_trajectories is not None and val_action_sequences is not None:
            key, subkey = jax.random.split(key)
            val_loss, val_diagnostics = eval_step(
                model,
                val_initial_states,
                val_target_trajectories,
                val_action_sequences,
                step_gamma,
                subkey,
            )
            val_loss_epoch = float(val_loss)
            last_val_loss = val_loss_epoch
            did_eval = True
            if val_loss_epoch < (best_val_loss - float(early_stopping_min_delta)):
                best_val_loss = val_loss_epoch
                _, best_model_state = nnx.split(model)
                best_epoch = epoch
                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

        if verbose and epoch % print_every == 0:
            msg = f"Epoch {epoch}/{n_epochs} - Train Loss: {avg_train_loss:.4f}"
            if val_loss_epoch < np.inf:
                msg += f", Val Loss: {val_loss_epoch:.4f}"
                if val_diagnostics is not None and "spread_skill_ratio" in val_diagnostics:
                    mean_ssr = float(jnp.mean(val_diagnostics["spread_skill_ratio"]))
                    msg += f", Spread/Skill: {mean_ssr:.3f}"
            msg += f", Gamma: {gamma_float:.4f} (Eff. H: {effective_horizon})"
            if hasattr(model, "process_noise_stdv"):
                process_noise = model.process_noise_stdv
                pn_str = ", ".join([f"{float(n):.4f}" for n in process_noise])
                msg += f", Process Noise: [{pn_str}]"
            print(msg)

        if val_loss_epoch < np.inf:
            val_losses.append(float(val_loss_epoch))
        else:
            val_losses.append(np.nan)

        if val_diagnostics is not None:
            val_diagnostics_history.append(val_diagnostics)

        if logger and did_eval and val_loss_epoch < np.inf:
            logger.add_scalar("Loss/val_loss", float(val_loss_epoch), epoch)
            if val_diagnostics is not None and "spread_skill_ratio" in val_diagnostics:
                mean_ssr = float(jnp.mean(val_diagnostics["spread_skill_ratio"]))
                logger.add_scalar("Diagnostics/mean_spread_skill_ratio", mean_ssr, epoch)
                log_per_horizon_diagnostics(logger, val_diagnostics, epoch)

        epoch_time = float(time.perf_counter() - epoch_t0)
        if logger:
            logger.add_scalar("Timing/epoch_seconds", epoch_time, epoch)
            logger.add_scalar(
                "Timing/cumulative_training_seconds",
                float(time.perf_counter() - training_t0),
                epoch,
            )
            logger.flush(epoch)

        process_noise = []
        if hasattr(model, "process_noise_stdv"):
            process_noise = [float(v) for v in model.process_noise_stdv]
        history["epoch"].append(int(epoch))
        history["epoch_time_sec"].append(epoch_time)
        history["train_loss"].append(float(avg_train_loss))
        history["data_fit_loss"].append(float(avg_data_fit_loss))
        history["regularization_loss"].append(float(avg_regularization_loss))
        history["val_loss"].append(float(val_loss_epoch) if val_loss_epoch < np.inf else np.nan)
        history["val_loss_eval_only"].append(float(val_loss_epoch) if did_eval else np.nan)
        history["did_eval"].append(bool(did_eval))
        history["grad_norm"].append(float(avg_grad_norm))
        history["grad_norm_max"].append(float(grad_stats["global_max"]))
        history["backbone_grad_norm"].append(float(grad_stats["backbone_mean"]))
        history["last_layer_grad_norm"].append(float(grad_stats["last_layer_mean"]))
        history["posterior_mean_grad_norm"].append(
            float(grad_stats["posterior_mean_mean"])
        )
        history["posterior_covariance_grad_norm"].append(
            float(grad_stats["posterior_covariance_mean"])
        )
        history["process_noise_grad_norm"].append(
            float(grad_stats["process_noise_mean"])
        )
        history["clip_fraction"].append(float(grad_stats["clip_fraction"]))
        history["nonfinite_batch_count"].append(
            int(grad_stats["nonfinite_batch_count"])
        )
        history["process_noise_std"].append(process_noise)
        history["gamma"].append(float(gamma_float))
        history["effective_horizon"].append(int(effective_horizon))

        if (
            early_stopping_patience is not None
            and did_eval
            and evals_without_improvement >= int(early_stopping_patience)
        ):
            if verbose:
                print(
                    "Early stopping triggered: "
                    f"no validation improvement for {evals_without_improvement} eval checks."
                )
            break

    if best_model_state is not None:
        if verbose:
            print(f"Restoring best model state with Val Loss: {best_val_loss:.4f} at epoch {best_epoch}")
        nnx.update(model, best_model_state)

    if return_history:
        return train_losses, val_losses, val_diagnostics_history, history
    return train_losses, val_losses, val_diagnostics_history
