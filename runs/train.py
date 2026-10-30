# Add the parent directory to the Python path (because we are executing hydra from within runs)
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
from omegaconf import DictConfig, OmegaConf
import jax
import jax.numpy as jnp
import numpy as np
import time
import math
from flax import nnx

from src.models.vbll_model import VBLLModel
from src.models import VBLLDeltaModel
from src.utils.trajectory_dataset import (
    generate_trajectories,
    trajectories_to_dataset,
    split_train_test_trajectories,
)
from src.utils.model_management import save_model_and_config, get_activation_fn
from src.training import (
    train_single_step_model,
    train_multi_step_model,
    run_post_train_eval,
)
from src.logging import CsvLogger


def create_gamma_scheduler(cfg):
    """Create gamma scheduler based on config."""
    schedule_type = cfg.training.multi_step.gamma_schedule.type
    gamma_init = cfg.training.multi_step.gamma_schedule.gamma_init
    gamma_final = cfg.training.multi_step.gamma_schedule.gamma_final
    max_epochs = cfg.training.n_epochs

    if schedule_type == 'constant':
        def constant_schedule(_epoch):
            return jnp.array(gamma_init)
        return constant_schedule
    elif schedule_type == 'linear':
        def linear_schedule(epoch):
            t = epoch / max_epochs
            return jnp.array(gamma_init + t * (gamma_final - gamma_init))
        return linear_schedule
    elif schedule_type == 'sigmoid':
        k = cfg.training.multi_step.gamma_schedule.k
        transition_fraction = cfg.training.multi_step.gamma_schedule.get('transition_fraction', 1.0)
        def sigmoid_schedule(epoch):
            transition_epochs = max_epochs * transition_fraction
            t = epoch / transition_epochs
            s = 1.0 / (1.0 + jnp.exp(-k * (t - 0.5)))
            return jnp.array(gamma_init + s * (gamma_final - gamma_init))
        return sigmoid_schedule
    else:
        raise ValueError(f"Unknown gamma schedule type: {schedule_type}")


# Registering the config path with Hydra
@hydra.main(config_path="../config", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """
    Train a VBLL dynamics model on a simulated environment.

    The loss is selected with the training config group:
        single_step_elbo               – s-NLL on (x_t, x_{t+1}) pairs
        single_step_elbo_hbatch        – s-NLL with batch size 64 * H
        multi_step_elbo                – m-NLL, teacher-forced
        multi_step_factored_crps       – m-CRPS, teacher-forced (sampled W)
        multi_step_kde                 – m-KDE on autoregressive rollouts
        multi_step_ar_crps             – m-arCRPS on autoregressive rollouts
        multi_step_ar_elbo             – m-arNLL on autoregressive rollouts

    Example:
        python runs/train.py +experiment=paper \\
            environment=variable_speed_limit_cycle training=multi_step_ar_crps
    """

    print("=" * 80)
    print("VBLL Model Training with Hydra")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    log_pwd_with_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"\nLog Time: {log_pwd_with_time}")

    np.random.seed(cfg.seed)

    # ========================================================================
    # STEP 1: Instantiate environment
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 1: Creating environment")
    print("=" * 80)
    environment = hydra.utils.instantiate(cfg.environment.environment)
    initial_state_bounds = np.array(cfg.environment.initial_state_bounds)
    print(f"Environment: {cfg.environment.name}")
    print(f"Initial state bounds: {initial_state_bounds.tolist()}")

    state_dim = initial_state_bounds.shape[0]
    print(f"State dimension: {state_dim}")

    # ========================================================================
    # STEP 2: Generate trajectories
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 2: Generating trajectories")
    print("=" * 80)
    all_trajectories = generate_trajectories(
        environment=environment,
        n_trajectories=cfg.environment.data.trajectories.n_trajectories,
        traj_length=cfg.environment.data.trajectories.traj_length,
        initial_state_bounds=initial_state_bounds,
        seed=cfg.seed,
        deterministic=cfg.environment.data.trajectories.deterministic,
    )

    print(f"Generated {len(all_trajectories)} trajectories")
    print(f"Each trajectory shape: {all_trajectories[0].shape}")

    train_trajectories, val_trajectories, test_trajectories = split_train_test_trajectories(
        trajectories=all_trajectories,
        test_fraction=cfg.environment.data.trajectories.test_fraction,
        val_fraction=cfg.environment.data.trajectories.val_fraction,
        seed=cfg.seed,
    )

    print(f"Split into {len(train_trajectories)} train, {len(val_trajectories)} val, {len(test_trajectories)} test")

    # ========================================================================
    # STEP 3: Process trajectories based on training mode
    # ========================================================================
    training_name = cfg.training.name
    print("\n" + "=" * 80)
    print(f"Step 3: Processing trajectories for '{training_name}' training")
    print("=" * 80)

    if "single_step" in training_name:
        train_states_array, train_next_states_array = trajectories_to_dataset(
            trajectories=train_trajectories,
            n_future=cfg.environment.data.single_step.n_future,
            n_past=cfg.environment.data.single_step.n_past,
            skip_n=cfg.environment.data.single_step.skip_n,
        )
        val_states_array, val_next_states_array = trajectories_to_dataset(
            trajectories=val_trajectories,
            n_future=cfg.environment.data.single_step.n_future,
            n_past=cfg.environment.data.single_step.n_past,
            skip_n=cfg.environment.data.single_step.skip_n,
        )
        train_inputs = jnp.array(train_states_array[:, 0, :])
        train_targets = jnp.array(train_next_states_array[:, 0, :])
        val_inputs = jnp.array(val_states_array[:, 0, :])
        val_targets = jnp.array(val_next_states_array[:, 0, :])
        print(f"Single-step train samples: {len(train_inputs)}")
        print(f"Single-step val samples: {len(val_inputs)}")

    elif "multi_step" in training_name:
        horizon = cfg.training.multi_step.horizon
        train_states_array, train_targets_array = trajectories_to_dataset(
            trajectories=train_trajectories,
            n_future=horizon,
            n_past=cfg.environment.data.multi_step.n_past,
            skip_n=cfg.environment.data.multi_step.skip_n,
        )
        val_states_array, val_targets_array = trajectories_to_dataset(
            trajectories=val_trajectories,
            n_future=horizon,
            n_past=cfg.environment.data.multi_step.n_past,
            skip_n=horizon,
        )
        train_inputs = jnp.array(train_states_array[:, 0, :])
        train_targets = jnp.array(train_targets_array)
        val_inputs = jnp.array(val_states_array[:, 0, :])
        val_targets = jnp.array(val_targets_array)
        print(f"Multi-step train samples: {len(train_inputs)}")
        print(f"Multi-step val samples: {len(val_inputs)}")
        print(f"Target shape: {train_targets.shape}")

    else:
        raise ValueError(
            f"Unknown training_name: '{training_name}'. "
            f"Expected a single_step_* or multi_step_* training config."
        )

    # ========================================================================
    # STEP 3b: Optional state normalization
    # ========================================================================
    normalize = cfg.environment.get('normalize_states', False)
    input_norm = output_norm = None
    if normalize:
        from src.utils.normalization import compute_normalization_params, NormalizedModelWrapper
        # Compute stats from training inputs only (correct ML practice).
        # Input and output normalizations are stored separately so that future
        # extensions (e.g. state+action inputs) can have different input stats.
        input_norm  = compute_normalization_params(np.array(train_inputs))
        output_norm = compute_normalization_params(np.array(train_inputs))  # same for now
        in_mean  = jnp.array(input_norm['mean'])
        in_std   = jnp.array(input_norm['std'])
        out_mean = jnp.array(output_norm['mean'])
        out_std  = jnp.array(output_norm['std'])
        train_inputs  = (train_inputs  - in_mean)  / in_std
        val_inputs    = (val_inputs    - in_mean)  / in_std
        train_targets = (train_targets - out_mean) / out_std  # broadcasts over H dim
        val_targets   = (val_targets   - out_mean) / out_std
        print(f"\nNormalization enabled:")
        print(f"  input  mean = {input_norm['mean'].round(4)}")
        print(f"  input  std  = {input_norm['std'].round(4)}")

    # ========================================================================
    # STEP 4: Create model
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 4: Creating model")
    print("=" * 80)

    reg_weight = cfg.training.regularization_weight / len(train_inputs)
    # reg_weight = 1. / len(train_inputs)

    print(f"Computed regularization_weight: {reg_weight}")

    activation_fn = get_activation_fn(cfg.model.model.activation_fn)
    hidden_dims = tuple([cfg.model.model.hidden_dim] * cfg.model.model.num_hidden_layers)

    # Prefer the model class declared in the model config; fall back to
    # training-mode heuristic so existing single_step / multi_step configs
    # that use model=vbll continue to work unchanged.
    ModelClass = (
        VBLLDeltaModel
        if "VBLLDeltaModel" in cfg.model.model._target_
        else VBLLModel
    )
    print(f"Model class: {ModelClass.__name__}")
    
    # Keep saved config aligned with actual tensors used for training.
    cfg.model.model.input_dim = state_dim
    cfg.model.model.output_dim = state_dim

    model = ModelClass(
        input_dim=state_dim,
        output_dim=state_dim,
        hidden_dims=hidden_dims,
        feature_dim=cfg.model.model.feature_dim,
        activation_fn=activation_fn,
        regularization_weight=reg_weight,
        prior_scale=cfg.model.model.prior_scale,
        wishart_scale=cfg.model.model.wishart_scale,
        rngs=nnx.Rngs(cfg.seed),
    )

    print("Model created:")
    nnx.display(model)

    # ========================================================================
    # STEP 5: Train model
    # ========================================================================
    print("\n" + "=" * 80)
    print(f"Step 5: Training '{training_name}' model")
    print("=" * 80)

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    logger = CsvLogger.from_config(
        cfg.logger,
        output_dir=save_dir,
        run_config=OmegaConf.to_container(cfg, resolve=True),
    )

    t0 = time.time()

    if "single_step" in training_name:
        train_losses, val_losses = train_single_step_model(
            model=model,
            inputs=train_inputs,
            targets=train_targets,
            val_inputs=val_inputs,
            val_targets=val_targets,
            n_epochs=cfg.training.n_epochs,
            learning_rate=cfg.training.learning_rate,
            batch_size=cfg.training.batch_size,
            weight_decay=cfg.training.weight_decay,
            clip_grad_norm=cfg.training.clip_grad_norm,
            verbose=cfg.training.verbose,
            print_every=cfg.training.print_every,
            logger=logger,
        )

    elif "multi_step" in training_name:
        gamma_scheduler = create_gamma_scheduler(cfg)
        train_losses, val_losses, _ = train_multi_step_model(
            model=model,
            strategy=training_name,
            initial_states=train_inputs,
            target_trajectories=train_targets,
            val_initial_states=val_inputs,
            val_target_trajectories=val_targets,
            n_epochs=cfg.training.n_epochs,
            learning_rate=cfg.training.learning_rate,
            batch_size=cfg.training.batch_size,
            weight_decay=cfg.training.weight_decay,
            clip_grad_norm=cfg.training.clip_grad_norm,
            n_samples=cfg.training.multi_step.n_samples,
            gamma_scheduler=gamma_scheduler,
            n_val_samples=cfg.training.multi_step.n_val_samples,
            verbose=cfg.training.verbose,
            print_every=cfg.training.print_every,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported training mode: '{training_name}'")

    t1 = time.time()
    print(f"\nTraining completed in {t1 - t0:.2f} seconds!")
    finite_val_losses = [float(value) for value in val_losses if math.isfinite(value)]
    logger.log_summary(
        {
            "training/time_seconds": float(t1 - t0),
            "training/epochs_completed": float(len(train_losses)),
            "training/final_train_loss": float(train_losses[-1]),
            "training/final_val_loss": (
                float(finite_val_losses[-1]) if finite_val_losses else float("nan")
            ),
            "training/best_val_loss": (
                float(min(finite_val_losses)) if finite_val_losses else float("nan")
            ),
        }
    )
    
    # ========================================================================
    # STEP 6: Save model parameters and training artifacts
    # ========================================================================
    print("\n" + "=" * 80)
    print("Step 6: Saving model and training artifacts")
    print("=" * 80)
    model_path = save_dir / "vbll_model.pkl"
    save_model_and_config(
        model=model,
        cfg=cfg,
        path=model_path,
        train_losses=train_losses,
        val_losses=val_losses,
        training_time=t1 - t0,
        n_train_samples=len(train_inputs),
        n_val_samples=len(val_inputs),
        input_normalization=input_norm,
        output_normalization=output_norm,
    )
    print(f"Model saved to: {model_path}")


    # ========================================================================
    # Post-training evaluation (quick metrics → local CSV summary)
    # ========================================================================
    if cfg.post_eval.enabled:
        env_dict  = OmegaConf.to_container(cfg.environment.environment, resolve=True)
        noise_std = float(env_dict.get('noise_std', 0.01))
        det_env   = hydra.utils.instantiate({**env_dict, 'noise_std': 0.0})
        noisy_env = hydra.utils.instantiate({**env_dict, 'noise_std': noise_std})
        # Wrap with normalization so post-eval receives/returns original-scale states
        if normalize:
            from src.utils.normalization import NormalizedModelWrapper
            model_for_eval = NormalizedModelWrapper(model, input_norm, output_norm)
        else:
            model_for_eval = model
        run_post_train_eval(
            model               = model_for_eval,
            det_env             = det_env,
            noisy_env           = noisy_env,
            initial_state_bounds= initial_state_bounds,
            logger              = logger,
            rollout_steps       = cfg.post_eval.rollout_steps,
            n_test_states       = cfg.post_eval.n_test_states,
            n_noisy_samples     = cfg.post_eval.n_noisy_samples,
            seed                = cfg.seed,
        )

    logger.close()

    # ========================================================================
    # STEP 7: Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Final train loss: {train_losses[-1]:.6f}")
    if val_losses:
        print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Metrics directory: {save_dir}")

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)



if __name__ == '__main__':
    main()
