import pickle as pkl
import numpy as np
import time
from pathlib import Path
from omegaconf import OmegaConf

import jax

from flax import nnx


def get_activation_fn(name: str):
    """Get activation function by name."""
    activations = {
        'relu': nnx.relu,
        'gelu': nnx.gelu,
        'tanh': nnx.tanh,
        'sigmoid': nnx.sigmoid,
        'silu': nnx.silu,
        'elu': nnx.elu,
        "mish": jax.nn.mish,
    }
    return activations.get(name.lower(), nnx.gelu)


def save_model_and_config(
    model,
    cfg,
    path,
    train_losses,
    val_losses,
    training_time=None,
    n_train_samples=None,
    n_val_samples=None,
    input_normalization=None,
    output_normalization=None,
    **kwargs
):
    """
    Save model and training artifacts (without raw data).
    
    Args:
        model: The trained VBLLModel
        cfg: Hydra configuration object
        path: Path to save the pickle file
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        training_time: Total training time in seconds (optional)
        n_train_samples: Number of training samples (optional)
        n_val_samples: Number of validation samples (optional)
        **kwargs: Additional metadata to save
    """
    
    _, state = nnx.split(model)
    
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare data dictionary
    save_dict = {
        'model_state': state,
        'model_config': cfg.model,
        'config': cfg,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'training_loss': cfg.training.name,
            'n_epochs': cfg.training.n_epochs,
            'batch_size': cfg.training.batch_size,
            'learning_rate': cfg.training.learning_rate,
            'training_time_seconds': training_time,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'n_train_samples': n_train_samples,
            'n_val_samples': n_val_samples,
        },
        # Normalization params (None when normalization is disabled).
        # input_normalization and output_normalization are stored separately
        # to support future extensions where inputs include actions.
        'input_normalization':  input_normalization,
        'output_normalization': output_normalization,
    }
    
    # Add any additional kwargs
    if kwargs:
        save_dict['additional_metadata'] = kwargs
    
    # Save to pickle file
    with open(path, 'wb') as f:
        pkl.dump(save_dict, f)
    
    # Also save a readable YAML version of the config
    config_path = str(path).replace('.pkl', '_config.yaml')
    with open(config_path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))
    
    print(f"\nModel and artifacts saved to: {path}")
    print(f"Config saved to: {config_path}")

    return path


def load_model_and_config(path):
    """
    Load model state and configuration from a saved pickle file.

    Args:
        path: Path to the pickle file (same path used in save_model_and_config)

    Returns:
        dict: Dictionary containing:
            - 'model_state': The nnx model state to be loaded into a model
            - 'config': The full Hydra configuration
            - 'model_config': The model-specific configuration
            - 'train_losses': List of training losses
            - 'val_losses': List of validation losses
            - 'training_metadata': Dictionary of training metadata
            - 'additional_metadata': Any additional metadata (if present)
    """
    with open(path, 'rb') as f:
        save_dict = pkl.load(f)

    print(f"\nModel and artifacts loaded from: {path}")
    print(f"Training mode: {save_dict['training_metadata']['training_loss']}")
    return save_dict