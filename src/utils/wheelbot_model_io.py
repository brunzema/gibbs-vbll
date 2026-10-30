import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
from flax import nnx

from src.models.state_action_delta_vbll_model import StateActionVBLLDeltaModel


def _activation_by_name(name: str):
    activations = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
        "mish": jax.nn.mish,
        "elu": jax.nn.elu,
    }
    key = (name or "mish").lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation '{name}'.")
    return activations[key]


def save_wheelbot_model_bundle(
    path: str,
    model: StateActionVBLLDeltaModel,
    model_init: Dict[str, Any],
    normalization: Dict[str, Any],
    train_losses,
    val_losses,
    metadata: Dict[str, Any] | None = None,
) -> str:
    """Save WheelBot model state and required reconstruction metadata."""
    _, model_state = nnx.split(model)
    save_dict = {
        "model_state": model_state,
        "model_init": model_init,
        "normalization": normalization,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metadata": metadata or {},
    }
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(save_dict, f)
    return str(out_path)


def load_wheelbot_model_bundle(path: str, rng_seed: int = 0) -> Tuple[StateActionVBLLDeltaModel, Dict[str, Any]]:
    """Load WheelBot model bundle and reconstruct the model instance."""
    with open(path, "rb") as f:
        bundle = pickle.load(f)

    init_cfg = dict(bundle["model_init"])
    activation_name = init_cfg.pop("activation_fn_name", "mish")
    init_cfg["activation_fn"] = _activation_by_name(activation_name)
    init_cfg["rngs"] = nnx.Rngs(rng_seed)

    model = StateActionVBLLDeltaModel(**init_cfg)
    nnx.update(model, bundle["model_state"])
    return model, bundle
