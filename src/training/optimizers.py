import jax
import optax
from flax import nnx

from src.models.vbll_model import VBLLModel


def get_optimizer(name: str, model: nnx.Module, **kwargs):
    if name == 'vbll':
        return get_vbll_optimizer(model, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")
    
    
def get_vbll_optimizer(
        model: VBLLModel, 
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        clip_grad_norm: float = 1.0
    ):    
    assert isinstance(model, VBLLModel), "Model must be an instance of VBLLModel."

    # NOTE: we already have a divergence term for the variational last layer 
    # parameters, so we use a simple Adam optimizer without weight decay for those 
    # parameters to avoid over-regularization. For the rest of the parameters, we use 
    # AdamW with weight decay.
    param_pytree_for_labels = nnx.state(model, nnx.Param)

    def label_fn(path, _):
        keys = {k.key for k in path if isinstance(k, jax.tree_util.DictKey)}
        if "feature_extractor" in keys:
            return "feature_extractor"
        if "last_layer" in keys:
            return "last_layer"
        return "default"

    param_labels = jax.tree.map_with_path(
        label_fn, 
        param_pytree_for_labels,
        is_leaf=lambda x: isinstance(x, nnx.Param)
    )

    partitioned_opt = optax.partition(
        transforms={
            "feature_extractor": optax.adamw(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
            "last_layer": optax.adam(
                learning_rate=learning_rate,
            ),
            "default": optax.adamw(
                learning_rate=learning_rate,
                weight_decay=weight_decay,
            ),
        },
        param_labels=param_labels,
    )

    if clip_grad_norm is not None:
        optimizer = nnx.Optimizer(
            model,
            optax.chain(
                optax.clip_by_global_norm(clip_grad_norm),
                partitioned_opt,
            ),
            wrt=nnx.Param,
        )
    else:
        optimizer = nnx.Optimizer(
            model,
            partitioned_opt,
            wrt=nnx.Param,
        )
    return optimizer


