import sys
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import DictConfig, OmegaConf

from src.logging import CsvLogger
from src.models.state_action_delta_vbll_model import StateActionVBLLDeltaModel
from src.training import train_single_step_model
from src.training.trainer import train_multi_step_model_with_actions
from src.wheelbot.racing import (
    FIELDS_ACTIONS,
    FIELDS_STATES,
    compute_one_step_metrics,
    ensure_split_meta_files,
    evaluate_teacher_forced_on_test,
    load_multistep_split,
    load_prediction_split,
    make_cut_fn,
    normalize,
    normalize_multistep,
    prepare_racing_split,
    save_model_bundle,
)

OmegaConf.register_new_resolver("half", lambda x: int(x) // 2, replace=True)


def _activation_by_name(name: str):
    act_name = str(name).lower()
    activations = {
        "relu": jax.nn.relu,
        "gelu": jax.nn.gelu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
        "mish": jax.nn.mish,
        "elu": jax.nn.elu,
    }
    if act_name not in activations:
        raise ValueError(
            f"Unsupported activation '{name}'. "
            f"Supported: {sorted(activations.keys())}"
        )
    return activations[act_name]


def _compute_normalization_params(
    states: jax.Array, actions: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    state_mean = jnp.mean(states, axis=0)
    state_std = jnp.std(states, axis=0) + 1e-8
    action_mean = jnp.mean(actions, axis=0)
    action_std = jnp.std(actions, axis=0) + 1e-8
    return state_mean, state_std, action_mean, action_std


def _to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        return np.asarray(obj).tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_json_and_pkl(base_path: Path, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    json_path = base_path.with_suffix(".json")
    pkl_path = base_path.with_suffix(".pkl")
    with json_path.open("w") as f:
        json.dump(_to_jsonable(payload), f, indent=2)
    with pkl_path.open("wb") as f:
        pickle.dump(payload, f)
    return json_path, pkl_path


def _nanmin_or_none(values: List[float]) -> Optional[float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return None
    return float(np.nanmin(arr))


def _run_single_step(
    cfg: DictConfig,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    cut_fn,
    out_dir: Path,
) -> None:
    logger = CsvLogger(
        output_dir=out_dir,
        run_config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"Loading train split: {train_dir}")
    tr_s, tr_a, tr_ns, _ = load_prediction_split(train_dir, cut_fn)
    print(f"Loading val split:   {val_dir}")
    va_s, va_a, va_ns, _ = load_prediction_split(val_dir, cut_fn)
    print(f"Loading test split:  {test_dir}")
    te_s, te_a, te_ns, ds_test = load_prediction_split(test_dir, cut_fn)

    print(f"Train samples: {tr_s.shape[0]}")
    print(f"Val samples:   {va_s.shape[0]}")
    print(f"Test samples:  {te_s.shape[0]}")
    print(f"State dim: {tr_s.shape[1]}")
    print(f"Action dim: {tr_a.shape[1]}")

    state_mean, state_std, action_mean, action_std = _compute_normalization_params(
        tr_s, tr_a
    )
    tr_s_n, tr_a_n, tr_ns_n = normalize(
        tr_s, tr_a, tr_ns, state_mean, state_std, action_mean, action_std
    )
    va_s_n, va_a_n, va_ns_n = normalize(
        va_s, va_a, va_ns, state_mean, state_std, action_mean, action_std
    )
    te_s_n, te_a_n, te_ns_n = normalize(
        te_s, te_a, te_ns, state_mean, state_std, action_mean, action_std
    )

    X_train = jnp.concatenate([tr_s_n, tr_a_n], axis=-1)
    Y_train = tr_ns_n
    X_val = jnp.concatenate([va_s_n, va_a_n], axis=-1)
    Y_val = va_ns_n
    X_test = jnp.concatenate([te_s_n, te_a_n], axis=-1)
    Y_test = te_ns_n

    model = StateActionVBLLDeltaModel(
        state_dim=int(tr_s.shape[1]),
        action_dim=int(tr_a.shape[1]),
        hidden_dims=tuple(int(v) for v in cfg.model.hidden_dims),
        feature_dim=int(cfg.model.feature_dim),
        activation_fn=_activation_by_name(cfg.model.activation_fn_single_step),
        regularization_weight=float(1.0 / tr_s.shape[0]),
        prior_scale=float(cfg.model.prior_scale),
        wishart_scale=float(cfg.model.wishart_scale),
        rngs=nnx.Rngs(int(cfg.seed)),
    )

    print(f"Training single-step model for {cfg.loss.training.n_epochs} epochs ...")
    train_t0 = time.perf_counter()
    train_losses, val_losses, train_history = train_single_step_model(
        model=model,
        inputs=X_train,
        targets=Y_train,
        val_inputs=X_val,
        val_targets=Y_val,
        n_epochs=int(cfg.loss.training.n_epochs),
        learning_rate=float(cfg.loss.training.learning_rate),
        batch_size=int(cfg.loss.training.batch_size),
        weight_decay=float(cfg.loss.training.weight_decay),
        clip_grad_norm=float(cfg.loss.training.clip_grad_norm),
        verbose=bool(cfg.verbose),
        print_every=int(cfg.print_every),
        logger=logger,
        return_history=True,
    )
    train_time_sec = float(time.perf_counter() - train_t0)
    print(f"Training time: {train_time_sec:.2f}s")

    one_step_metrics = compute_one_step_metrics(
        model=model,
        X=X_test,
        Y=Y_test,
        state_dim=int(tr_s.shape[1]),
    )
    print(f"Test normalized MSE: {one_step_metrics['mse_norm']:.6f}")
    print(f"Test normalized NLL: {one_step_metrics['nll_norm']:.6f}")
    print(f"Test normalized CRPS: {one_step_metrics['crps_norm']:.6f}")
    best_val_loss = _nanmin_or_none(val_losses)
    logger.log_summary(
        {
            "training/time_seconds": train_time_sec,
            "training/epochs_completed": len(train_losses),
            "training/final_train_loss": train_losses[-1],
            "training/best_val_loss": (
                best_val_loss if best_val_loss is not None else float("nan")
            ),
            "test/one_step_mse_normalized": one_step_metrics["mse_norm"],
            "test/one_step_nll_normalized": one_step_metrics["nll_norm"],
            "test/one_step_crps_normalized": one_step_metrics["crps_norm"],
        }
    )
    logger.close()

    model_path = out_dir / str(cfg.output.model_filename_single_step)
    save_model_bundle(
        path=model_path,
        model=model,
        model_init={
            "state_dim": int(tr_s.shape[1]),
            "action_dim": int(tr_a.shape[1]),
            "hidden_dims": tuple(int(v) for v in cfg.model.hidden_dims),
            "feature_dim": int(cfg.model.feature_dim),
            "regularization_weight": float(1.0 / tr_s.shape[0]),
            "prior_scale": float(cfg.model.prior_scale),
            "wishart_scale": float(cfg.model.wishart_scale),
            "activation_fn_name": str(cfg.model.activation_fn_single_step),
        },
        normalization={
            "state_mean": np.asarray(state_mean),
            "state_std": np.asarray(state_std),
            "action_mean": np.asarray(action_mean),
            "action_std": np.asarray(action_std),
        },
        train_losses=train_losses,
        val_losses=val_losses,
        metadata={
            "dataset": str(cfg.env.name),
            "training_mode": str(cfg.loss.strategy),
            "states_labels": list(FIELDS_STATES),
            "actions_labels": list(FIELDS_ACTIONS),
            "dt": float(cfg.env.dt),
            "seed": int(cfg.seed),
            "train_time_sec": float(train_time_sec),
            "test_mse_norm": float(one_step_metrics["mse_norm"]),
            "test_nll_norm": float(one_step_metrics["nll_norm"]),
            "test_crps_norm": float(one_step_metrics["crps_norm"]),
            "best_val_loss_norm": float(best_val_loss)
            if best_val_loss is not None
            else None,
        },
    )
    print(f"Model bundle -> {model_path}")

    logs_payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "train_samples": int(tr_s.shape[0]),
            "val_samples": int(va_s.shape[0]),
            "test_samples": int(te_s.shape[0]),
            "state_dim": int(tr_s.shape[1]),
            "action_dim": int(tr_a.shape[1]),
        },
        "training": {
            "train_time_sec": float(train_time_sec),
            "train_losses": [float(v) for v in train_losses],
            "val_losses": [float(v) for v in val_losses],
            "history": train_history,
            "best_val_loss_norm": float(best_val_loss)
            if best_val_loss is not None
            else None,
        },
        "metrics_norm_test": {"one_step": one_step_metrics},
    }
    logs_json, logs_pkl = _save_json_and_pkl(out_dir / "training_logs", logs_payload)
    metrics_json, metrics_pkl = _save_json_and_pkl(
        out_dir / "one_step_test_metrics", one_step_metrics
    )
    print(f"Logs -> {logs_json} and {logs_pkl}")
    print(f"One-step metrics -> {metrics_json} and {metrics_pkl}")

    summary_path = out_dir / "summary_single_step.txt"
    with summary_path.open("w") as f:
        f.write("Wheelbot Racing VBLL (Hydra) - Single-step\n")
        f.write("==========================================\n")
        f.write(f"Train samples: {tr_s.shape[0]}\n")
        f.write(f"Val samples:   {va_s.shape[0]}\n")
        f.write(f"Test samples:  {te_s.shape[0]}\n")
        f.write(f"State dim:     {tr_s.shape[1]}\n")
        f.write(f"Action dim:    {tr_a.shape[1]}\n")
        f.write(f"Epochs:        {cfg.loss.training.n_epochs}\n")
        f.write(f"Batch size:    {cfg.loss.training.batch_size}\n")
        f.write(f"Learning rate: {cfg.loss.training.learning_rate}\n")
        f.write(f"Train time [s]: {train_time_sec:.2f}\n")
        if best_val_loss is not None:
            f.write(f"Best Val loss (norm): {best_val_loss:.6f}\n")
        f.write(f"Test MSE norm: {one_step_metrics['mse_norm']:.6f}\n")
        f.write(f"Test NLL norm: {one_step_metrics['nll_norm']:.6f}\n")
        f.write(f"Test CRPS norm: {one_step_metrics['crps_norm']:.6f}\n")
        f.write(f"Model path:    {model_path}\n")
        f.write(f"Logs json:     {logs_json}\n")
        f.write(f"Logs pkl:      {logs_pkl}\n")
        f.write(f"Metrics json:  {metrics_json}\n")
        f.write(f"Metrics pkl:   {metrics_pkl}\n")
    print(f"Summary -> {summary_path}")


def _run_multi_step(
    cfg: DictConfig,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    cut_fn,
    out_dir: Path,
) -> None:
    logger = CsvLogger(
        output_dir=out_dir,
        run_config=OmegaConf.to_container(cfg, resolve=True),
    )
    print(f"Loading train split: {train_dir}")
    tr_init, tr_acts, tr_traj, _ = load_multistep_split(
        split_dir=train_dir,
        cut_fn=cut_fn,
        horizon=int(cfg.loss.training.horizon),
        skip_n=int(cfg.loss.training.skip_n),
    )
    print(f"Loading val split:   {val_dir}")
    va_init, va_acts, va_traj, _ = load_multistep_split(
        split_dir=val_dir,
        cut_fn=cut_fn,
        horizon=int(cfg.loss.training.horizon),
        skip_n=int(cfg.loss.training.skip_n),
    )
    print(f"Loading test split:  {test_dir}")
    te_init, te_acts, te_traj, ds_test = load_multistep_split(
        split_dir=test_dir,
        cut_fn=cut_fn,
        horizon=int(cfg.loss.training.horizon),
        skip_n=int(cfg.loss.training.skip_n),
    )

    state_dim = int(tr_init.shape[1])
    action_dim = int(tr_acts.shape[2])
    horizon = int(tr_acts.shape[1])
    print(f"Train chunks: {tr_init.shape[0]}")
    print(f"Val chunks:   {va_init.shape[0]}")
    print(f"Test chunks:  {te_init.shape[0]}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Horizon: {horizon}")

    all_states = jnp.concatenate([tr_init, tr_traj.reshape(-1, state_dim)], axis=0)
    all_actions = tr_acts.reshape(-1, action_dim)
    state_mean, state_std, action_mean, action_std = _compute_normalization_params(
        all_states, all_actions
    )

    tr_init_n, tr_acts_n, tr_traj_n = normalize_multistep(
        tr_init, tr_acts, tr_traj, state_mean, state_std, action_mean, action_std
    )
    va_init_n, va_acts_n, va_traj_n = normalize_multistep(
        va_init, va_acts, va_traj, state_mean, state_std, action_mean, action_std
    )
    te_init_n, te_acts_n, te_traj_n = normalize_multistep(
        te_init, te_acts, te_traj, state_mean, state_std, action_mean, action_std
    )

    model = StateActionVBLLDeltaModel(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=tuple(int(v) for v in cfg.model.hidden_dims),
        feature_dim=int(cfg.model.feature_dim),
        activation_fn=_activation_by_name(cfg.model.activation_fn_multi_step),
        regularization_weight=float(1.0 / tr_init_n.shape[0]),
        prior_scale=float(cfg.model.prior_scale),
        wishart_scale=float(cfg.model.wishart_scale),
        rngs=nnx.Rngs(int(cfg.seed)),
    )

    print(f"Training multi-step model for {cfg.loss.training.n_epochs} epochs ...")
    train_t0 = time.perf_counter()
    train_losses, val_losses, val_diagnostics_history, train_history = train_multi_step_model_with_actions(
        model=model,
        strategy=str(cfg.loss.strategy),
        initial_states=tr_init_n,
        target_trajectories=tr_traj_n,
        action_sequences=tr_acts_n,
        val_initial_states=va_init_n,
        val_target_trajectories=va_traj_n,
        val_action_sequences=va_acts_n,
        n_epochs=int(cfg.loss.training.n_epochs),
        learning_rate=float(cfg.loss.training.learning_rate),
        batch_size=int(cfg.loss.training.batch_size),
        weight_decay=float(cfg.loss.training.weight_decay),
        clip_grad_norm=float(cfg.loss.training.clip_grad_norm),
        n_samples=int(cfg.loss.training.n_samples),
        n_val_samples=int(cfg.loss.training.n_val_samples),
        gamma_scheduler=lambda _epoch: float(cfg.loss.training.gamma),
        verbose=bool(cfg.verbose),
        print_every=int(cfg.print_every),
        logger=logger,
        return_history=True,
    )
    train_time_sec = float(time.perf_counter() - train_t0)
    print(f"Training time: {train_time_sec:.2f}s")

    test_tf_loss = evaluate_teacher_forced_on_test(
        model=model,
        test_init=te_init_n,
        test_traj=te_traj_n,
        test_actions=te_acts_n,
        n_val_samples=int(cfg.loss.training.n_val_samples),
        seed=int(cfg.seed),
    )
    print(f"Test teacher-forced loss (normalized): {test_tf_loss:.6f}")
    best_val_loss = _nanmin_or_none(val_losses)

    te1_s, te1_a, te1_ns, _ = load_prediction_split(test_dir, cut_fn)
    te1_s_n = (te1_s - state_mean) / state_std
    te1_a_n = (te1_a - action_mean) / action_std
    te1_ns_n = (te1_ns - state_mean) / state_std
    X_test_1step = jnp.concatenate([te1_s_n, te1_a_n], axis=-1)
    one_step_metrics = compute_one_step_metrics(
        model=model,
        X=X_test_1step,
        Y=te1_ns_n,
        state_dim=int(state_dim),
    )
    logger.log_summary(
        {
            "training/time_seconds": train_time_sec,
            "training/epochs_completed": len(train_losses),
            "training/final_train_loss": train_losses[-1],
            "training/best_val_loss": (
                best_val_loss if best_val_loss is not None else float("nan")
            ),
            "test/teacher_forced_loss_normalized": test_tf_loss,
            "test/one_step_mse_normalized": one_step_metrics["mse_norm"],
            "test/one_step_nll_normalized": one_step_metrics["nll_norm"],
            "test/one_step_crps_normalized": one_step_metrics["crps_norm"],
        }
    )
    logger.close()
    print(f"Test one-step NLL (norm): {one_step_metrics['nll_norm']:.6f}")
    print(f"Test one-step CRPS (norm): {one_step_metrics['crps_norm']:.6f}")

    model_path = out_dir / str(cfg.output.model_filename_multi_step)
    save_model_bundle(
        path=model_path,
        model=model,
        model_init={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_dims": tuple(int(v) for v in cfg.model.hidden_dims),
            "feature_dim": int(cfg.model.feature_dim),
            "regularization_weight": float(1.0 / tr_init_n.shape[0]),
            "prior_scale": float(cfg.model.prior_scale),
            "wishart_scale": float(cfg.model.wishart_scale),
            "activation_fn_name": str(cfg.model.activation_fn_multi_step),
        },
        normalization={
            "state_mean": np.asarray(state_mean),
            "state_std": np.asarray(state_std),
            "action_mean": np.asarray(action_mean),
            "action_std": np.asarray(action_std),
        },
        train_losses=train_losses,
        val_losses=val_losses,
        metadata={
            "dataset": str(cfg.env.name),
            "training_mode": str(cfg.loss.strategy),
            "states_labels": list(FIELDS_STATES),
            "actions_labels": list(FIELDS_ACTIONS),
            "dt": float(cfg.env.dt),
            "seed": int(cfg.seed),
            "horizon": int(horizon),
            "skip_n": int(cfg.loss.training.skip_n),
            "gamma": float(cfg.loss.training.gamma),
            "train_time_sec": float(train_time_sec),
            "best_val_loss_norm": float(best_val_loss)
            if best_val_loss is not None
            else None,
            "test_teacher_forced_norm": float(test_tf_loss),
            "test_one_step_nll_norm": float(one_step_metrics["nll_norm"]),
            "test_one_step_crps_norm": float(one_step_metrics["crps_norm"]),
        },
    )
    print(f"Model bundle -> {model_path}")

    logs_payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "dataset": {
            "train_chunks": int(tr_init.shape[0]),
            "val_chunks": int(va_init.shape[0]),
            "test_chunks": int(te_init.shape[0]),
            "test_samples_one_step": int(te1_s.shape[0]),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "horizon": int(horizon),
        },
        "training": {
            "train_time_sec": float(train_time_sec),
            "train_losses": [float(v) for v in train_losses],
            "val_losses": [float(v) for v in val_losses],
            "history": train_history,
            "val_diagnostics_history": val_diagnostics_history,
            "best_val_loss_norm": float(best_val_loss)
            if best_val_loss is not None
            else None,
        },
        "metrics_norm_test": {
            "teacher_forced_loss": float(test_tf_loss),
            "one_step": one_step_metrics,
        },
    }
    logs_json, logs_pkl = _save_json_and_pkl(out_dir / "training_logs", logs_payload)
    metrics_json, metrics_pkl = _save_json_and_pkl(
        out_dir / "one_step_test_metrics", one_step_metrics
    )
    print(f"Logs -> {logs_json} and {logs_pkl}")
    print(f"One-step metrics -> {metrics_json} and {metrics_pkl}")

    summary_path = out_dir / "summary_multi_step.txt"
    with summary_path.open("w") as f:
        f.write("Wheelbot Racing VBLL (Hydra) - Multi-step\n")
        f.write("=========================================\n")
        f.write(f"Train chunks: {tr_init.shape[0]}\n")
        f.write(f"Val chunks:   {va_init.shape[0]}\n")
        f.write(f"Test chunks:  {te_init.shape[0]}\n")
        f.write(f"State dim:    {state_dim}\n")
        f.write(f"Action dim:   {action_dim}\n")
        f.write(f"Horizon:      {horizon}\n")
        f.write(f"Skip_N:       {cfg.loss.training.skip_n}\n")
        f.write(f"Epochs:       {cfg.loss.training.n_epochs}\n")
        f.write(f"LR:           {cfg.loss.training.learning_rate}\n")
        f.write(f"Gamma:        {cfg.loss.training.gamma}\n")
        f.write(f"Train time [s]: {train_time_sec:.2f}\n")
        if best_val_loss is not None:
            f.write(f"Best Val loss (norm): {best_val_loss:.6f}\n")
        f.write(f"Test TF (norm): {test_tf_loss:.6f}\n")
        f.write(f"Test one-step NLL (norm): {one_step_metrics['nll_norm']:.6f}\n")
        f.write(f"Test one-step CRPS (norm): {one_step_metrics['crps_norm']:.6f}\n")
        f.write(f"Model path:   {model_path}\n")
        f.write(f"Logs json:     {logs_json}\n")
        f.write(f"Logs pkl:      {logs_pkl}\n")
        f.write(f"Metrics json:  {metrics_json}\n")
        f.write(f"Metrics pkl:   {metrics_pkl}\n")
    print(f"Summary -> {summary_path}")


@hydra.main(
    config_path="../config/wheelbot_dynamics", config_name="default", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    print("=" * 80)
    print("WheelBot Dynamics Learning (Hydra)")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))

    np.random.seed(int(cfg.seed))

    data_root = Path(hydra.utils.to_absolute_path(str(cfg.env.data_root))).resolve()
    out_dir = Path(hydra.utils.to_absolute_path(str(cfg.save_dir))).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "resolved_config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    print(f"Resolved config -> {config_path}")

    if bool(cfg.env.prepare_split):
        prepare_racing_split(
            data_root=data_root,
            val_fraction=float(cfg.env.val_fraction),
            test_fraction=float(cfg.env.test_fraction),
            seed=int(cfg.env.split_seed),
        )
    ensure_split_meta_files(data_root)

    train_dir = data_root / str(cfg.env.train_dir)
    val_dir = data_root / str(cfg.env.val_dir)
    test_dir = data_root / str(cfg.env.test_dir)
    for p in (train_dir, val_dir, test_dir):
        if not p.exists():
            raise FileNotFoundError(f"Split directory not found: {p}")

    cut_fn = make_cut_fn(
        dt=float(cfg.env.dt),
        trim_start=float(cfg.env.trim_start),
        trim_end=float(cfg.env.trim_end),
    )

    mode = str(cfg.loss.mode).lower()
    if mode == "single_step":
        _run_single_step(
            cfg=cfg,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            cut_fn=cut_fn,
            out_dir=out_dir,
        )
    elif mode == "multi_step":
        _run_multi_step(
            cfg=cfg,
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            cut_fn=cut_fn,
            out_dir=out_dir,
        )
    else:
        raise ValueError(f"Unsupported loss.mode='{cfg.loss.mode}'.")


if __name__ == "__main__":
    main()
