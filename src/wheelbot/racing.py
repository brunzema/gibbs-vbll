"""
Data loading, normalization and evaluation helpers for the Wheelbot racing
experiments run by runs/wheelbot_dynamics.py.

Expected data layout:
    data/wheelbot_racing/
      train/<group>/*.csv + *.meta
      val/<group>/*.csv + *.meta
      test/<group>/*.csv + *.meta

Setting env.prepare_split=true moves data from raw_data/ into train|val|test
folders and creates the metadata sidecars required by wheelbot_dataset.Dataset.
"""

import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from wheelbot_dataset import Dataset, to_prediction_dataset

from src.models.state_action_delta_vbll_model import StateActionVBLLDeltaModel
from src.training.loss_functions_with_actions import (
    multi_step_factored_crps_with_actions_val_loss_fn,
)
from src.training.utils import crps_gaussian


FIELDS_STATES = [
    # "/q_yrp/yaw", 
    "/q_yrp/roll", "/q_yrp/pitch",
    "/dq_yrp/yaw_vel", "/dq_yrp/roll_vel", "/dq_yrp/pitch_vel",
    # "/q_DR/drive_wheel", "/q_DR/reaction_wheel",
    "/dq_DR/drive_wheel", "/dq_DR/reaction_wheel",
]


FIELDS_ACTIONS = [
    "/tau_DR_command/drive_wheel",
    "/tau_DR_command/reaction_wheel",
]


def _flatten_1step_array(arr: np.ndarray, name: str) -> jnp.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be [N, D] or [N, 1, D], got shape {arr.shape}")
    return jnp.array(arr)


def normalize(states, actions, next_states, state_mean, state_std, action_mean, action_std):
    return (
        (states - state_mean) / state_std,
        (actions - action_mean) / action_std,
        (next_states - state_mean) / state_std,
    )


def compute_one_step_metrics(
    model: StateActionVBLLDeltaModel,
    X: jax.Array,
    Y: jax.Array,
    state_dim: int,
) -> Dict[str, float]:
    @jax.jit
    def _eval_metrics(m, x, y):
        preds = jax.vmap(m)(x)
        mse = jnp.mean((preds - y) ** 2)
        nll = jnp.mean(jax.vmap(m.val_loss_fn)(x, y))

        def _pred_stats(xi):
            state = xi[:state_dim]
            features = m.feature_extractor(xi)
            delta_dist = m.last_layer.predictive(features)
            return state + delta_dist.mean, delta_dist.scale

        means, scales = jax.vmap(_pred_stats)(x)
        crps = jnp.mean(crps_gaussian(means, scales, y))
        return mse, nll, crps

    mse, nll, crps = _eval_metrics(model, X, Y)
    return {
        "mse_norm": float(mse),
        "nll_norm": float(nll),
        "crps_norm": float(crps),
    }


def make_cut_fn(dt: float, trim_start: float, trim_end: float):
    def _cut(exp):
        cut = (
            exp
            .cut_by_condition(
                start_condition=lambda df: df["/tau_DR_command/reaction_wheel"].abs() > 0,
                end_condition=lambda df: df["/tau_DR_command/reaction_wheel"].abs() == 0,
            )
            .resample(dt=dt)
            .cut_time(start=trim_start, end=trim_end)
        )
        return cut if len(cut.data) >= 3 else None
    return _cut


def _parse_track_map_from_pngs(png_dir: Path) -> Dict[str, int]:
    # e.g. experiment_2026-03-31_10-34-23_track121.png
    regex = re.compile(r"^(experiment_[0-9\-]{10}_[0-9\-]{8})_track(\d+)\.png$")
    mapping: Dict[str, int] = {}
    for png_path in sorted(png_dir.glob("experiment_*_track*.png")):
        m = regex.match(png_path.name)
        if m:
            mapping[m.group(1)] = int(m.group(2))
    return mapping


def _count_existing_split_csvs(root: Path) -> int:
    count = 0
    for split in ("train", "val", "test"):
        split_dir = root / split
        if split_dir.exists():
            count += len(list(split_dir.rglob("*.csv")))
    return count


def _write_meta_file(meta_path: Path, split: str, track_id: int, source_file: str) -> None:
    meta = {
        "experiment_status": "success",
        "wheelbot": "wheelbot-beta-1",
        "surface": "black_pvc",
        "dataset": "wheelbot_racing",
        "track_id": int(track_id),
        "track_name": f"track_{track_id}",
        "split": split,
        "source_file": source_file,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def prepare_racing_split(
    data_root: Path,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> None:
    raw_dir = data_root / "raw_data"
    png_dir = data_root / "pngs"

    raw_csvs = sorted(raw_dir.glob("experiment_*.csv"))
    if not raw_csvs:
        existing = _count_existing_split_csvs(data_root)
        if existing > 0:
            print("No files in raw_data/. Existing split folders detected; skipping split preparation.")
            return
        raise ValueError(
            f"No CSV trajectories found in {raw_dir} and no existing split data in train/val/test."
        )

    track_map = _parse_track_map_from_pngs(png_dir)
    if not track_map:
        raise ValueError(
            f"Could not parse track ids from {png_dir}. "
            "Expected files like experiment_YYYY-MM-DD_hh-mm-ss_trackNN.png"
        )

    by_track: Dict[int, List[Path]] = {}
    missing = []
    for csv_path in raw_csvs:
        exp_name = csv_path.stem
        track_id = track_map.get(exp_name)
        if track_id is None:
            missing.append(exp_name)
            continue
        by_track.setdefault(track_id, []).append(csv_path)

    if missing:
        raise ValueError(
            "Missing track mapping for CSV files (no corresponding png): "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )

    rng = np.random.default_rng(seed)
    moved = {"train": 0, "val": 0, "test": 0}

    for track_id in sorted(by_track):
        files = by_track[track_id]
        rng.shuffle(files)
        n = len(files)

        if n == 1:
            n_train, n_val, n_test = 1, 0, 0
        elif n == 2:
            n_train, n_val, n_test = 1, 1, 0
        else:
            n_val = max(1, int(round(val_fraction * n)))
            n_test = max(1, int(round(test_fraction * n)))
            n_train = n - n_val - n_test
            if n_train < 1:
                deficit = 1 - n_train
                if n_val >= n_test:
                    n_val = max(1, n_val - deficit)
                else:
                    n_test = max(1, n_test - deficit)
                n_train = n - n_val - n_test

        split_assignments = (
            [("train", p) for p in files[:n_train]]
            + [("val", p) for p in files[n_train:n_train + n_val]]
            + [("test", p) for p in files[n_train + n_val:]]
        )

        track_name = f"track_{track_id}"
        for split, src_csv in split_assignments:
            dst_dir = data_root / split / track_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_csv = dst_dir / src_csv.name

            shutil.move(str(src_csv), str(dst_csv))
            _write_meta_file(
                meta_path=dst_csv.with_suffix(".meta"),
                split=split,
                track_id=track_id,
                source_file=src_csv.name,
            )

            png_candidates = list(png_dir.glob(f"{src_csv.stem}_track*.png"))
            if png_candidates:
                shutil.move(str(png_candidates[0]), str(dst_dir / png_candidates[0].name))

            moved[split] += 1

    print("Prepared wheelbot_racing split:")
    for split in ("train", "val", "test"):
        print(f"  {split}: {moved[split]} trajectories moved")


def ensure_split_meta_files(data_root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for csv_path in split_dir.rglob("*.csv"):
            meta_path = csv_path.with_suffix(".meta")
            if meta_path.exists():
                continue
            parent = csv_path.parent.name
            track_match = re.match(r"track_(\d+)", parent)
            track_id = int(track_match.group(1)) if track_match else -1
            _write_meta_file(
                meta_path=meta_path,
                split=split,
                track_id=track_id,
                source_file=csv_path.name,
            )


def load_prediction_split(split_dir: Path, cut_fn):
    ds = Dataset(str(split_dir)).map(cut_fn)
    states, actions, next_states, _ = to_prediction_dataset(
        ds,
        fields_states=FIELDS_STATES,
        fields_actions=FIELDS_ACTIONS,
        N_future=1,
        skip_N=1,
    )

    if states.size == 0:
        raise ValueError(
            f"No usable samples found in {split_dir}. "
            "Check metadata files and cut/resample settings."
        )

    states = _flatten_1step_array(states, "states")
    actions = _flatten_1step_array(actions, "actions")
    next_states = _flatten_1step_array(next_states, "next_states")
    return states, actions, next_states, ds


def extract_full_trajectories(ds) -> List[dict]:
    trajs = []
    for group in ds.groups.values():
        for exp in group.experiments:
            trajs.append({
                "states": exp.to_numpy(FIELDS_STATES),
                "actions": exp.to_numpy(FIELDS_ACTIONS),
            })
    return trajs


def save_model_bundle(
    path: Path,
    model,
    model_init: Dict,
    normalization: Dict,
    train_losses: List[float],
    val_losses: List[float],
    metadata: Dict,
) -> None:
    _, model_state = nnx.split(model)
    payload = {
        "model_state": model_state,
        "model_init": model_init,
        "normalization": normalization,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _to_init_states(arr: np.ndarray) -> jax.Array:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    if arr.ndim != 2:
        raise ValueError(f"Initial states must be [N, D] or [N, 1, D], got {arr.shape}")
    return jnp.array(arr)


def _to_action_sequences(arr: np.ndarray) -> jax.Array:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Action sequences must be [N, H, A], got {arr.shape}")
    return jnp.array(arr)


def _to_target_trajectories(arr: np.ndarray) -> jax.Array:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Target trajectories must be [N, H, D], got {arr.shape}")
    return jnp.array(arr)


def normalize_multistep(
    initial_states: jax.Array,
    action_sequences: jax.Array,
    target_trajectories: jax.Array,
    state_mean: jax.Array,
    state_std: jax.Array,
    action_mean: jax.Array,
    action_std: jax.Array,
):
    return (
        (initial_states - state_mean) / state_std,
        (action_sequences - action_mean) / action_std,
        (target_trajectories - state_mean) / state_std,
    )


def load_multistep_split(
    split_dir: Path,
    cut_fn,
    horizon: int,
    skip_n: int,
):
    ds = Dataset(str(split_dir)).map(cut_fn)
    states, actions, next_states, _ = to_prediction_dataset(
        ds,
        fields_states=FIELDS_STATES,
        fields_actions=FIELDS_ACTIONS,
        N_future=horizon,
        skip_N=skip_n,
    )
    if states.size == 0:
        raise ValueError(
            f"No usable multistep chunks in {split_dir}. "
            f"Try smaller --horizon or smaller trims."
        )

    init_states = _to_init_states(states)
    action_sequences = _to_action_sequences(actions)
    target_trajectories = _to_target_trajectories(next_states)
    return init_states, action_sequences, target_trajectories, ds


def evaluate_teacher_forced_on_test(
    model: StateActionVBLLDeltaModel,
    test_init: jax.Array,
    test_traj: jax.Array,
    test_actions: jax.Array,
    n_val_samples: int,
    seed: int,
):
    N = test_init.shape[0]
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, N)
    losses, _ = jax.vmap(
        lambda x0, traj, acts, k: multi_step_factored_crps_with_actions_val_loss_fn(
            model, x0, traj, acts, n_samples=n_val_samples, gamma=1.0, key=k
        )
    )(test_init, test_traj, test_actions, keys)
    return float(jnp.mean(losses))
