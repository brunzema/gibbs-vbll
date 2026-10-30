from typing import List, Optional, Tuple

import numpy as np


def build_rollout_chunks_from_trajectories(
    trajectories: List[dict],
    rollout_length: int,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build all valid contiguous rollout chunks from trajectory dicts."""
    candidate_chunks = []

    for ti, traj in enumerate(trajectories):
        states = np.asarray(traj["states"])
        actions = np.asarray(traj["actions"])

        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"Trajectory {ti} is not 2D")
        if state_dim is not None and states.shape[1] != state_dim:
            raise ValueError(f"Trajectory {ti} state dim mismatch: {states.shape[1]} vs expected {state_dim}")
        if action_dim is not None and actions.shape[1] != action_dim:
            raise ValueError(f"Trajectory {ti} action dim mismatch: {actions.shape[1]} vs expected {action_dim}")

        max_start = min(actions.shape[0] - rollout_length, states.shape[0] - rollout_length - 1)
        if max_start < 0:
            continue

        for start in range(max_start + 1):
            candidate_chunks.append((
                states[start],
                actions[start:start + rollout_length],
                states[start + 1:start + rollout_length + 1],
            ))

    return candidate_chunks
