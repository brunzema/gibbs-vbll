"""
Trajectory Dataset Utilities

General utilities for generating trajectories from environments and converting them
into datasets suitable for training sequence prediction models. Supports multi-step 
prediction with configurable history and future horizons.
"""

import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.environments.base_environment import BaseEnvironment


def generate_trajectories(
    environment: "BaseEnvironment",
    n_trajectories: int,
    traj_length: int,
    initial_state_scale: float = 1.0,
    initial_state_bounds: Optional[np.ndarray] = None,
    seed: int = 0,
    method: str = "sobol",
    deterministic: bool = False,
) -> List[np.ndarray]:
    """
    Generate trajectories from an environment.
    
    Creates multiple trajectories by sampling initial states and rolling them out
    using the environment's dynamics. Each trajectory includes the initial state.
    
    Args:
        environment: Environment instance (must have sample_initial_state and rollout methods)
        n_trajectories: Number of trajectories to generate
        traj_length: Number of time steps per trajectory (excluding initial state)
        initial_state_scale: Scale for random initial states (ignored if bounds specified)
        initial_state_bounds: Optional array of shape (state_dim, 2) with [min, max] for each dimension
        seed: Random seed for reproducibility
        deterministic: If True, use deterministic rollout (no noise); if False, use noisy rollout
        
    Returns:
        List of trajectory arrays, each of shape (traj_length+1, state_dim)
        where the first state is the initial state
    
    Example:
        >>> from src.environments.spiral_dynamics import SpiralDynamics
        >>> env = SpiralDynamics()
        >>> trajectories = generate_trajectories(env, n_trajectories=10, traj_length=100)
        >>> print(len(trajectories), trajectories[0].shape)
        10 (101, 2)
    """
    rng = np.random.RandomState(seed)
    
    trajectories = []
    for _ in range(n_trajectories):
        # Sample initial state
        x0 = environment.sample_initial_state(
            rng,
            scale=initial_state_scale,
            bounds=initial_state_bounds,
            method=method
        )
        
        # Rollout trajectory
        if deterministic:
            # Deterministic rollout (no noise)
            full_trajectory = environment.rollout_deterministic(x0, traj_length)
        else:
            # Noisy rollout
            states, _ = environment.rollout_trajectory(x0, traj_length, rng)
            # Include initial state at the beginning
            full_trajectory = np.vstack([x0[np.newaxis, :], states])
        
        trajectories.append(full_trajectory)
    
    return trajectories


def trajectories_to_dataset(
    trajectories: List[np.ndarray],
    n_future: int = 1,
    n_past: int = 1,
    skip_n: int = 1,
    include_actions: bool = False,
    actions: Optional[List[np.ndarray]] = None,
    include_observations: bool = False,
    observations: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Convert a list of trajectories into a multi-step prediction dataset.
    
    Creates overlapping trajectory snippets from time-series data. Each snippet contains:
    - n_past historical states
    - n_past+n_future-1 actions (if provided)
    - n_future future states to predict
    - n_past observations (if provided)
    
    The skip_n parameter controls the stride between consecutive snippets.
    
    Args:
        trajectories: List of trajectory arrays, each of shape (T_i, state_dim)
                     where T_i is the length of trajectory i
        n_future: Number of future timesteps to predict (default: 1)
        n_past: Number of past timesteps to include as context (default: 1)
        skip_n: Stride between consecutive snippets in timesteps (default: 1)
                When skip_n < n_future+n_past, snippets will overlap
        include_actions: Whether to include actions in the dataset
        actions: List of action arrays, each of shape (T_i, action_dim)
                Must be provided if include_actions=True
        include_observations: Whether to include observations in the dataset
        observations: List of observation arrays, each of shape (T_i, obs_dim)
                     Must be provided if include_observations=True
    
    Returns:
        Tuple containing:
        - states: Historical states (shape: [D, n_past, state_dim])
        - next_states: Future states to predict (shape: [D, n_future, state_dim])
        - actions: Action sequence (shape: [D, n_past+n_future-1, action_dim]) if include_actions=True
        - observations: Historical observations (shape: [D, n_past, obs_dim]) if include_observations=True
        
        where D is the total number of trajectory snippets across all trajectories.
    
    Raises:
        ValueError: If include_actions=True but actions is None, or
                   if include_observations=True but observations is None, or
                   if trajectory lengths don't match action/observation lengths
    
    Example:
        >>> # Simple 1-step prediction
        >>> trajectories = [np.random.randn(100, 2), np.random.randn(80, 2)]
        >>> states, next_states = trajectories_to_dataset(trajectories, n_future=1, n_past=1)
        >>> print(states.shape, next_states.shape)
        (178, 1, 2) (178, 1, 2)
        
        >>> # Multi-step prediction with actions
        >>> trajectories = [np.random.randn(100, 4)]
        >>> actions = [np.random.randn(100, 2)]
        >>> states, next_states, acts = trajectories_to_dataset(
        ...     trajectories, n_future=5, n_past=3, include_actions=True, actions=actions
        ... )
        >>> print(states.shape, next_states.shape, acts.shape)
        (93, 3, 4) (93, 5, 4) (93, 7, 2)
    """
    # Validation
    if include_actions and actions is None:
        raise ValueError("actions must be provided when include_actions=True")
    if include_observations and observations is None:
        raise ValueError("observations must be provided when include_observations=True")
    
    if include_actions and len(actions) != len(trajectories):
        raise ValueError(f"Number of action sequences ({len(actions)}) must match "
                        f"number of trajectories ({len(trajectories)})")
    if include_observations and len(observations) != len(trajectories):
        raise ValueError(f"Number of observation sequences ({len(observations)}) must match "
                        f"number of trajectories ({len(trajectories)})")
    
    all_states = []
    all_next_states = []
    all_actions = [] if include_actions else None
    all_observations = [] if include_observations else None
    
    for traj_idx, trajectory in enumerate(trajectories):
        traj_length = len(trajectory)
        
        # Validate lengths
        if include_actions and len(actions[traj_idx]) != traj_length:
            raise ValueError(f"Trajectory {traj_idx} has length {traj_length} but "
                           f"actions have length {len(actions[traj_idx])}")
        if include_observations and len(observations[traj_idx]) != traj_length:
            raise ValueError(f"Trajectory {traj_idx} has length {traj_length} but "
                           f"observations have length {len(observations[traj_idx])}")
        
        # Calculate number of snippets we can extract from this trajectory
        snippet_length = n_past + n_future
        if traj_length < snippet_length:
            # Skip trajectories that are too short
            continue
        
        num_snippets = (traj_length - snippet_length) // skip_n + 1
        
        # Extract snippets from this trajectory
        for snippet_idx in range(num_snippets):
            start_idx = snippet_idx * skip_n
            end_past = start_idx + n_past
            end_future = end_past + n_future
            
            # Make sure we don't go out of bounds
            if end_future > traj_length:
                break
            
            # Extract state snippet
            states_snippet = trajectory[start_idx:end_past]
            next_states_snippet = trajectory[end_past:end_future]
            
            all_states.append(states_snippet)
            all_next_states.append(next_states_snippet)
            
            # Extract action snippet if needed
            if include_actions:
                # Actions cover from start to end_future-1 (n_past + n_future - 1 steps)
                actions_snippet = actions[traj_idx][start_idx:end_future - 1]
                all_actions.append(actions_snippet)
            
            # Extract observation snippet if needed
            if include_observations:
                observations_snippet = observations[traj_idx][start_idx:end_past]
                all_observations.append(observations_snippet)
    
    # Handle case where no valid snippets were found
    if len(all_states) == 0:
        state_dim = trajectories[0].shape[1] if len(trajectories) > 0 else 0
        empty_states = np.empty((0, n_past, state_dim))
        empty_next_states = np.empty((0, n_future, state_dim))
        
        result = [empty_states, empty_next_states]
        
        if include_actions:
            action_dim = actions[0].shape[1] if len(actions) > 0 else 0
            result.append(np.empty((0, n_past + n_future - 1, action_dim)))
        
        if include_observations:
            obs_dim = observations[0].shape[1] if len(observations) > 0 else 0
            result.append(np.empty((0, n_past, obs_dim)))
        
        return tuple(result)
    
    # Convert lists to arrays
    states_array = np.array(all_states)
    next_states_array = np.array(all_next_states)
    
    result = [states_array, next_states_array]
    
    if include_actions:
        result.append(np.array(all_actions))
    
    if include_observations:
        result.append(np.array(all_observations))
    
    return tuple(result)


def simple_trajectories_to_dataset(
    trajectories: List[np.ndarray],
    n_steps: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple wrapper for single-step or multi-step prediction without actions.
    
    This is a convenience function that calls trajectories_to_dataset with
    sensible defaults for pure state prediction.
    
    Args:
        trajectories: List of trajectory arrays, each of shape (T_i, state_dim)
        n_steps: Number of future steps to predict (default: 1)
    
    Returns:
        Tuple of (states, next_states) where:
        - states: Historical states (shape: [D, 1, state_dim])
        - next_states: Future states (shape: [D, n_steps, state_dim])
    
    Example:
        >>> trajectories = [np.random.randn(100, 2), np.random.randn(80, 2)]
        >>> states, next_states = simple_trajectories_to_dataset(trajectories, n_steps=5)
        >>> print(states.shape, next_states.shape)
        (174, 1, 2) (174, 5, 2)
    """
    states, next_states = trajectories_to_dataset(
        trajectories=trajectories,
        n_future=n_steps,
        n_past=1,
        skip_n=1,
        include_actions=False,
        include_observations=False
    )
    return states, next_states


def flatten_trajectory_snippets(
    states: np.ndarray,
    next_states: np.ndarray,
    actions: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, ...]:
    """
    Flatten trajectory snippets into single vectors for models that don't handle sequences.
    
    Useful when you want to use a standard regression model instead of a sequence model.
    
    Args:
        states: Historical states of shape (D, n_past, state_dim)
        next_states: Future states of shape (D, n_future, state_dim)
        actions: Optional actions of shape (D, n_past+n_future-1, action_dim)
    
    Returns:
        Tuple of flattened arrays:
        - inputs: Flattened inputs of shape (D, n_past * state_dim) or
                 (D, n_past * state_dim + (n_past+n_future-1) * action_dim) if actions provided
        - targets: Flattened targets of shape (D, n_future * state_dim)
    
    Example:
        >>> states = np.random.randn(100, 3, 4)  # 100 samples, 3 past steps, 4-dim state
        >>> next_states = np.random.randn(100, 2, 4)  # 100 samples, 2 future steps, 4-dim state
        >>> inputs, targets = flatten_trajectory_snippets(states, next_states)
        >>> print(inputs.shape, targets.shape)
        (100, 12) (100, 8)
    """
    batch_size = states.shape[0]
    
    # Flatten states
    states_flat = states.reshape(batch_size, -1)
    next_states_flat = next_states.reshape(batch_size, -1)
    
    if actions is not None:
        actions_flat = actions.reshape(batch_size, -1)
        inputs = np.concatenate([states_flat, actions_flat], axis=1)
    else:
        inputs = states_flat
    
    return inputs, next_states_flat


def split_train_test_trajectories(
    trajectories: List[np.ndarray],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    seed: int = 0
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Split trajectories into training, validation, and test sets.
    
    Args:
        trajectories: List of trajectory arrays
        test_fraction: Fraction of trajectories to use for testing (default: 0.2)
        val_fraction: Fraction of trajectories to use for validation (default: 0.1)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_trajectories, val_trajectories, test_trajectories)
    
    Example:
        >>> trajectories = [np.random.randn(100, 2) for _ in range(50)]
        >>> train_trajs, val_trajs, test_trajs = split_train_test_trajectories(
        ...     trajectories, test_fraction=0.2, val_fraction=0.1
        ... )
        >>> print(len(train_trajs), len(val_trajs), len(test_trajs))
        35 5 10
    """
    rng = np.random.RandomState(seed)
    n_trajectories = len(trajectories)
    n_test = int(n_trajectories * test_fraction)
    n_val = int(n_trajectories * val_fraction)
    
    indices = rng.permutation(n_trajectories)
    test_indices = indices[:n_test]
    val_indices = indices[n_test:n_test + n_val]
    train_indices = indices[n_test + n_val:]
    
    train_trajectories = [trajectories[i] for i in train_indices]
    val_trajectories = [trajectories[i] for i in val_indices]
    test_trajectories = [trajectories[i] for i in test_indices]
    
    return train_trajectories, val_trajectories, test_trajectories
