"""
Base Environment Class

Abstract base class for dynamical systems providing a consistent interface
for trajectory generation, data collection, and visualization.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import numpy as np
from scipy.stats import qmc



class BaseEnvironment(ABC):
    """
    Abstract base class for dynamical system environments.
    
    Provides a consistent interface for:
    - Computing dynamics/velocity fields
    - Stepping forward in time
    - Generating trajectories
    - Creating datasets
    - Deterministic rollouts
    """
    
    def __init__(self, dt: float = 0.1, noise_std: float = 0.05):
        """
        Initialize the environment.
        
        Args:
            dt: Time step for integration
            noise_std: Standard deviation of observation/process noise
        """
        self.dt = dt
        self.noise_std = noise_std
    
    @abstractmethod
    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the dynamics dx/dt at state x.
        
        Args:
            x: State vector
            
        Returns:
            Derivative dx/dt
        """
        pass
    
    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the state space."""
        pass
    
    
    def step(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Perform one step of 4th-order Runge-Kutta integration with noise.
        
        Args:
            x: Current state
            rng: Random number generator
            
        Returns:
            Next state
        """
        dt = self.dt
        f = self.dynamics

        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)

        x_det = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x_next = x_det + self.noise_std * rng.randn(*x.shape)

        return x_next
    
    def rollout_trajectory(
        self,
        initial_state: np.ndarray,
        n_steps: int,
        rng: np.random.RandomState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a single trajectory from an initial state.
        
        Args:
            initial_state: Initial state
            n_steps: Number of time steps
            rng: Random number generator
            
        Returns:
            Tuple of (states, next_states)
        """
        states = []
        next_states = []
        
        x = initial_state.copy()
        for _ in range(n_steps):
            states.append(x.copy())
            x_next = self.step(x, rng)
            next_states.append(x_next.copy())
            x = x_next
        
        return np.array(states), np.array(next_states)
    
    @abstractmethod
    def sample_initial_state(
        self,
        rng: np.random.RandomState,
        scale: float = 1.0,
        bounds: Optional[np.ndarray] = None,
        method: str = "sobol",
        sobol_engine: Optional[qmc.Sobol] = None,
    ) -> np.ndarray:
        """
        Sample a random initial state.
        
        Args:
            rng: Random number generator
            scale: Scale for random initial states (used if bounds not specified)
            bounds: Optional array of shape (state_dim, 2) with [min, max] for each dimension.
                   If provided, overrides the scale parameter.
                   Example: np.array([[-1, 1], [-2, 2]]) for 2D state space
            method: One of {"gaussian", "uniform", "sobol"}
            sobol_engine: Optional Sobol engine (recommended to reuse across calls)
            
        Returns:
            Initial state vector
        """
        pass
    
    def generate_dataset(
        self,
        n_trajectories: int,
        n_steps: int,
        initial_state_scale: float = 1.0,
        initial_state_bounds: Optional[np.ndarray] = None,
        seed: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from multiple trajectories.
        
        Args:
            n_trajectories: Number of trajectories to generate
            n_steps: Number of time steps per trajectory
            initial_state_scale: Scale for random initial states (ignored if bounds specified)
            initial_state_bounds: Optional array of shape (state_dim, 2) with [min, max] for each dimension.
                                 Example: np.array([[-1, 1], [-2, 2]]) for 2D state space
            seed: Random seed
            
        Returns:
            Tuple of (states, next_states)
        """
        rng = np.random.RandomState(seed)
        sobol_engine = qmc.Sobol(d=2, scramble=True)
        
        all_states = []
        all_next_states = []
        
        for _ in range(n_trajectories):
            # Random initial state
            x0 = self.sample_initial_state(rng, initial_state_scale, initial_state_bounds, sobol_engine=sobol_engine)
            
            states, next_states = self.rollout_trajectory(x0, n_steps, rng)
            all_states.append(states)
            all_next_states.append(next_states)
        
        return np.concatenate(all_states, axis=0), np.concatenate(all_next_states, axis=0)
    
    def rollout_deterministic(
        self,
        initial_state: np.ndarray,
        n_steps: int
    ) -> np.ndarray:
        """
        Rollout a deterministic trajectory (no noise).
        
        Args:
            initial_state: Initial state
            n_steps: Number of time steps
            
        Returns:
            States array of shape (n_steps+1, state_dim)
        """
        states = [initial_state.copy()]
        x = initial_state.copy()
        
        for _ in range(n_steps):
            dx = self.dynamics(x)
            x_next = x + self.dt * dx
            states.append(x_next.copy())
            x = x_next
        
        return np.array(states)
