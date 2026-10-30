"""
Spiral Dynamics Environment

A simple 2D dynamical system with stable limit cycle dynamics.
Dynamics: dx/dt = [-y + 0.1*x*(1 - r^2), x + 0.1*y*(1 - r^2)]
where r^2 = x^2 + y^2
"""

import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from src.environments.base_environment import BaseEnvironment


class SpiralDynamics(BaseEnvironment):
    """
    2D Spiral dynamics system with stable limit cycle.
    
    The dynamics create a stable limit cycle (spiral) with the form:
        dx/dt = -y + 0.1*x*(1 - r^2)
        dy/dt = x + 0.1*y*(1 - r^2)
    where r^2 = x^2 + y^2
    """
    
    def __init__(self, dt: float = 0.1, noise_std: float = 0.05):
        """
        Initialize the spiral dynamics system.
        
        Args:
            dt: Time step for integration
            noise_std: Standard deviation of observation noise
        """
        super().__init__(dt=dt, noise_std=noise_std)
    
    @property
    def dim(self) -> int:
        """Dimensionality of the state space."""
        return 2
    
    def sample_initial_state(
        self,
        rng: np.random.RandomState,
        scale: float = 2.0,
        bounds: Optional[np.ndarray] = None,
        method: str = "sobol",
        sobol_engine: Optional[qmc.Sobol] = None,
    ) -> np.ndarray:
        """
        Sample a random initial state.

        Args:
            rng: Random number generator (used for gaussian/uniform)
            scale: Scale for random initial states (used for gaussian)
            bounds: Optional array of shape (2, 2) with [min, max] for each dimension
            method: One of {"gaussian", "uniform", "sobol"}
            sobol_engine: Optional Sobol engine (recommended to reuse across calls)
            
        Returns:
            Initial state vector of shape (2,)
        """
        if method == "sobol":
            if bounds is None:
                raise ValueError("Sobol sampling requires bounds.")

            if sobol_engine is None:
                sobol_engine = qmc.Sobol(d=2, scramble=True)

            # Sample in [0, 1]^2 then scale to bounds
            u = sobol_engine.random(1)[0]
            return bounds[:, 0] + u * (bounds[:, 1] - bounds[:, 0])

        if bounds is not None:
            # Uniform within bounds
            return np.array([
                rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(2)
            ])

        # Default Gaussian
        return rng.randn(2) * scale
    
    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the dynamics dx/dt at state x.
        
        Args:
            x: State vector of shape (2,)
            
        Returns:
            Derivative dx/dt of shape (2,)
        """
        r_sq = x[0]**2 + x[1]**2
        dx = np.array([
            -x[1] + 0.1 * x[0] * (1 - r_sq),
            x[0] + 0.1 * x[1] * (1 - r_sq)
        ])
        return dx
    
    def step(self, x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """
        Perform one step of Euler integration with noise.
        
        Args:
            x: Current state of shape (2,)
            rng: Random number generator
            
        Returns:
            Next state of shape (2,)
        """
        dx = self.dynamics(x)
        x_next = x + self.dt * dx + self.noise_std * rng.randn(2)
        return x_next


def generate_spiral_dynamics_data(
    n_trajectories: int = 20,
    n_steps: int = 50,
    dt: float = 0.1,
    noise_std: float = 0.05,
    seed: int = 0,
    initial_state_scale: float = 2.0,
    initial_state_bounds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data from a spiral dynamics system.
    
    This is a convenience function that wraps the SpiralDynamics class.
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_steps: Number of time steps per trajectory
        dt: Time step for integration
        noise_std: Standard deviation of observation noise
        seed: Random seed
        initial_state_scale: Scale for random initial states (used if bounds not specified)
        initial_state_bounds: Optional array of shape (2, 2) with [min, max] for each dimension.
                            Example: np.array([[-2, 2], [-2, 2]])
        
    Returns:
        Tuple of (states, next_states) where:
            - states: Array of shape (n_trajectories * n_steps, 2)
            - next_states: Array of shape (n_trajectories * n_steps, 2)
    """
    system = SpiralDynamics(dt=dt, noise_std=noise_std)
    return system.generate_dataset(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        initial_state_scale=initial_state_scale,
        initial_state_bounds=initial_state_bounds,
        seed=seed
    )
