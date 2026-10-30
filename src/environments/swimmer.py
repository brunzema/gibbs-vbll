"""
Swimmer Environment

A 2D dynamical system with tidal drag and vortex dynamics.
The environment features configurable parameters (alpha, eta, kappa) that can be
randomized across trajectories to create diverse behaviors.
"""

import numpy as np
from scipy.stats import qmc
from typing import Any, Dict, Optional, Tuple

from src.environments.base_environment import BaseEnvironment


class SwimmerDynamics(BaseEnvironment):
    """
    2D Tidal Drag + Vortex Dynamics system.
    
    The dynamics combine two components:
    1. Tidal drag: Oscillating flow field with damping
    2. Vortex: Circular flow around a vortex center
    
    Parameters:
        - alpha: Vortex strength (default: 0.7)
        - eta: Tidal damping coefficient (default: 1.0)
        - kappa: Wave number for tidal oscillations (default: 1.5*pi)
    """
    
    # Default parameter values
    DEFAULT_ALPHA = 0.7
    DEFAULT_ETA = 1.0
    DEFAULT_KAPPA = 1.5 * np.pi
    
    # Fixed constants
    FIXED_U0 = 0.4
    FIXED_V0 = 0.2
    FIXED_R = 0.5
    VORTEX_CENTER = np.array([-0.5, 0.5])
    
    def __init__(
        self,
        dt: float = 0.1,
        noise_std: float = 0.05,
        alpha: Optional[float] = DEFAULT_ALPHA,
        eta: Optional[float] = DEFAULT_ETA,
        kappa: Optional[float] = DEFAULT_KAPPA,
        param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        param_list: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the swimmer dynamics system.
        
        Args:
            dt: Time step for integration
            noise_std: Standard deviation of observation noise
            param_bounds: Dictionary defining continuous parameter ranges.
                         Keys can be 'alpha', 'eta', 'kappa'.
                         Example: {'alpha': (0.2, 1.2), 'kappa': (3.0, 6.0)}
            param_list: Dictionary defining discrete parameter sets.
                       Keys can be 'alpha', 'eta', 'kappa'.
                       Example: {'alpha': [0.5, 1.0, 1.5], 'kappa': [3.0, 4.0, 5.0]}
        """
        super().__init__(dt=dt, noise_std=noise_std)
        
        # Handle discrete parameter lists
        self.param_list = {} if param_list is None else param_list
        
        # Handle continuous parameter bounds
        self.param_bounds = {} if param_bounds is None else param_bounds
        
        # Current parameters (can be set via sample_params)
        self.alpha = alpha
        self.eta = eta
        self.kappa = kappa
        
    @property
    def dim(self) -> int:
        return 2
    
    def sample_params(self, rng: np.random.RandomState) -> Tuple[float, float, float]:
        """
        Sample random parameters based on param_list or param_bounds.
        
        Args:
            rng: Random number generator
            
        Returns:
            Tuple of (alpha, eta, kappa)
        """
        def get_param(name: str, default_val: float) -> float:
            # Check for discrete list (precedence)
            if name in self.param_list:
                choices = self.param_list[name]
                return rng.choice(choices)
            
            # Check for continuous range
            if name in self.param_bounds:
                min_v, max_v = self.param_bounds[name]
                return rng.uniform(min_v, max_v)
            
            # Fallback to default
            return default_val
        
        alpha = get_param("alpha", self.DEFAULT_ALPHA)
        eta = get_param("eta", self.DEFAULT_ETA)
        kappa = get_param("kappa", self.DEFAULT_KAPPA)
        
        return alpha, eta, kappa
    
    def set_params(self, alpha: float, eta: float, kappa: float):
        """
        Set the current parameters.
        
        Args:
            alpha: Vortex strength
            eta: Tidal damping coefficient
            kappa: Wave number
        """
        self.alpha = alpha
        self.eta = eta
        self.kappa = kappa
    
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
    
    def _tidal_drag(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate tidal drag velocity component.
        
        Args:
            x: State vector of shape (2,)
            
        Returns:
            Tidal drag velocity of shape (2,)
        """
        px, py = x[0], x[1]
        
        # Oscillating flow field
        c0_x = self.FIXED_U0 * np.sin(self.kappa * py)
        c0_y = self.FIXED_V0 * np.sin(self.kappa * px)
        c0 = np.array([c0_x, c0_y])
        
        # Damping based on flow magnitude
        c0_sq_norm = np.dot(c0, c0)
        damping = 1.0 / (1.0 + self.eta * c0_sq_norm)
        
        return c0 * damping
    
    def _vortex(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate vortex velocity component.
        
        Args:
            x: State vector of shape (2,)
            
        Returns:
            Vortex velocity of shape (2,)
        """
        dxy = x - self.VORTEX_CENTER
        dx, dy = dxy[0], dxy[1]
        r2 = dx**2 + dy**2 + 1e-6
        
        factor = self.alpha * self.FIXED_R**2 / (r2 + self.FIXED_R**2)
        return np.array([-factor * dy, factor * dx])
    
    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the dynamics dx/dt at state x.
        
        Combines tidal drag and vortex components.
        
        Args:
            x: State vector of shape (2,)
            
        Returns:
            Derivative dx/dt of shape (2,)
        """
        v_tidal = self._tidal_drag(x)
        v_vortex = self._vortex(x)
        return v_tidal + v_vortex
    
    def generate_dataset(
        self,
        n_trajectories: int,
        n_steps: int,
        initial_state_scale: float = 0.5,
        initial_state_bounds: Optional[np.ndarray] = None,
        seed: int = 0,
        randomize_params: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from multiple trajectories.
        
        This method overrides the base class to support parameter randomization
        across trajectories.
        
        Args:
            n_trajectories: Number of trajectories to generate
            n_steps: Number of time steps per trajectory
            initial_state_scale: Scale for random initial states (ignored if bounds specified)
            initial_state_bounds: Optional array of shape (2, 2) with [min, max] for each dimension.
                                 Example: np.array([[-1, 1], [-0.5, 0.5]])
            seed: Random seed
            randomize_params: If True, sample new parameters for each trajectory
            
        Returns:
            Tuple of (states, next_states) where:
                - states: Array of shape (n_trajectories * n_steps, 2)
                - next_states: Array of shape (n_trajectories * n_steps, 2)
        """
        rng = np.random.RandomState(seed)
        
        all_states = []
        all_next_states = []
        
        for _ in range(n_trajectories):
            # Optionally randomize parameters for this trajectory
            if randomize_params:
                alpha, eta, kappa = self.sample_params(rng)
                self.set_params(alpha, eta, kappa)
            
            # Random initial state
            x0 = self.sample_initial_state(rng, initial_state_scale, initial_state_bounds)
            
            states, next_states = self.rollout_trajectory(x0, n_steps, rng)
            all_states.append(states)
            all_next_states.append(next_states)
        
        return np.concatenate(all_states, axis=0), np.concatenate(all_next_states, axis=0)


def generate_swimmer_data(
    n_trajectories: int = 20,
    n_steps: int = 50,
    dt: float = 0.1,
    noise_std: float = 0.05,
    seed: int = 0,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    initial_state_bounds: Optional[np.ndarray] = None,
    randomize_params: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate training data from a swimmer dynamics system.
    
    This is a convenience function that wraps the SwimmerDynamics class.
    
    Args:
        n_trajectories: Number of trajectories to generate
        n_steps: Number of time steps per trajectory
        dt: Time step for integration
        noise_std: Standard deviation of observation noise
        seed: Random seed
        param_bounds: Optional parameter bounds for randomization
        initial_state_bounds: Optional array of shape (2, 2) with [min, max] for each dimension
        randomize_params: If True, randomize parameters across trajectories
        
    Returns:
        Tuple of (states, next_states) where:
            - states: Array of shape (n_trajectories * n_steps, 2)
            - next_states: Array of shape (n_trajectories * n_steps, 2)
    """
    system = SwimmerDynamics(
        dt=dt,
        noise_std=noise_std,
        param_bounds=param_bounds
    )
    return system.generate_dataset(
        n_trajectories=n_trajectories,
        n_steps=n_steps,
        seed=seed,
        initial_state_bounds=initial_state_bounds,
        randomize_params=randomize_params
    )
