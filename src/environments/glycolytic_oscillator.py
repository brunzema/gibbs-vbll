"""
Glycolytic Oscillator Environment

Highly nonlinear and stiff biochemical dynamics modeling yeast glycolysis
(Daniels and Nemenman, 2015).

The system consists of 7 coupled ODEs with a scalar control input u acting
on the first state. Discrete parameter variations are used to define
multiple dynamics models.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

from src.environments.base_environment import BaseEnvironment


class GlycolyticOscillatorDynamics(BaseEnvironment):
    """
    Reference Tracking Glycolytic Oscillator (GO).

    State vector:
        x = [x1, x2, x3, x4, x5, x6, x7]

    Control input:
        u ∈ [-4, 4], injected into x1 dynamics

    Fixed parameters:
        J0, k2, k3, k4, k5, k6, q, N, A, kappa, psi, k

    Discrete parameters:
        k1 ∈ {80, 90, 100}
        K1 ∈ {0.5, 0.75}

    Notes:
        - The system is stiff and highly nonlinear
        - Small integration steps are recommended
    """

    # -----------------------------
    # Fixed system parameters
    # -----------------------------
    J0 = 2.5
    k2 = 6.0
    k3 = 16.0
    k4 = 100.0
    k5 = 1.28
    k6 = 12.0
    q = 4.0
    N = 1.0
    A = 4.0
    kappa = 13.0
    psi = 0.1
    k = 1.8

    # Discrete parameter sets
    DEFAULT_K1_VALUES = [80.0, 90.0, 100.0]
    DEFAULT_K1_HALF_SAT = [0.5, 0.75]

    # State dimensionality
    STATE_DIM = 7

    # State bounds (used for initialization)
    STATE_LOW = np.array([0.15, 0.19, 0.04, 0.10, 0.08, 0.14, 0.05])
    STATE_HIGH = np.array([1.60, 2.16, 0.20, 0.35, 0.30, 2.67, 0.10])

    # Control bounds
    U_MIN = -4.0
    U_MAX = 4.0

    def __init__(
        self,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        param_list: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the glycolytic oscillator dynamics.

        Args:
            dt: Integration time step (small due to stiffness)
            noise_std: Observation noise standard deviation
            param_list: Optional discrete parameter lists.
                        Keys: 'k1', 'K1'
        """
        super().__init__(dt=dt, noise_std=noise_std)

        self.param_list = {} if param_list is None else param_list

        # Current parameters
        self.k1 = self.DEFAULT_K1_VALUES[0]
        self.K1 = self.DEFAULT_K1_HALF_SAT[0]

    @property
    def dim(self) -> int:
        """Dimensionality of the state space."""
        return self.STATE_DIM

    # ------------------------------------------------------------------
    # Parameter handling
    # ------------------------------------------------------------------
    def sample_params(self, rng: np.random.RandomState) -> Tuple[float, float]:
        """
        Sample discrete parameters defining the dynamics model.

        Args:
            rng: Random number generator

        Returns:
            (k1, K1)
        """
        k1_vals = self.param_list.get("k1", self.DEFAULT_K1_VALUES)
        K1_vals = self.param_list.get("K1", self.DEFAULT_K1_HALF_SAT)

        k1 = rng.choice(k1_vals)
        K1 = rng.choice(K1_vals)

        return k1, K1

    def set_params(self, k1: float, K1: float):
        """
        Set the current dynamics parameters.

        Args:
            k1: Reaction rate constant
            K1: Half-saturation constant
        """
        self.k1 = k1
        self.K1 = K1

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def sample_initial_state(
        self,
        rng: np.random.RandomState,
        scale: float = 1.0,
        bounds: Optional[np.ndarray] = None,
        method: str = "uniform",
        **kwargs,
    ) -> np.ndarray:
        """
        Sample a random initial state within biologically valid bounds.

        Args:
            rng: Random number generator
            scale: Ignored (bounds determined by STATE_LOW/STATE_HIGH)
            bounds: Optional array of shape (2, 7) with [low, high] rows,
                    overriding STATE_LOW/STATE_HIGH defaults
            method: Ignored (always uniform sampling)

        Returns:
            Initial state vector of shape (7,)
        """
        if bounds is None:
            low, high = self.STATE_LOW, self.STATE_HIGH
        else:
            # bounds has shape (state_dim, 2): each row is [min, max]
            bounds = np.asarray(bounds)
            low, high = bounds[:, 0], bounds[:, 1]

        return rng.uniform(low, high)

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    def dynamics(self, x: np.ndarray, u: float = 0.0) -> np.ndarray:
        """
        Compute dx/dt for the glycolytic oscillator.

        Args:
            x: State vector of shape (7,)
            u: Control input (scalar)

        Returns:
            Time derivative dx/dt of shape (7,)
        """
        # Clamp control input
        u = np.clip(u, self.U_MIN, self.U_MAX)

        x1, x2, x3, x4, x5, x6, x7 = x

        # Hill-type nonlinearity
        hill = x6**self.q / (1.0 + (x6 / self.K1)**self.q)

        dx1 = self.J0 - self.k1 * x1 * hill + u

        dx2 = (
            2.0 * self.k1 * x1 * hill
            - self.k2 * x2 * (self.N - x5)
            - self.k6 * x2 * x5
        )

        dx3 = self.k2 * x2 * (self.N - x5) - self.k3 * x3 * (self.A - x6)

        dx4 = (
            self.k3 * x3 * (self.A - x6)
            - self.k4 * x4 * x5
            - self.kappa * (x4 - x7)
        )

        dx5 = (
            self.k2 * x2 * (self.N - x5)
            - self.k4 * x4 * x5
            - self.k6 * x2 * x5
        )

        dx6 = (
            -2.0 * self.k1 * x1 * hill
            + 2.0 * self.k3 * x3 * (self.A - x6)
            - self.k5 * x6
        )

        dx7 = self.psi * self.kappa * (x4 - x7) - self.k * x7

        return np.array([dx1, dx2, dx3, dx4, dx5, dx6, dx7])

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------
    def generate_dataset(
        self,
        n_trajectories: int,
        n_steps: int,
        seed: int = 0,
        randomize_params: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate trajectory data from the glycolytic oscillator.

        Args:
            n_trajectories: Number of trajectories
            n_steps: Number of steps per trajectory
            seed: Random seed
            randomize_params: If True, sample new (k1, K1) per trajectory

        Returns:
            (states, next_states)
                states.shape = (N, 7)
                next_states.shape = (N, 7)
        """
        rng = np.random.RandomState(seed)

        all_states = []
        all_next_states = []

        for _ in range(n_trajectories):
            if randomize_params:
                k1, K1 = self.sample_params(rng)
                self.set_params(k1, K1)

            x0 = self.sample_initial_state(rng)
            states, next_states = self.rollout_trajectory(x0, n_steps, rng)

            all_states.append(states)
            all_next_states.append(next_states)

        return np.concatenate(all_states, axis=0), np.concatenate(all_next_states, axis=0)
