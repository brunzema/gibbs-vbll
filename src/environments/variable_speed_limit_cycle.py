import sys
from pathlib import Path

# Add the project root to sys.path to resolve 'src' imports
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))
    
import numpy as np
from scipy.stats import qmc
from typing import Optional, Tuple

from src.environments.base_environment import BaseEnvironment


class VariableSpeedLimitCycleEnv(BaseEnvironment):
    """
    2D Variable-Speed Limit Cycle (VSLC) environment.

    The limit cycle lives at radius R.  Once on the cycle the angular velocity
    follows a sinusoidal modulation that creates alternating fast and slow zones,
    making accurate long-horizon prediction much harder for single-step models.

    Parameters
    ----------
    dt : float
        Euler integration time step.
    noise_std : float
        Standard deviation of isotropic Gaussian process noise added at each step.
    R : float
        Limit cycle radius.
    mu : float
        Radial attraction strength.  Higher values force faster convergence to R.
    omega_0 : float
        Base (mean) angular velocity in rad / time-unit.
    epsilon : float
        Angular velocity modulation depth in [0, 1).
        omega(theta) ranges in [omega_0*(1-epsilon), omega_0*(1+epsilon)].
        epsilon=0.85 gives a 12x speed ratio between fast and slow zones.
    n_modes : int
        Number of fast/slow zone *pairs* per full revolution.
        n_modes=2 places 2 fast peaks and 2 slow troughs per orbit.
    """

    DEFAULT_R       = 1.0
    DEFAULT_MU      = 3.0
    DEFAULT_OMEGA_0 = 2.0
    DEFAULT_EPSILON = 0.85
    DEFAULT_N_MODES = 2

    def __init__(
        self,
        dt:        float = 0.1,
        noise_std: float = 0.01,
        R:         float = DEFAULT_R,
        mu:        float = DEFAULT_MU,
        omega_0:   float = DEFAULT_OMEGA_0,
        epsilon:   float = DEFAULT_EPSILON,
        n_modes:   int   = DEFAULT_N_MODES,
    ):
        super().__init__(dt=dt, noise_std=noise_std)
        self.R       = R
        self.mu      = mu
        self.omega_0 = omega_0
        self.epsilon = epsilon
        self.n_modes = n_modes

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return 2

    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Compute dx/dt at state x.

        Decomposes into:
          (a) Radial term  : linear attraction toward the limit cycle radius R.
          (b) Angular term : rotation with state-dependent angular velocity.

        Args:
            x: State vector of shape (2,).

        Returns:
            Velocity vector dx/dt of shape (2,).
        """
        r     = np.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-8
        omega = self.angular_velocity(x)

        # Radial: pull toward / push away from limit cycle
        radial = self.mu * (self.R - r)

        dx1 = radial * x[0] / r - omega * x[1]
        dx2 = radial * x[1] / r + omega * x[0]
        return np.array([dx1, dx2])

    def sample_initial_state(
        self,
        rng:          np.random.RandomState,
        scale:        float          = 1.5,
        bounds:       Optional[np.ndarray] = None,
        method:       str            = "uniform",
        sobol_engine: Optional[qmc.Sobol] = None,
    ) -> np.ndarray:
        """
        Sample a random initial state.

        Args:
            rng:          NumPy random state.
            scale:        Half-width for uniform/gaussian sampling.
            bounds:       Optional (2, 2) array [[x1_min, x1_max], [x2_min, x2_max]].
                          Takes precedence over scale/method.
            method:       "uniform" | "gaussian" | "sobol".
            sobol_engine: Optional Sobol engine; created internally if None.

        Returns:
            Initial state of shape (2,).
        """
        if bounds is not None:
            return np.array([
                rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(2)
            ])

        if method == "sobol":
            if sobol_engine is None:
                sobol_engine = qmc.Sobol(d=2, scramble=True)
            u = sobol_engine.random(1)[0]
            return (u * 2.0 - 1.0) * scale

        if method == "gaussian":
            return rng.randn(2) * scale

        # Default: uniform in [-scale, scale]^2
        return rng.uniform(-scale, scale, size=2)

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def set_params(
        self,
        R:       Optional[float] = None,
        mu:      Optional[float] = None,
        omega_0: Optional[float] = None,
        epsilon: Optional[float] = None,
        n_modes: Optional[int]   = None,
    ) -> None:
        """Update any subset of system parameters in-place."""
        if R       is not None: self.R       = R
        if mu      is not None: self.mu      = mu
        if omega_0 is not None: self.omega_0 = omega_0
        if epsilon is not None: self.epsilon = epsilon
        if n_modes is not None: self.n_modes = n_modes

    def angular_velocity(self, x: np.ndarray) -> float:
        """
        Compute the state-dependent angular velocity.

            omega(theta) = omega_0 * (1 + epsilon * sin(n_modes * theta))

        Args:
            x: State vector of shape (2,).

        Returns:
            Angular velocity in rad/time-unit.
        """
        theta = np.arctan2(x[1], x[0])
        return self.omega_0 * (1.0 + self.epsilon * np.sin(self.n_modes * theta))

    def angular_velocity_field(
        self, grid_points: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the angular velocity at an array of states.

        Useful for plotting the fast/slow zone structure.

        Args:
            grid_points: Array of shape (N, 2).

        Returns:
            Angular velocities of shape (N,).
        """
        thetas = np.arctan2(grid_points[:, 1], grid_points[:, 0])
        return self.omega_0 * (1.0 + self.epsilon * np.sin(self.n_modes * thetas))

    def slow_zone_fraction(self) -> float:
        """
        Fraction of the orbit period spent in the slow zone.

        Because the time spent at angle theta is proportional to 1/omega(theta),
        the fraction of time in the slow half [omega < omega_0] is:

            F_slow = (1/T) * integral_{slow half} dt
                   = integral_{slow half} dtheta / omega(theta)  /
                     integral_0^{2pi}  dtheta / omega(theta)

        This is computed numerically.

        Returns:
            Fraction in [0, 1].
        """
        thetas = np.linspace(0, 2 * np.pi, 10_000)
        omegas = self.omega_0 * (
            1.0 + self.epsilon * np.sin(self.n_modes * thetas)
        )
        inv_omega     = 1.0 / omegas
        slow_mask     = omegas < self.omega_0
        return float(np.trapezoid(inv_omega[slow_mask]) / np.trapezoid(inv_omega))

    def fast_to_slow_speed_ratio(self) -> float:
        """Maximum angular velocity divided by minimum angular velocity."""
        return (1.0 + self.epsilon) / (1.0 - self.epsilon)

    def visualize_vector_field(self, grid_size: int = 25):
        """
        Plots the vector field and streamlines, heavily emphasizing the 
        fast and slow zones using a speed-based colormap.
        """
        import matplotlib.pyplot as plt
        
        limit = self.R + 0.5
        x = np.linspace(-limit, limit, grid_size)
        y = np.linspace(-limit, limit, grid_size)
        X, Y = np.meshgrid(x, y)
        
        U, V = np.zeros(X.shape), np.zeros(Y.shape)
        for i in range(grid_size):
            for j in range(grid_size):
                dx = self.dynamics(np.array([X[i,j], Y[i,j]]))
                U[i,j], V[i,j] = dx[0], dx[1]

        # Calculate speed for the colormap
        speed = np.sqrt(U**2 + V**2)
        
        # Normalize vectors for the quiver plot to show pure direction
        U_norm = U / (speed + 1e-8)
        V_norm = V / (speed + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Streamlines colored by speed ('magma' highlights high speeds well)
        strm = ax.streamplot(X, Y, U, V, color=speed, cmap='magma', density=1.5, linewidth=1.5)
        plt.colorbar(strm.lines, ax=ax, label='Velocity Magnitude ||dx/dt||')
        
        # 2. Normalized Quiver for uniform directional flow
        ax.quiver(X, Y, U_norm, V_norm, color='grey', alpha=0.4, pivot='mid', scale=grid_size * 1.5)

        # 3. Limit Cycle Reference
        theta_grid = np.linspace(0, 2 * np.pi, 200)
        ax.plot(self.R * np.cos(theta_grid), self.R * np.sin(theta_grid), 
                 'k--', alpha=0.5, label=f"Limit Cycle (R={self.R})")

        ax.set_title(f"Variable-Speed Vector Field (12x ratio, n_modes={self.n_modes})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.axis('equal')
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('variable_speed_limit_cycle_vector_field.png', dpi=150)
        plt.show()

    def visualize(self, steps: int = 200):
        """
        Rolls out a trajectory and plots it as a scatter plot. 
        The distance between dots visualizes the variable angular velocity.
        """
        import matplotlib.pyplot as plt
        
        rng = np.random.RandomState(42)
        # Start exactly on the cycle to isolate angular dynamics
        x = np.array([self.R, 0.0]) 
        
        traj = [x.copy()]
        for _ in range(steps):
            v = self.dynamics(x)
            # Euler step with slight noise
            x = x + v * self.dt + self.noise_std * np.sqrt(self.dt) * rng.randn(2)
            traj.append(x.copy())
            
        traj = np.array(traj)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Scatter plot of the trajectory
        # Dense dots = slow zone; Sparse dots = fast zone
        ax1.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), 
                    cmap='viridis', s=15, alpha=0.8, zorder=2)
        
        # Background limit cycle
        theta_grid = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(self.R * np.cos(theta_grid), self.R * np.sin(theta_grid), 
                 'k-', alpha=0.2, zorder=1)

        ax1.set_title("Trajectory Marker Density\n(Constant dt = Variable Spacing)")
        ax1.set_xlabel("x1")
        ax1.set_ylabel("x2")
        ax1.axis('equal')

        # Plot 2: Angular Velocity over time
        thetas = np.arctan2(traj[:, 1], traj[:, 0])
        omegas = self.omega_0 * (1.0 + self.epsilon * np.sin(self.n_modes * thetas))
        
        times = np.arange(steps + 1) * self.dt
        ax2.plot(times, omegas, color='tab:orange', lw=2)
        ax2.fill_between(times, omegas, alpha=0.2, color='tab:orange')
        
        ax2.set_title("Angular Velocity $\omega(t)$ Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel(r"$\omega$ (rad/s)")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('variable_speed_limit_cycle_demo.png', dpi=150)
        plt.show()
        
        
        
if __name__ == "__main__":
    # Initialize the environment with default parameters
    env = VariableSpeedLimitCycleEnv(
        dt=0.1, 
        noise_std=0.01, 
        # n_modes=2,
        # omega_0=2.0,
        # epsilon=0.85
    )
    
    # Run the visualization
    env.visualize_vector_field(grid_size=30)
    env.visualize(steps=60,)