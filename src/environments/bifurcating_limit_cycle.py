"""
Bifurcating Limit Cycle Environment
====================================

A 2D limit cycle with angle-dependent radial pitchfork bifurcations.
At most angles the dynamics have a single stable radius R.  In
bifurcation zones the radius splits into two stable branches
R ± δ(θ), and process noise determines which branch the particle
takes.  This creates genuinely **bimodal** multi-step predictive
distributions that the ELBO structurally cannot capture.

Why ELBO multi-step fails here
-------------------------------
The multi-step ELBO evaluates

    L = -E_{q(W)} Σ_h log N(x*_h | x̂_h^W, Σ_ν)

Each term is a unimodal Gaussian centred on one rollout's prediction.
Near a bifurcation, the particle commits to one branch; the target may
be on either.  The optimiser's only option is to inflate Σ_ν globally
to cover both branches.  A single Gaussian is a poor fit for a bimodal
marginal: it is over-dispersed when the target is on the near branch,
and still under-covers the far branch.  The result is poor sharpness
everywhere to accommodate bifurcation zones.

Why Energy Score succeeds
--------------------------
With noise injected into rollouts, sampled trajectories naturally split
across branches.  The ES evaluates the *sample ensemble* directly:

    ES = E[||X - x*||] - ½ E[||X - X'||]

Samples on the wrong branch contribute to the first term (pulling some
toward the correct branch).  Samples on the correct branch contribute
beneficially to both terms.  The diversity term rewards maintaining
coverage of both branches when the data warrants it.  No density
evaluation, no unimodal assumption.

Polar dynamics
--------------
    α(θ)   = α₀ · sin(n_bif · θ)
    dr/dt  = (-μ + α(θ))(r - R) - κ(r - R)³       [pitchfork]
    dθ/dt  = ω₀                                     [constant rotation]

Bifurcation condition:  α(θ) > μ
Branch radii:           R ± √((α(θ) - μ) / κ)

Cartesian form (implemented):
    ρ      = r - R           (radial deviation)
    f_r    = (-μ + α(θ)) ρ - κ ρ³
    dx₁/dt = f_r · x₁/r  - ω₀ · x₂
    dx₂/dt = f_r · x₂/r  + ω₀ · x₁

Default parameters
------------------
    R      = 1.0      Nominal radius
    μ      = 2.0      Radial restoring strength
    κ      = 8.0      Cubic saturation (controls branch separation)
    α₀     = 5.0      Bifurcation amplitude  (α₀ > μ required)
    n_bif  = 2        Bifurcation zones per revolution
    ω₀     = 1.0      Angular velocity (slow enough to resolve branches)
    dt     = 0.05     Time step
    σ_ν    = 0.03     Process noise std

Key derived quantities (defaults):
    Max branch separation : 2√((α₀ - μ)/κ) ≈ 1.22
    Fraction bifurcated   : arcsin(μ/α₀)/(π) ≈ 0.74 of orbit
    Steps per orbit       : 2π / (ω₀ · dt) ≈ 126
    Steps in bif. zone    : ~93  (plenty of time for branch separation)
    Radial timescale      : 1/(α₀ - μ) ≈ 0.33s → ~7 steps to separate
"""
import sys
from pathlib import Path

# Add the project root to sys.path to resolve 'src' imports
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import numpy as np
from typing import Optional, Tuple
from scipy.stats import qmc
from matplotlib import pyplot as plt

from src.environments.base_environment import BaseEnvironment


class BifurcatingLimitCycleEnv(BaseEnvironment):
    """
    2D limit cycle with angle-dependent pitchfork bifurcations.

    Parameters
    ----------
    dt : float
        Euler integration time step.
    noise_std : float
        Isotropic Gaussian process noise σ_ν.
    R : float
        Nominal limit cycle radius.
    mu : float
        Base radial restoring strength.
    kappa : float
        Cubic saturation for the pitchfork. Controls branch separation:
        Δr_max = 2·√((α₀ - μ) / κ).  Larger κ → tighter branches.
    alpha_0 : float
        Bifurcation modulation amplitude.  Must satisfy α₀ > μ for
        bifurcation to occur.
    n_bif : int
        Number of bifurcation zone pairs per revolution.
    omega_0 : float
        Constant angular velocity [rad / time-unit].
    """

    DEFAULT_R       = 1.0
    DEFAULT_MU      = 2.0
    DEFAULT_KAPPA   = 8.0
    DEFAULT_ALPHA_0 = 5.0
    DEFAULT_N_BIF   = 2
    DEFAULT_OMEGA_0 = 1.0

    def __init__(
        self,
        dt:        float = 0.05,
        noise_std: float = 0.03,
        R:         float = DEFAULT_R,
        mu:        float = DEFAULT_MU,
        kappa:     float = DEFAULT_KAPPA,
        alpha_0:   float = DEFAULT_ALPHA_0,
        n_bif:     int   = DEFAULT_N_BIF,
        omega_0:   float = DEFAULT_OMEGA_0,
    ):
        super().__init__(dt=dt, noise_std=noise_std)
        self.R       = R
        self.mu      = mu
        self.kappa   = kappa
        self.alpha_0 = alpha_0
        self.n_bif   = n_bif
        self.omega_0 = omega_0

        if self.alpha_0 <= self.mu:
            raise ValueError(
                f"Need α₀ > μ for bifurcation; got α₀={alpha_0}, μ={mu}"
            )

    # ------------------------------------------------------------------
    # BaseEnvironment interface
    # ------------------------------------------------------------------

    @property
    def dim(self) -> int:
        return 2

    def dynamics(self, x: np.ndarray) -> np.ndarray:
        """
        Cartesian velocity dx/dt.

        Radial component:
            ρ  = r - R
            f_r = (-μ + α(θ)) ρ  -  κ ρ³

        Angular component:
            constant rotation at ω₀
        """
        r     = np.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-8
        theta = np.arctan2(x[1], x[0])

        # Angle-dependent bifurcation parameter
        alpha = self.alpha_0 * np.sin(self.n_bif * theta)

        # Radial pitchfork
        rho    = r - self.R
        f_r    = (-self.mu + alpha) * rho - self.kappa * rho ** 3

        # Project to Cartesian
        dx1 = f_r * x[0] / r - self.omega_0 * x[1]
        dx2 = f_r * x[1] / r + self.omega_0 * x[0]
        return np.array([dx1, dx2])

    def sample_initial_state(
        self,
        rng:          np.random.RandomState,
        scale:        float                  = 1.5,
        bounds:       Optional[np.ndarray]   = None,
        method:       str                    = "on_cycle",
        sobol_engine: Optional[qmc.Sobol]    = None,
    ) -> np.ndarray:
        """
        Sample an initial condition.

        method="on_cycle" places the particle exactly on the nominal
        radius R at a random angle — useful for studying bifurcation
        dynamics without transient radial attraction.
        """
        if bounds is not None:
            return np.array([
                rng.uniform(bounds[i, 0], bounds[i, 1]) for i in range(2)
            ])

        if method == "on_cycle":
            theta = rng.uniform(0, 2 * np.pi)
            return self.R * np.array([np.cos(theta), np.sin(theta)])

        if method == "near_cycle":
            theta = rng.uniform(0, 2 * np.pi)
            r = self.R + rng.normal(0, 0.05)
            return r * np.array([np.cos(theta), np.sin(theta)])

        if method == "sobol":
            if sobol_engine is None:
                sobol_engine = qmc.Sobol(d=2, scramble=True)
            u = sobol_engine.random(1)[0]
            return (u * 2.0 - 1.0) * scale

        return rng.uniform(-scale, scale, size=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_params(
        self,
        R:       Optional[float] = None,
        mu:      Optional[float] = None,
        kappa:   Optional[float] = None,
        alpha_0: Optional[float] = None,
        n_bif:   Optional[int]   = None,
        omega_0: Optional[float] = None,
    ) -> None:
        if R       is not None: self.R       = R
        if mu      is not None: self.mu      = mu
        if kappa   is not None: self.kappa   = kappa
        if alpha_0 is not None: self.alpha_0 = alpha_0
        if n_bif   is not None: self.n_bif   = n_bif
        if omega_0 is not None: self.omega_0 = omega_0

    def bifurcation_parameter(self, theta: np.ndarray) -> np.ndarray:
        """α(θ) = α₀ · sin(n_bif · θ)."""
        return self.alpha_0 * np.sin(self.n_bif * theta)

    def is_bifurcated(self, theta: np.ndarray) -> np.ndarray:
        """Boolean mask: True where α(θ) > μ (bifurcation active)."""
        return self.bifurcation_parameter(theta) > self.mu

    def branch_radii(self, theta: float) -> np.ndarray:
        """
        Stable equilibrium radii at angle θ.

        Returns shape (1,) outside bifurcation zones, (2,) inside.
        """
        alpha = self.alpha_0 * np.sin(self.n_bif * theta)
        if alpha <= self.mu:
            return np.array([self.R])
        delta = np.sqrt((alpha - self.mu) / self.kappa)
        return np.array([self.R - delta, self.R + delta])

    def max_branch_separation(self) -> float:
        """Peak-to-peak radial distance between branches."""
        excess = self.alpha_0 - self.mu
        if excess <= 0:
            return 0.0
        return 2.0 * np.sqrt(excess / self.kappa)

    def fraction_bifurcated(self) -> float:
        """Fraction of the orbit (by angle) in a bifurcation zone."""
        thetas = np.linspace(0, 2 * np.pi, 10_000, endpoint=False)
        return float(np.mean(self.is_bifurcated(thetas)))

    def steps_per_orbit(self) -> float:
        """Number of dt-steps for one full revolution."""
        return 2.0 * np.pi / (self.omega_0 * self.dt)

    def radial_timescale(self) -> float:
        """
        Characteristic time for radial separation at peak bifurcation.

        τ = 1 / (α₀ - μ).  The particle needs several τ to reach a branch.
        """
        return 1.0 / (self.alpha_0 - self.mu)

    def steps_to_separate(self) -> float:
        """Approximate dt-steps needed for branch separation."""
        return self.radial_timescale() / self.dt

    def summary(self) -> dict:
        """Diagnostic summary of environment parameters."""
        return {
            "max_branch_separation": self.max_branch_separation(),
            "fraction_bifurcated":   self.fraction_bifurcated(),
            "steps_per_orbit":       self.steps_per_orbit(),
            "radial_timescale":      self.radial_timescale(),
            "steps_to_separate":     self.steps_to_separate(),
            "noise_std":             self.noise_std,
            "params": {
                "R": self.R, "mu": self.mu, "kappa": self.kappa,
                "alpha_0": self.alpha_0, "n_bif": self.n_bif,
                "omega_0": self.omega_0, "dt": self.dt,
            },
        }
        
    def visualize(self, steps: int = 400, num_trajectories: int = 50):
        """
        Runs multiple rollouts to visualize the bifurcation branches.
        """
        rng = np.random.RandomState(42)
        # Start at a point just before a bifurcation zone
        # (For n_bif=2, first zone is around theta=pi/4)
        x0 = np.array([self.R, 0.0]) 
        
        all_trajectories = []
        for _ in range(num_trajectories):
            x = x0.copy()
            traj = [x.copy()]
            for _ in range(steps):
                # Step dynamics with noise
                v = self.dynamics(x)
                x = x + v * self.dt + self.noise_std * np.sqrt(self.dt) * rng.randn(2)
                traj.append(x.copy())
            all_trajectories.append(np.array(traj))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: 2D Trajectories
        theta_grid = np.linspace(0, 2 * np.pi, 500)
        # Plot nominal cycle
        ax1.plot(self.R * np.cos(theta_grid), self.R * np.sin(theta_grid), 
                 'k--', alpha=0.3, label="Nominal Cycle (R)")
        
        for i, traj in enumerate(all_trajectories):
            ax1.plot(traj[:, 0], traj[:, 1], color='tab:blue', alpha=0.2, 
                     linewidth=0.8, label="Noisy Rollouts" if i == 0 else "")

        ax1.set_title(f"Bifurcating Limit Cycle ($n={self.n_bif}$)")
        ax1.set_xlabel("$x_1$")
        ax1.set_ylabel("$x_2$")
        ax1.legend()
        ax1.axis('equal')

        # Plot 2: Radius over Time (Showing Bimodality)
        times = np.arange(steps + 1) * self.dt
        for traj in all_trajectories:
            radii = np.linalg.norm(traj, axis=1)
            ax2.plot(times, radii, color='tab:red', alpha=0.1)
        
        ax2.axhline(self.R, color='k', linestyle='--', label="Target Radius $R$")
        ax2.set_title("Radius Over Time (Evidence of Bimodality)")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Radius $r$")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig("bifurcating_limit_cycle_visualization.png", dpi=300)
        plt.show()
        
    def visualize_vector_field(self, grid_size: int = 25):
        """
        Plots the vector field and streamlines to show the bifurcation zones.
        """
        import matplotlib.pyplot as plt
        
        limit = self.R + 1.0
        x = np.linspace(-limit, limit, grid_size)
        y = np.linspace(-limit, limit, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate velocity at each grid point
        U, V = np.zeros(X.shape), np.zeros(Y.shape)
        for i in range(grid_size):
            for j in range(grid_size):
                dx = self.dynamics(np.array([X[i,j], Y[i,j]]))
                U[i,j], V[i,j] = dx[0], dx[1]

        # ---------------------------------------------------------
        # THE FIX: Calculate speed and normalize the quiver vectors
        # ---------------------------------------------------------
        speed = np.sqrt(U**2 + V**2)
        
        # Normalize to prevent the cubic term from creating massive outer arrows
        # Add a small epsilon (1e-8) to prevent division by zero
        U_norm = U / (speed + 1e-8)
        V_norm = V / (speed + 1e-8)

        plt.figure(figsize=(10, 8))
        
        # 1. Streamlines (The 'Flow') - Now colored by speed to retain magnitude info
        strm = plt.streamplot(X, Y, U, V, color=speed, cmap='viridis', density=1.2)
        plt.colorbar(strm.lines, label='Velocity Magnitude (Speed)')
        
        # 2. Bifurcation Highlight
        # Highlight where alpha(theta) > mu
        theta_grid = np.linspace(0, 2*np.pi, 500)
        alpha = self.bifurcation_parameter(theta_grid)
        bif_mask = alpha > self.mu
        
        # Plot nominal circle, colored by bifurcation status
        for i in range(len(theta_grid)-1):
            color = 'red' if bif_mask[i] else 'black'
            plt.plot(self.R * np.cos(theta_grid[i:i+2]), 
                     self.R * np.sin(theta_grid[i:i+2]), 
                     color=color, lw=3, alpha=0.7)

        # 3. Quiver (Arrows) - Using normalized vectors and explicit grey color
        plt.quiver(X, Y, U_norm, V_norm, color='grey', alpha=0.5, pivot='mid', scale=grid_size * 1.5)

        plt.title(f"Vector Field: Red = Bifurcated Zone (n={self.n_bif})")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.axis('equal')
        plt.grid(alpha=0.2)
        plt.show()

    def visualize_radius_histogram(
        self,
        steps: int = 200,
        num_trajectories: int = 500,
        time_index: Optional[int] = None,
        bins: int = 60,
    ):
        """
        Plots histogram of radii at a fixed time step to reveal bimodality.

        Parameters
        ----------
        steps : int
            Total rollout length.
        num_trajectories : int
            Number of stochastic rollouts.
        time_index : int or None
            Which time index to evaluate.
            If None, uses 70% of rollout (typically inside bifurcation zone).
        bins : int
            Histogram bin count.
        """
        import matplotlib.pyplot as plt

        rng = np.random.RandomState(0)

        if time_index is None:
            time_index = int(0.7 * steps)

        x0 = np.array([self.R, 0.0])
        radii = []

        for _ in range(num_trajectories):
            x = x0.copy()
            for t in range(steps):
                v = self.dynamics(x)
                x = x + v * self.dt + self.noise_std * np.sqrt(self.dt) * rng.randn(2)
                if t == time_index:
                    radii.append(np.linalg.norm(x))

        radii = np.array(radii)

        # Compute angle at that time to overlay theoretical branches
        theta = self.omega_0 * self.dt * time_index
        branch_r = self.branch_radii(theta)

        plt.figure(figsize=(8, 5))
        plt.hist(radii, bins=bins, density=True, alpha=0.7)

        for r_eq in branch_r:
            plt.axvline(r_eq, linestyle="--", linewidth=2)

        plt.title("Radius Histogram at Fixed Time")
        plt.xlabel("Radius r")
        plt.ylabel("Density")
        plt.grid(alpha=0.3)
        plt.show()
        
if __name__ == "__main__":
    # Initialize the environment with default parameters
    env = BifurcatingLimitCycleEnv(
        dt=0.05,
        noise_std=0.02,
        mu=1.5,             # Lower restoring baseline
        kappa=5,          # Weak cubic → huge separation
        alpha_0=10.0,       # Strong bifurcation
        n_bif=2,            # One very wide zone
        omega_0=1         # Slow rotation → long dwell time
    )

    print(f"Visualizing {env.__class__.__name__}...")
    print("- Dashed line represents the nominal radius.")
    print("- Watch how trajectories split into two distinct 'arms' in bifurcation zones.")
    
    # Run the visualization
    env.visualize_vector_field(grid_size=30)
    env.visualize_radius_histogram(
        steps=250,
        num_trajectories=800
    )    
    env.visualize(steps=400, num_trajectories=60)