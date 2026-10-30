"""
Hydra-compatible wrapper for continuous-time systems from the vendored `dysts`.

This environment adapts `dysts.flows.*` systems to the local BaseEnvironment
interface used by the training pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import qmc

from src.environments.base_environment import BaseEnvironment


def _import_dysts_flows():
    """
    Import dysts flows, preferring a normal install and falling back to the
    vendored submodule path at src/environments/dysts.
    """
    try:
        from dysts import flows as dflows  # type: ignore
        from dysts.base import DynSys  # type: ignore
    except ModuleNotFoundError:
        vendor_root = Path(__file__).resolve().parent / "dysts"
        vendor_root_str = str(vendor_root)
        if vendor_root_str not in sys.path:
            sys.path.insert(0, vendor_root_str)

        from dysts import flows as dflows  # type: ignore
        from dysts.base import DynSys  # type: ignore

    return dflows, DynSys


class DystsFlowEnvironment(BaseEnvironment):
    """
    Wrapper around one class in `dysts.flows`.

    Notes:
    - Supports continuous, non-delay systems inheriting from `dysts.base.DynSys`.
    - Uses this repository's BaseEnvironment stepping (RK4 + optional noise).
    """

    def __init__(
        self,
        system_name: str,
        dt: Optional[float] = None,
        noise_std: float = 0.01,
        parameters: Optional[dict] = None,
        initial_conditions: Optional[list[float]] = None,
    ):
        dflows, DynSys = _import_dysts_flows()

        if not hasattr(dflows, system_name):
            raise ValueError(f"Unknown dysts flow system '{system_name}'")

        system_cls = getattr(dflows, system_name)
        if not issubclass(system_cls, DynSys):
            raise ValueError(
                f"System '{system_name}' is not a supported continuous DynSys flow"
            )

        kwargs = {}
        if parameters is not None:
            kwargs["parameters"] = parameters
        if initial_conditions is not None:
            kwargs["initial_conditions"] = initial_conditions

        self.system_name = system_name
        self._system = system_cls(**kwargs)
        self._dim = int(self._system.dimension)

        effective_dt = float(dt) if dt is not None else float(getattr(self._system, "dt", 0.01))
        super().__init__(dt=effective_dt, noise_std=noise_std)

    @property
    def dim(self) -> int:
        return self._dim

    def dynamics(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != (self._dim,):
            raise ValueError(f"Expected x shape ({self._dim},), got {x.shape}")

        rhs = self._system.rhs(x, 0.0)
        return np.asarray(rhs, dtype=float)

    def sample_initial_state(
        self,
        rng: np.random.RandomState,
        scale: float = 1.0,
        bounds: Optional[np.ndarray] = None,
        method: str = "sobol",
        sobol_engine: Optional[qmc.Sobol] = None,
    ) -> np.ndarray:
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
            if bounds.shape != (self._dim, 2):
                raise ValueError(
                    f"bounds must have shape ({self._dim}, 2), got {bounds.shape}"
                )
            return rng.uniform(bounds[:, 0], bounds[:, 1])

        if method == "sobol":
            if sobol_engine is None:
                sobol_engine = qmc.Sobol(d=self._dim, scramble=True)
            u = sobol_engine.random(1)[0]
            return (u * 2.0 - 1.0) * scale

        if method == "gaussian":
            return rng.randn(self._dim) * scale

        return rng.uniform(-scale, scale, size=self._dim)
