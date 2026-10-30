"""Local CSV metric logger used by cluster training jobs.

The files use a tidy (long-form) layout so runs with different state dimensions
and rollout horizons can be concatenated without reconciling CSV headers.

Files written per run:
  training_metrics.csv       per-epoch metrics (epoch, metric, value): losses,
                             gradient norms, clip fraction, posterior and
                             process-noise statistics, parameter drift
  validation_by_horizon.csv  validation RMSE, spread and spread-skill per step
  run_summary.csv            training time, final/best losses, rollout metrics
  run_config.yaml            fully resolved config
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

import numpy as np
from omegaconf import OmegaConf


class CsvLogger:
    """Collect scalar metrics and write them to CSV files in a run directory."""

    def __init__(
        self,
        output_dir: str | Path,
        metrics_filename: str = "training_metrics.csv",
        horizon_filename: str = "validation_by_horizon.csv",
        summary_filename: str = "run_summary.csv",
        config_filename: str = "run_config.yaml",
        run_config: Optional[Any] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / metrics_filename
        self.horizon_path = self.output_dir / horizon_filename
        self.summary_path = self.output_dir / summary_filename
        self.config_path = self.output_dir / config_filename

        self._step_cache: dict[int, dict[str, float]] = {}
        self._horizon_cache: dict[int, list[dict[str, float | int]]] = {}
        self._summary: dict[str, float] = {}

        self._initialize_csv(self.metrics_path, ["epoch", "metric", "value"])
        self._initialize_csv(
            self.horizon_path,
            ["epoch", "horizon", "rmse", "spread", "spread_skill_ratio"],
        )
        self._initialize_csv(self.summary_path, ["metric", "value"])

        if run_config is not None:
            cfg = (
                run_config
                if OmegaConf.is_config(run_config)
                else OmegaConf.create(run_config)
            )
            OmegaConf.save(cfg, self.config_path, resolve=True)

    @classmethod
    def from_config(
        cls,
        logger_cfg: Any,
        output_dir: str | Path,
        run_config: Optional[Any] = None,
    ) -> "CsvLogger":
        raw = OmegaConf.to_container(logger_cfg, resolve=True)
        return cls(
            output_dir=output_dir,
            metrics_filename=raw.get("metrics_filename", "training_metrics.csv"),
            horizon_filename=raw.get(
                "horizon_filename", "validation_by_horizon.csv"
            ),
            summary_filename=raw.get("summary_filename", "run_summary.csv"),
            config_filename=raw.get("config_filename", "run_config.yaml"),
            run_config=run_config,
        )

    @staticmethod
    def _initialize_csv(path: Path, fieldnames: list[str]) -> None:
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    @staticmethod
    def _append_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        if not rows:
            return
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerows(rows)

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self._step_cache.setdefault(int(global_step), {})[tag] = float(scalar_value)

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        """Store compact histogram summaries rather than large binary payloads."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        self.add_scalar(f"{tag}/count", float(arr.size), global_step)
        self.add_scalar(f"{tag}/nonfinite_count", float(arr.size - finite.size), global_step)
        if finite.size:
            self.add_scalar(f"{tag}/mean", float(np.mean(finite)), global_step)
            self.add_scalar(f"{tag}/std", float(np.std(finite)), global_step)
            self.add_scalar(f"{tag}/min", float(np.min(finite)), global_step)
            self.add_scalar(f"{tag}/max", float(np.max(finite)), global_step)

    def add_horizon_diagnostics(
        self,
        epoch: int,
        rmse: Any,
        spread: Any,
        spread_skill_ratio: Any,
    ) -> None:
        rmse_arr = np.asarray(rmse, dtype=float).reshape(-1)
        spread_arr = np.asarray(spread, dtype=float).reshape(-1)
        ssr_arr = np.asarray(spread_skill_ratio, dtype=float).reshape(-1)
        if not (len(rmse_arr) == len(spread_arr) == len(ssr_arr)):
            raise ValueError("Per-horizon diagnostic arrays must have equal lengths")
        self._horizon_cache[int(epoch)] = [
            {
                "epoch": int(epoch),
                "horizon": int(horizon),
                "rmse": float(rmse_arr[horizon]),
                "spread": float(spread_arr[horizon]),
                "spread_skill_ratio": float(ssr_arr[horizon]),
            }
            for horizon in range(len(rmse_arr))
        ]

    def flush(self, step: Optional[int] = None) -> None:
        steps = [int(step)] if step is not None else sorted(
            set(self._step_cache) | set(self._horizon_cache)
        )
        for current_step in steps:
            scalars = self._step_cache.pop(current_step, {})
            self._append_rows(
                self.metrics_path,
                ["epoch", "metric", "value"],
                [
                    {
                        "epoch": current_step,
                        "metric": metric,
                        "value": value,
                    }
                    for metric, value in sorted(scalars.items())
                ],
            )
            horizon_rows = self._horizon_cache.pop(current_step, [])
            self._append_rows(
                self.horizon_path,
                ["epoch", "horizon", "rmse", "spread", "spread_skill_ratio"],
                horizon_rows,
            )

    def log_summary(self, data: dict[str, float]) -> None:
        self._summary.update({key: float(value) for key, value in data.items()})
        with self.summary_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(
                {"metric": metric, "value": value}
                for metric, value in sorted(self._summary.items())
            )

    def close(self) -> None:
        self.flush()
