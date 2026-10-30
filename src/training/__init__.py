"""
Training utilities for VBLL models.

This package provides:
- Training functions for single-step and multi-step models
- Post-training evaluation
- Evaluation metrics for trajectories
"""

from src.training.trainer import (
    train_single_step_model,
    train_multi_step_model,
    train_multi_step_model_with_actions,
)

from src.training.post_train_eval import run_post_train_eval

from src.training.metrics import (
    compute_dtw_distance,
    compute_l2_errors,
    compute_final_displacement_error,
    compute_average_displacement_error,
    compute_trajectory_metrics,
    compute_batch_metrics,
    summarize_metrics,
    print_metrics_summary,
    compute_mmd,
    compute_energy_distance
)

__all__ = [
    # Post-training evaluation
    "run_post_train_eval",
    # Training functions
    "train_single_step_model",
    "train_multi_step_model",
    "train_multi_step_model_with_actions",
    # Metrics
    "compute_dtw_distance",
    "compute_l2_errors",
    "compute_final_displacement_error",
    "compute_average_displacement_error",
    "compute_trajectory_metrics",
    "compute_batch_metrics",
    "summarize_metrics",
    "print_metrics_summary",
    "compute_mmd",
    "compute_energy_distance"
]
