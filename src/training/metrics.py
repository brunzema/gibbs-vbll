"""
Metrics for evaluating trajectory predictions.

This module provides various metrics for comparing predicted trajectories
against ground truth, including DTW distance, L2 errors, and more.
"""

import numpy as np
from typing import List, Dict
from dtaidistance import dtw


def compute_mmd(samples_p: np.ndarray, samples_q: np.ndarray, bandwidth: float = None) -> float:
    """
    Compute MMD^2 between two sets of trajectory samples using RBF kernel.
    
    Each sample is a flattened trajectory: (n_samples, T * state_dim).
    Uses the median heuristic for bandwidth if not provided.
    
    MMD^2(P, Q) = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    """
    if bandwidth is None:
        # Median heuristic: bandwidth = median of pairwise distances
        all_samples = np.concatenate([samples_p, samples_q], axis=0)
        dists = np.linalg.norm(all_samples[:, None] - all_samples[None, :], axis=-1)
        bandwidth = np.median(dists[dists > 0])
        if bandwidth == 0:
            bandwidth = 1.0

    def rbf_kernel(x, y, bw):
        sq_dists = np.sum((x[:, None] - y[None, :]) ** 2, axis=-1)
        return np.exp(-sq_dists / (2 * bw ** 2))

    k_pp = rbf_kernel(samples_p, samples_p, bandwidth)
    k_qq = rbf_kernel(samples_q, samples_q, bandwidth)
    k_pq = rbf_kernel(samples_p, samples_q, bandwidth)

    n, m = len(samples_p), len(samples_q)

    # Unbiased estimator (exclude diagonal for k_pp and k_qq)
    mmd2 = ((k_pp.sum() - np.trace(k_pp)) / (n * (n - 1))
            + (k_qq.sum() - np.trace(k_qq)) / (m * (m - 1))
            - 2 * k_pq.mean())
    return float(mmd2)


def compute_energy_distance(samples_p: np.ndarray, samples_q: np.ndarray, beta: float = 1.0) -> float:
    """
    Compute Energy Distance between two sets of trajectory samples.
    
    D_E(P, Q) = 2 E||X - Y||^beta - E||X - X'||^beta - E||Y - Y'||^beta
    
    Strictly proper for beta in (0, 2). Default beta=1.
    """
    def mean_pairwise_dist(a, b, beta):
        diffs = np.linalg.norm(a[:, None] - b[None, :], axis=-1)
        return np.mean(diffs ** beta)

    cross = mean_pairwise_dist(samples_p, samples_q, beta)
    intra_p = mean_pairwise_dist(samples_p, samples_p, beta)
    intra_q = mean_pairwise_dist(samples_q, samples_q, beta)

    return float(2 * cross - intra_p - intra_q)


def compute_dtw_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance between two trajectories.
    
    DTW is a measure of similarity between two temporal sequences which may
    vary in speed. It's particularly useful for comparing trajectories that
    may have similar shapes but different time scales.
    
    Args:
        traj1: First trajectory of shape (T, D) where T is timesteps and D is dimension
        traj2: Second trajectory of shape (T, D)
        
    Returns:
        DTW distance as a scalar float
    """
    # DTW requires 1D sequences, so we compute DTW for each dimension and average
    if traj1.ndim == 1:
        return dtw.distance(traj1, traj2)
    
    # For multi-dimensional trajectories, compute DTW per dimension and sum
    total_distance = 0.0
    for dim in range(traj1.shape[1]):
        total_distance += dtw.distance(traj1[:, dim], traj2[:, dim])
    
    return total_distance


def compute_l2_errors(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """
    Compute L2 (Euclidean) errors at each timestep.
    
    Args:
        predicted: Predicted trajectory of shape (T, D)
        ground_truth: Ground truth trajectory of shape (T, D)
        
    Returns:
        Array of L2 errors at each timestep, shape (T,)
    """
    return np.linalg.norm(predicted - ground_truth, axis=-1)


def compute_final_displacement_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute the L2 error at the final timestep.
    
    This is also known as Final Displacement Error (FDE) in trajectory
    prediction literature.
    
    Args:
        predicted: Predicted trajectory of shape (T, D)
        ground_truth: Ground truth trajectory of shape (T, D)
        
    Returns:
        L2 error at final timestep
    """
    return float(np.linalg.norm(predicted[-1] - ground_truth[-1]))


def compute_average_displacement_error(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> float:
    """
    Compute the average L2 error across all timesteps.
    
    This is also known as Average Displacement Error (ADE) in trajectory
    prediction literature.
    
    Args:
        predicted: Predicted trajectory of shape (T, D)
        ground_truth: Ground truth trajectory of shape (T, D)
        
    Returns:
        Average L2 error across timesteps
    """
    errors = compute_l2_errors(predicted, ground_truth)
    return float(np.mean(errors))


def compute_trajectory_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of metrics for a single trajectory.
    
    Args:
        predicted: Predicted trajectory of shape (T, D)
        ground_truth: Ground truth trajectory of shape (T, D)
        
    Returns:
        Dictionary containing:
        - 'dtw': DTW distance
        - 'ade': Average Displacement Error
        - 'fde': Final Displacement Error
        - 'max_error': Maximum L2 error across timesteps
    """
    return {
        'dtw': compute_dtw_distance(predicted, ground_truth),
        'ade': compute_average_displacement_error(predicted, ground_truth),
        'fde': compute_final_displacement_error(predicted, ground_truth),
        'max_error': float(np.max(compute_l2_errors(predicted, ground_truth))),
    }


def compute_batch_metrics(
    predicted_list: List[np.ndarray],
    ground_truth_list: List[np.ndarray],
) -> Dict[str, List[float]]:
    """
    Compute metrics for a batch of trajectories.
    
    Args:
        predicted_list: List of predicted trajectories, each of shape (T, D)
        ground_truth_list: List of ground truth trajectories, each of shape (T, D)
        
    Returns:
        Dictionary with keys 'dtw', 'ade', 'fde', 'max_error',
        each mapping to a list of values (one per trajectory)
    """
    metrics = {
        'dtw': [],
        'ade': [],
        'fde': [],
        'max_error': [],
    }
    
    for pred, gt in zip(predicted_list, ground_truth_list):
        traj_metrics = compute_trajectory_metrics(pred, gt)
        for key in metrics:
            metrics[key].append(traj_metrics[key])
    
    return metrics


def summarize_metrics(
    metrics: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics (mean, std, min, max) for each metric.
    
    Args:
        metrics: Dictionary from compute_batch_metrics
        
    Returns:
        Dictionary mapping metric names to summary statistics
    """
    summary = {}
    
    for key, values in metrics.items():
        values_arr = np.array(values)
        summary[key] = {
            'mean': float(np.mean(values_arr)),
            'std': float(np.std(values_arr)),
            'min': float(np.min(values_arr)),
            'max': float(np.max(values_arr)),
        }
    
    return summary


def print_metrics_summary(
    metrics: Dict[str, List[float]],
    prefix: str = "",
) -> None:
    """
    Print a formatted summary of trajectory metrics.
    
    Args:
        metrics: Dictionary from compute_batch_metrics
        prefix: Optional prefix for the print output
    """
    summary = summarize_metrics(metrics)
    
    if prefix:
        print(f"\n{prefix}")
    
    for metric_name, stats in summary.items():
        print(f"  {metric_name.upper()}: "
              f"mean={stats['mean']:.4f}, "
              f"std={stats['std']:.4f}, "
              f"min={stats['min']:.4f}, "
              f"max={stats['max']:.4f}")
