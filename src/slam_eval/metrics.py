"""Evaluation metrics for aligned trajectories."""

from __future__ import annotations

import numpy as np

from slam_eval.trajectory import PoseTrajectory3D


def translation_errors(estimate: PoseTrajectory3D, reference: PoseTrajectory3D) -> np.ndarray:
    """Per-pose Euclidean translation error."""

    if len(estimate.poses) != len(reference.poses):
        raise ValueError("Trajectory lengths differ; associate or align before computing metrics")
    return np.linalg.norm(estimate.xyz - reference.xyz, axis=1)


def ate_rmse(estimate: PoseTrajectory3D, reference: PoseTrajectory3D) -> float:
    """Absolute trajectory error RMSE in trajectory distance units."""

    errors = translation_errors(estimate, reference)
    return float(np.sqrt(np.mean(errors**2)))


def summarize_ate(estimate: PoseTrajectory3D, reference: PoseTrajectory3D) -> dict[str, float]:
    """Return common ATE summary statistics."""

    errors = translation_errors(estimate, reference)
    return {
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "std": float(np.std(errors)),
        "min": float(np.min(errors)),
        "max": float(np.max(errors)),
        "count": float(len(errors)),
    }

