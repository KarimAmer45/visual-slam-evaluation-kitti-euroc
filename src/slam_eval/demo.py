"""Synthetic demo data used for smoke tests and README screenshots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slam_eval.metrics import summarize_ate
from slam_eval.plotting import plot_trajectories, plot_translation_errors
from slam_eval.trajectory import PoseTrajectory3D, align_umeyama


def make_demo_trajectories(samples: int = 160) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """Create a repeatable curved reference path and a drifting estimate."""

    rng = np.random.default_rng(7)
    timestamps = np.linspace(0.0, 16.0, samples)
    theta = np.linspace(0.0, 2.4 * np.pi, samples)
    radius = 18.0 + 2.0 * np.sin(theta * 0.7)
    reference_xyz = np.column_stack(
        [
            radius * np.cos(theta),
            0.3 * np.sin(theta * 1.3),
            radius * np.sin(theta) + np.linspace(0.0, 8.0, samples),
        ]
    )
    drift = np.column_stack(
        [
            np.linspace(0.0, 2.2, samples),
            np.linspace(0.0, -0.4, samples),
            np.linspace(0.0, 1.4, samples),
        ]
    )
    estimate_xyz = reference_xyz * 0.92 + drift + rng.normal(0.0, 0.18, size=reference_xyz.shape)
    return _trajectory_from_xyz(timestamps, estimate_xyz), _trajectory_from_xyz(timestamps, reference_xyz)


def generate_demo_outputs(output_dir: str | Path) -> dict[str, object]:
    """Generate aligned demo plots and metric summary."""

    output_path = Path(output_dir)
    estimate, reference = make_demo_trajectories()
    alignment = align_umeyama(estimate, reference, with_scale=True)
    metrics = summarize_ate(alignment.aligned, alignment.matched_reference)
    trajectory_plot = plot_trajectories(
        alignment.aligned,
        alignment.matched_reference,
        output_path / "demo_trajectory.png",
        title="Demo Trajectory Alignment",
    )
    error_plot = plot_translation_errors(
        alignment.aligned,
        alignment.matched_reference,
        output_path / "demo_ate.png",
        title=f"Demo ATE RMSE: {metrics['rmse']:.3f}",
    )
    return {
        "metrics": metrics,
        "scale": alignment.scale,
        "trajectory_plot": str(trajectory_plot),
        "error_plot": str(error_plot),
    }


def _trajectory_from_xyz(timestamps: np.ndarray, xyz: np.ndarray) -> PoseTrajectory3D:
    poses = np.repeat(np.eye(4, dtype=float)[None, :, :], repeats=len(timestamps), axis=0)
    poses[:, :3, 3] = xyz
    return PoseTrajectory3D(timestamps, poses)

