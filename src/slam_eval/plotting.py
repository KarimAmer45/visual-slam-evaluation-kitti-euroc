"""Plotting helpers for trajectory evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from slam_eval.metrics import translation_errors
from slam_eval.trajectory import PoseTrajectory3D


def plot_trajectories(
    estimate: PoseTrajectory3D,
    reference: PoseTrajectory3D,
    output: str | Path,
    title: str = "Estimated vs Ground Truth Trajectory",
    axes: tuple[int, int] = (0, 2),
) -> Path:
    """Save a top-down trajectory comparison plot."""

    output_path = _prepare_output(output)
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=140)
    ax.plot(reference.xyz[:, axes[0]], reference.xyz[:, axes[1]], label="ground truth", linewidth=2.4)
    ax.plot(estimate.xyz[:, axes[0]], estimate.xyz[:, axes[1]], label="estimated", linewidth=2.0)
    ax.scatter(reference.xyz[0, axes[0]], reference.xyz[0, axes[1]], s=35, label="start")
    ax.set_title(title)
    ax.set_xlabel(_axis_label(axes[0]))
    ax.set_ylabel(_axis_label(axes[1]))
    ax.axis("equal")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_translation_errors(
    estimate: PoseTrajectory3D,
    reference: PoseTrajectory3D,
    output: str | Path,
    title: str = "Absolute Trajectory Error",
) -> Path:
    """Save a per-frame translation error plot."""

    output_path = _prepare_output(output)
    errors = translation_errors(estimate, reference)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=140)
    ax.plot(reference.timestamps, errors, color="#c23b22", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("timestamp")
    ax.set_ylabel("translation error")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _prepare_output(output: str | Path) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _axis_label(axis: int) -> str:
    return ["x", "y", "z"][axis]

