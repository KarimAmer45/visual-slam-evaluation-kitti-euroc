"""Trajectory representations and alignment routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseTrajectory3D:
    """Timestamped 4x4 pose matrices in a common world frame."""

    timestamps: np.ndarray
    poses: np.ndarray

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps, dtype=float)
        poses = np.asarray(self.poses, dtype=float)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must have shape (N, 4, 4), got {poses.shape}")
        if len(timestamps) != len(poses):
            raise ValueError("timestamps and poses must have the same length")
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "poses", poses)

    @property
    def xyz(self) -> np.ndarray:
        return self.poses[:, :3, 3]

    def transformed(self, rotation: np.ndarray, translation: np.ndarray, scale: float = 1.0) -> "PoseTrajectory3D":
        poses = self.poses.copy()
        poses[:, :3, 3] = (scale * (rotation @ self.xyz.T)).T + translation
        poses[:, :3, :3] = rotation @ poses[:, :3, :3]
        return PoseTrajectory3D(self.timestamps.copy(), poses)

    def take(self, indices: np.ndarray) -> "PoseTrajectory3D":
        return PoseTrajectory3D(self.timestamps[indices], self.poses[indices])


@dataclass(frozen=True)
class AlignmentResult:
    aligned: PoseTrajectory3D
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    matched_reference: PoseTrajectory3D


def align_umeyama(
    estimate: PoseTrajectory3D,
    reference: PoseTrajectory3D,
    with_scale: bool = False,
    max_time_delta: float | None = None,
) -> AlignmentResult:
    """Align estimate positions to reference positions using Umeyama similarity.

    Set ``with_scale=True`` for monocular visual odometry where scale is arbitrary.
    """

    matched_estimate, matched_reference = associate_by_time(estimate, reference, max_time_delta)
    source = matched_estimate.xyz
    target = matched_reference.xyz
    if len(source) < 3:
        raise ValueError("At least 3 matched poses are required for trajectory alignment")

    mu_source = source.mean(axis=0)
    mu_target = target.mean(axis=0)
    source_centered = source - mu_source
    target_centered = target - mu_target
    covariance = (target_centered.T @ source_centered) / len(source)

    u_matrix, singular_values, v_transposed = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(u_matrix) * np.linalg.det(v_transposed) < 0:
        sign[-1] = -1
    correction = np.diag(sign)
    rotation = u_matrix @ correction @ v_transposed

    if with_scale:
        variance = np.mean(np.sum(source_centered**2, axis=1))
        scale = float(np.sum(singular_values * sign) / variance) if variance > 0 else 1.0
    else:
        scale = 1.0

    translation = mu_target - scale * (rotation @ mu_source)
    aligned = matched_estimate.transformed(rotation, translation, scale)
    return AlignmentResult(aligned, rotation, translation, scale, matched_reference)


def associate_by_time(
    estimate: PoseTrajectory3D,
    reference: PoseTrajectory3D,
    max_time_delta: float | None = None,
) -> tuple[PoseTrajectory3D, PoseTrajectory3D]:
    """Associate trajectories by nearest timestamp."""

    if len(estimate.timestamps) == len(reference.timestamps) and np.allclose(
        estimate.timestamps, reference.timestamps
    ):
        indices = np.arange(len(estimate.timestamps))
        return estimate.take(indices), reference.take(indices)

    ref_times = reference.timestamps
    estimate_indices: list[int] = []
    reference_indices: list[int] = []
    for est_index, timestamp in enumerate(estimate.timestamps):
        ref_index = int(np.argmin(np.abs(ref_times - timestamp)))
        delta = abs(ref_times[ref_index] - timestamp)
        if max_time_delta is None or delta <= max_time_delta:
            estimate_indices.append(est_index)
            reference_indices.append(ref_index)

    if not estimate_indices:
        raise ValueError("No trajectory samples could be associated by timestamp")
    return estimate.take(np.asarray(estimate_indices)), reference.take(np.asarray(reference_indices))


def load_tum_trajectory(path: str) -> PoseTrajectory3D:
    """Load a TUM RGB-D trajectory file: timestamp tx ty tz qx qy qz qw."""

    from slam_eval.datasets import quaternion_to_matrix

    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(item) for item in line.split()]
            if len(values) != 8:
                raise ValueError(f"{path}:{line_number} expected 8 floats, got {len(values)}")
            timestamp, tx, ty, tz, qx, qy, qz, qw = values
            pose = np.eye(4, dtype=float)
            pose[:3, :3] = quaternion_to_matrix(qw, qx, qy, qz)
            pose[:3, 3] = (tx, ty, tz)
            timestamps.append(timestamp)
            poses.append(pose)
    return PoseTrajectory3D(np.asarray(timestamps, dtype=float), np.stack(poses, axis=0))


def save_tum_trajectory(path: str, trajectory: PoseTrajectory3D) -> None:
    """Save positions with identity quaternions in TUM format.

    This is intended for quick baseline outputs and demos. Rotation export can be
    extended if downstream tools need full orientation fidelity.
    """

    from pathlib import Path

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for timestamp, xyz in zip(trajectory.timestamps, trajectory.xyz):
            handle.write(f"{timestamp:.9f} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f} 0 0 0 1\n")
