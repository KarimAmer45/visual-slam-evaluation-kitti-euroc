"""A small ORB feature-matching visual odometry baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slam_eval.trajectory import PoseTrajectory3D


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.asarray([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=float)


@dataclass(frozen=True)
class OrbVoConfig:
    max_features: int = 3000
    ratio_test: float = 0.75
    min_matches: int = 40
    ransac_threshold: float = 1.0


def estimate_orb_visual_odometry(
    image_paths: list[Path],
    timestamps: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: OrbVoConfig | None = None,
) -> PoseTrajectory3D:
    """Estimate a monocular VO trajectory with ORB matches and essential matrices.

    The baseline is intentionally simple: it provides a reproducible comparison
    point for evaluation tooling, not a production SLAM system.
    """

    cv2 = _import_cv2()
    config = config or OrbVoConfig()
    orb = cv2.ORB_create(nfeatures=config.max_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    poses = [np.eye(4, dtype=float)]
    previous = _read_gray(cv2, image_paths[0])
    prev_keypoints, prev_descriptors = orb.detectAndCompute(previous, None)

    camera_matrix = intrinsics.matrix
    world_from_camera = np.eye(4, dtype=float)
    for image_path in image_paths[1:]:
        current = _read_gray(cv2, image_path)
        keypoints, descriptors = orb.detectAndCompute(current, None)
        motion = np.eye(4, dtype=float)

        if prev_descriptors is not None and descriptors is not None:
            matches = matcher.knnMatch(prev_descriptors, descriptors, k=2)
            good_matches = [
                pair[0]
                for pair in matches
                if len(pair) == 2 and pair[0].distance < config.ratio_test * pair[1].distance
            ]
            if len(good_matches) >= config.min_matches:
                prev_points = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches])
                curr_points = np.float32([keypoints[m.trainIdx].pt for m in good_matches])
                essential, mask = cv2.findEssentialMat(
                    curr_points,
                    prev_points,
                    camera_matrix,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=config.ransac_threshold,
                )
                if essential is not None and mask is not None:
                    _, rotation, translation, _ = cv2.recoverPose(
                        essential, curr_points, prev_points, camera_matrix
                    )
                    motion[:3, :3] = rotation
                    motion[:3, 3] = translation.reshape(3)

        world_from_camera = world_from_camera @ motion
        poses.append(world_from_camera.copy())
        previous = current
        prev_keypoints = keypoints
        prev_descriptors = descriptors

    return PoseTrajectory3D(np.asarray(timestamps[: len(poses)], dtype=float), np.stack(poses, axis=0))


def _read_gray(cv2: object, path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _import_cv2() -> object:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for ORB visual odometry. Install opencv-python-headless or use Docker."
        ) from exc
    return cv2
