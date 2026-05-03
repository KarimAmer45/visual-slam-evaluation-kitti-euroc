"""Dataset discovery and trajectory loading for KITTI and EuRoC-style folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from slam_eval.trajectory import PoseTrajectory3D

DatasetName = Literal["kitti", "euroc"]


@dataclass(frozen=True)
class ImageSequence:
    """A timestamped image stream discovered from a dataset folder."""

    name: str
    image_paths: list[Path]
    timestamps: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    """Images plus an optional ground-truth trajectory."""

    dataset: DatasetName
    root: Path
    sequence: str | None
    images: ImageSequence
    ground_truth: PoseTrajectory3D | None


def load_dataset(
    dataset: DatasetName,
    root: str | Path,
    sequence: str | None = None,
    camera: str = "cam0",
    image_limit: int | None = None,
) -> DatasetBundle:
    """Load KITTI or EuRoC image metadata and ground truth when available."""

    root_path = Path(root).expanduser().resolve()
    if dataset == "kitti":
        return load_kitti(root_path, sequence=sequence, camera=camera, image_limit=image_limit)
    if dataset == "euroc":
        return load_euroc(root_path, sequence=sequence, camera=camera, image_limit=image_limit)
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_kitti(
    root: Path,
    sequence: str | None = None,
    camera: str = "image_0",
    image_limit: int | None = None,
) -> DatasetBundle:
    """Load a KITTI odometry sequence.

    Expected layouts:
    - ``root/sequences/00/image_0/*.png`` and ``root/poses/00.txt``
    - or ``root/00/image_0/*.png`` and ``root/00.txt``
    """

    sequence = sequence or _guess_sequence(root)
    sequence_dir = _first_existing(root / "sequences" / sequence, root / sequence)
    image_dir = _first_existing(sequence_dir / camera, sequence_dir / "image_0", sequence_dir / "image_2")
    image_paths = _list_images(image_dir, limit=image_limit)
    times_path = _first_existing_or_none(sequence_dir / "times.txt", root / "sequences" / sequence / "times.txt")
    timestamps = _read_timestamps(times_path, count=len(image_paths))
    poses_path = _first_existing_or_none(root / "poses" / f"{sequence}.txt", root / f"{sequence}.txt")
    ground_truth = load_kitti_poses(poses_path, timestamps=timestamps) if poses_path else None
    return DatasetBundle("kitti", root, sequence, ImageSequence(camera, image_paths, timestamps), ground_truth)


def load_euroc(
    root: Path,
    sequence: str | None = None,
    camera: str = "cam0",
    image_limit: int | None = None,
) -> DatasetBundle:
    """Load an EuRoC MAV sequence.

    Expected layout: ``root[/sequence]/mav0/cam0/data.csv`` and optional
    ``mav0/state_groundtruth_estimate0/data.csv``.
    """

    sequence_dir = root / sequence if sequence else root
    mav0 = sequence_dir / "mav0"
    cam_dir = mav0 / camera
    csv_path = _first_existing(cam_dir / "data.csv")
    timestamps, image_paths = _read_euroc_image_csv(csv_path, cam_dir / "data")
    if image_limit is not None:
        timestamps = timestamps[:image_limit]
        image_paths = image_paths[:image_limit]
    gt_path = _first_existing_or_none(
        mav0 / "state_groundtruth_estimate0" / "data.csv",
    )
    ground_truth = load_euroc_ground_truth(gt_path) if gt_path else None
    return DatasetBundle("euroc", root, sequence, ImageSequence(camera, image_paths, timestamps), ground_truth)


def load_kitti_poses(path: str | Path, timestamps: Iterable[float] | None = None) -> PoseTrajectory3D:
    """Read KITTI odometry poses stored as 3x4 row-major matrices."""

    path = Path(path)
    matrices: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = [float(item) for item in line.split()]
            if len(values) != 12:
                raise ValueError(f"{path}:{line_number} expected 12 floats, got {len(values)}")
            matrix = np.eye(4, dtype=float)
            matrix[:3, :4] = np.asarray(values, dtype=float).reshape(3, 4)
            matrices.append(matrix)
    if timestamps is None:
        timestamps_array = np.arange(len(matrices), dtype=float)
    else:
        timestamps_array = np.asarray(list(timestamps), dtype=float)[: len(matrices)]
    return PoseTrajectory3D(timestamps_array, np.stack(matrices, axis=0))


def load_euroc_ground_truth(path: str | Path) -> PoseTrajectory3D:
    """Read EuRoC ground truth CSV into a trajectory.

    The EuRoC ground-truth state format stores nanosecond timestamps followed by
    position and quaternion as ``p_RS_R_x,y,z`` and ``q_RS_w,x,y,z``.
    """

    path = Path(path)
    timestamps: list[float] = []
    poses: list[np.ndarray] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [item.strip() for item in line.split(",")]
            if values[0] == "timestamp":
                continue
            if len(values) < 8:
                continue
            timestamp_s = float(values[0]) * 1e-9
            tx, ty, tz = (float(values[1]), float(values[2]), float(values[3]))
            qw, qx, qy, qz = (float(values[4]), float(values[5]), float(values[6]), float(values[7]))
            matrix = np.eye(4, dtype=float)
            matrix[:3, :3] = quaternion_to_matrix(qw, qx, qy, qz)
            matrix[:3, 3] = (tx, ty, tz)
            timestamps.append(timestamp_s)
            poses.append(matrix)
    if not poses:
        raise ValueError(f"No EuRoC ground-truth poses found in {path}")
    return PoseTrajectory3D(np.asarray(timestamps, dtype=float), np.stack(poses, axis=0))


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a unit quaternion in w,x,y,z order to a rotation matrix."""

    quaternion = np.asarray([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(quaternion)
    if norm == 0:
        return np.eye(3)
    qw, qx, qy, qz = quaternion / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def _read_euroc_image_csv(path: Path, image_dir: Path) -> tuple[np.ndarray, list[Path]]:
    timestamps: list[float] = []
    image_paths: list[Path] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            values = [item.strip() for item in line.split(",")]
            if values[0] == "timestamp":
                continue
            timestamps.append(float(values[0]) * 1e-9)
            image_paths.append(image_dir / values[1])
    if not image_paths:
        raise ValueError(f"No EuRoC image rows found in {path}")
    return np.asarray(timestamps, dtype=float), image_paths


def _list_images(image_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not paths:
        raise ValueError(f"No images found in {image_dir}")
    return paths if limit is None else paths[:limit]


def _read_timestamps(path: Path | None, count: int) -> np.ndarray:
    if path is None:
        return np.arange(count, dtype=float)
    values = np.loadtxt(path, dtype=float)
    return np.atleast_1d(values).astype(float)[:count]


def _guess_sequence(root: Path) -> str:
    sequences_dir = root / "sequences"
    if sequences_dir.exists():
        candidates = sorted(path.name for path in sequences_dir.iterdir() if path.is_dir())
        if candidates:
            return candidates[0]
    candidates = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if candidates:
        return candidates[0]
    raise ValueError("Could not infer KITTI sequence; pass --sequence explicitly")


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of these paths exist: {joined}")


def _first_existing_or_none(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None
