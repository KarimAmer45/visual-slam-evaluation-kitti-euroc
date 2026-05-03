import numpy as np

from slam_eval.metrics import ate_rmse
from slam_eval.trajectory import PoseTrajectory3D, align_umeyama


def _trajectory(xyz: np.ndarray) -> PoseTrajectory3D:
    poses = np.repeat(np.eye(4)[None, :, :], repeats=len(xyz), axis=0)
    poses[:, :3, 3] = xyz
    return PoseTrajectory3D(np.arange(len(xyz), dtype=float), poses)


def test_umeyama_alignment_recovers_translation_and_scale():
    reference = _trajectory(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 1.0],
            ]
        )
    )
    estimate = _trajectory(reference.xyz * 0.5 + np.asarray([2.0, -1.0, 0.4]))

    alignment = align_umeyama(estimate, reference, with_scale=True)

    assert np.isclose(alignment.scale, 2.0)
    assert ate_rmse(alignment.aligned, alignment.matched_reference) < 1e-10
