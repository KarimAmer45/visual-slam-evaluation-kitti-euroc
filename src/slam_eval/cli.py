"""Command-line interface for the evaluation toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slam_eval.datasets import load_dataset, load_euroc_ground_truth, load_kitti_poses
from slam_eval.demo import generate_demo_outputs
from slam_eval.metrics import summarize_ate
from slam_eval.plotting import plot_trajectories, plot_translation_errors
from slam_eval.trajectory import align_umeyama, load_tum_trajectory, save_tum_trajectory
from slam_eval.vo import CameraIntrinsics, OrbVoConfig, estimate_orb_visual_odometry


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slam-eval", description="Evaluate KITTI/EuRoC VO trajectories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("eval", help="align an estimated trajectory and compute ATE")
    eval_parser.add_argument("--estimate", required=True, help="estimated trajectory in TUM format")
    eval_parser.add_argument("--ground-truth", required=True, help="ground truth trajectory path")
    eval_parser.add_argument("--ground-truth-format", choices=["tum", "kitti", "euroc"], default="tum")
    eval_parser.add_argument("--with-scale", action="store_true", help="use Sim(3) alignment for monocular VO")
    eval_parser.add_argument("--max-time-delta", type=float, default=None)
    eval_parser.add_argument("--plot-dir", default="outputs")
    eval_parser.set_defaults(func=run_eval)

    vo_parser = subparsers.add_parser("orb-vo", help="run the ORB VO baseline on a dataset folder")
    vo_parser.add_argument("--dataset", choices=["kitti", "euroc"], required=True)
    vo_parser.add_argument("--root", required=True)
    vo_parser.add_argument("--sequence")
    vo_parser.add_argument("--camera", default=None)
    vo_parser.add_argument("--fx", type=float, required=True)
    vo_parser.add_argument("--fy", type=float, required=True)
    vo_parser.add_argument("--cx", type=float, required=True)
    vo_parser.add_argument("--cy", type=float, required=True)
    vo_parser.add_argument("--image-limit", type=int, default=None)
    vo_parser.add_argument("--output", default="outputs/orb_vo_tum.txt")
    vo_parser.add_argument("--plot-dir", default="outputs")
    vo_parser.add_argument("--with-scale", action="store_true")
    vo_parser.set_defaults(func=run_orb_vo)

    demo_parser = subparsers.add_parser("demo", help="generate synthetic trajectory plots and metrics")
    demo_parser.add_argument("--output-dir", default="docs/results")
    demo_parser.set_defaults(func=run_demo)
    return parser


def run_eval(args: argparse.Namespace) -> None:
    estimate = load_tum_trajectory(args.estimate)
    reference = _load_ground_truth(args.ground_truth, args.ground_truth_format)
    alignment = align_umeyama(
        estimate,
        reference,
        with_scale=args.with_scale,
        max_time_delta=args.max_time_delta,
    )
    metrics = summarize_ate(alignment.aligned, alignment.matched_reference)
    plot_dir = Path(args.plot_dir)
    plot_trajectories(alignment.aligned, alignment.matched_reference, plot_dir / "trajectory.png")
    plot_translation_errors(alignment.aligned, alignment.matched_reference, plot_dir / "ate.png")
    print(json.dumps({"metrics": metrics, "scale": alignment.scale}, indent=2))


def run_orb_vo(args: argparse.Namespace) -> None:
    camera = args.camera or ("image_0" if args.dataset == "kitti" else "cam0")
    bundle = load_dataset(
        args.dataset,
        args.root,
        sequence=args.sequence,
        camera=camera,
        image_limit=args.image_limit,
    )
    trajectory = estimate_orb_visual_odometry(
        bundle.images.image_paths,
        bundle.images.timestamps,
        CameraIntrinsics(args.fx, args.fy, args.cx, args.cy),
        OrbVoConfig(),
    )
    save_tum_trajectory(args.output, trajectory)

    result: dict[str, object] = {
        "frames": len(trajectory.poses),
        "trajectory": args.output,
    }
    if bundle.ground_truth is not None:
        alignment = align_umeyama(trajectory, bundle.ground_truth, with_scale=args.with_scale)
        metrics = summarize_ate(alignment.aligned, alignment.matched_reference)
        plot_dir = Path(args.plot_dir)
        plot_trajectories(alignment.aligned, alignment.matched_reference, plot_dir / "orb_vo_trajectory.png")
        plot_translation_errors(alignment.aligned, alignment.matched_reference, plot_dir / "orb_vo_ate.png")
        result["metrics"] = metrics
        result["scale"] = alignment.scale
    print(json.dumps(result, indent=2))


def run_demo(args: argparse.Namespace) -> None:
    print(json.dumps(generate_demo_outputs(args.output_dir), indent=2))


def _load_ground_truth(path: str, trajectory_format: str):
    if trajectory_format == "tum":
        return load_tum_trajectory(path)
    if trajectory_format == "kitti":
        return load_kitti_poses(path)
    if trajectory_format == "euroc":
        return load_euroc_ground_truth(path)
    raise ValueError(f"Unknown trajectory format: {trajectory_format}")


if __name__ == "__main__":
    raise SystemExit(main())

