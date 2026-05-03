#!/usr/bin/env bash
set -euo pipefail

slam-eval orb-vo \
  --dataset kitti \
  --root data/kitti_odometry \
  --sequence 00 \
  --camera image_0 \
  --fx 718.856 --fy 718.856 --cx 607.1928 --cy 185.2157 \
  --image-limit 500 \
  --with-scale \
  --output outputs/kitti_00_orb_vo.txt

