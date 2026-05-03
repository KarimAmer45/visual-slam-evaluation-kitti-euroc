#!/usr/bin/env bash
set -euo pipefail

slam-eval orb-vo \
  --dataset euroc \
  --root data/euroc/MH_01_easy \
  --camera cam0 \
  --fx 458.654 --fy 457.296 --cx 367.215 --cy 248.375 \
  --image-limit 500 \
  --with-scale \
  --output outputs/euroc_mh01_orb_vo.txt

