#!/usr/bin/env bash
# Energy-based Pointing Game (Table 2) on 500 ImageNet validation images, then
# the EPG vs FF_100 correlation (Sec. 6). Needs the ILSVRC val images and
# bounding-box XMLs (see README).
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE_DIR="${IMAGE_DIR:-data/imagenet/ILSVRC/Data/CLS-LOC/val}"
BBOX_DIR="${BBOX_DIR:-data/imagenet/ILSVRC/Annotations/CLS-LOC/val}"

PYTHONPATH=. python experiments/epg.py \
    --image_dir "$IMAGE_DIR" --bbox_dir "$BBOX_DIR" \
    --n_samples 500 --seed 42 \
    --output outputs/epg_iou_results.json

python experiments/paper_tables.py   # Table 2 + Pearson / Spearman
