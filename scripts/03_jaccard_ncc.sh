#!/usr/bin/env bash
# Intra-class Jaccard@50 and nearest-centroid classification (Table 1) on the
# 50 ImageNet validation images per class.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. python experiments/jaccard.py \
    --n_per_class 50 --topk_k 50 \
    --output outputs/jaccard.json

PYTHONPATH=. python experiments/ncc.py \
    --mode pred --n_per_class 50 \
    --proto_root outputs/imagenet_130/prototypes \
    --proto_tag 130_train_ImageNet_cls0-1000 \
    --output outputs/ncc_full_val.json
