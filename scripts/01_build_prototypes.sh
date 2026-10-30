#!/usr/bin/env bash
# Build HG-PatchSAE class prototypes from 130 ImageNet training images per class
# (Sec. 4.1, 4.4). Writes outputs/imagenet_130/prototypes/<method>/cls_hgpatchsae_*.npy.
# For multi-GPU runs see the docstring of experiments/build_prototypes.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHONPATH=. python experiments/build_prototypes.py \
    --max_per_class 130 \
    --methods legrad grad_sam attention_rollout gradcam raw_attention \
              transformer_attribution layercam random \
    --activation_mode thresholded_real --sae_threshold 0.2 \
    --output_root outputs/imagenet_130 \
    "$@"
