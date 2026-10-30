#!/usr/bin/env bash
# Top-K Feature Faithfulness (Table 1, Fig. 3): ON/OFF masking on the ImageNet
# validation split, once per prototype. Writes outputs/ff_masking/sae_base/.
set -euo pipefail
cd "$(dirname "$0")/.."

SAE_PATH="${SAE_PATH:-data/sae_weight/base/out.pt}"
PROTO_ROOT="${PROTO_ROOT:-outputs/imagenet_130/prototypes}"
TAG="${TAG:-130_train_ImageNet_cls0-1000}"
METHODS="${METHODS:-uniform random raw_attention gradcam attention_rollout legrad transformer_attribution grad_sam layercam}"

for m in $METHODS; do
    echo "=== $m"
    PYTHONPATH=. python experiments/feature_faithfulness.py \
        --sae_path "$SAE_PATH" \
        --cls_wise_sae_activation_path "$PROTO_ROOT/$m/cls_hgpatchsae_${TAG}.npy" \
        --save_name outputs/ff_masking
done

python experiments/plot_topk_sweep.py --layout lr --output outputs/masking_sweep.pdf   # Fig. 3
python experiments/paper_tables.py                                                     # Table 1 (FF)
