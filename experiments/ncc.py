"""
ncc.py — Nearest-centroid classification with on-the-fly val
feature extraction, under two heatmap-conditioning regimes.

What it does
------------
1. Loads class centroids from prototype files:
       {proto_root}/{method}/cls_hgpatchsae_{tag}.npy   shape (1000, 49152)
   These are the 130/class (or whatever the run used) weighted-SAE means.
   Centroids are L2-normalised. NCC runs in the full 49152-D SAE space
   (no random projection — the query is extracted in the same space).

2. For every ImageNet val image, extracts the per-method weighted-SAE
   feature vector with HGPatchSAE.run_all_methods, then classifies it by
   cosine nearest centroid.

3. The heatmap that pools the patches is conditioned on a text prompt.
   Two regimes control which label drives that prompt:

     --mode gt    text prompt = ground-truth class name.
                  Oracle upper bound: the pooling already "knows" the answer.

     --mode pred  text prompt = CLIP zero-shot predicted class name.
                  Realistic: first classify the image zero-shot, then use
                  that prediction to drive the heatmap, then do NCC.
                  No label leakage into the pooling step.

     --mode both  run both regimes (heatmaps are recomputed per regime
                  because gradient maps depend on the prompt).

For class-agnostic methods (raw_attention, attention_rollout, random) the
prompt does not affect the heatmap, so gt and pred give identical numbers.

Usage (from the repo root):
    PYTHONPATH="." python experiments/ncc.py \\
        --proto_root outputs/imagenet_130/prototypes \\
        --proto_tag  130_train_ImageNet_cls0-1000 \\
        --mode both \\
        --n_per_class 10 \\
        --output outputs/ncc.json

Set --n_per_class -1 to use all 50 val images/class (50 000 total, slow).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

# Make the repo root (src/, tasks/, hgpatchsae/) importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from datasets import load_dataset

NUM_CLASSES = 1000
SAE_DIM     = 49152

METHODS_9 = [
    "uniform",
    "legrad",
    "grad_sam",
    "attention_rollout",
    "gradcam",
    "raw_attention",
    "transformer_attribution",
    "layercam",
    "random",
]
# Methods whose heatmap ignores the text prompt → gt == pred.
CLASS_AGNOSTIC = {"raw_attention", "attention_rollout", "random", "uniform"}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--proto_root",
    default="outputs/imagenet_130/prototypes",
    help="Dir with {method}/cls_hgpatchsae_{tag}.npy centroid files.")
parser.add_argument("--proto_tag", default="130_train_ImageNet_cls0-1000",
    help="Filename stem: cls_hgpatchsae_<tag>.npy")
parser.add_argument("--methods", nargs="+", default=METHODS_9,
    help="Subset of methods to evaluate (must have a centroid file).")
parser.add_argument("--mode", choices=["gt", "pred", "both"], default="both",
    help="Heatmap conditioning regime for the val feature extraction.")
parser.add_argument("--n_per_class", type=int, default=10,
    help="Val images per class. -1 = all 50/class (50 000 total).")
parser.add_argument("--activation_mode", default="thresholded_real")
parser.add_argument("--sae_threshold", type=float, default=0.2)
parser.add_argument("--sae_device", default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device for the SAE forward pass. Default cuda (much faster on 50k images).")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="outputs/ncc.json")
args = parser.parse_args()


def l2norm_rows(a, eps=1e-8):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)


# ── Load centroids ────────────────────────────────────────────────────────────
print("Loading centroids...")
centroids = {}
for m in args.methods:
    path = os.path.join(args.proto_root, m, f"cls_hgpatchsae_{args.proto_tag}.npy")
    if not os.path.exists(path):
        print(f"  [{m}] MISSING centroid: {path}")
        continue
    proto = np.load(path).astype(np.float32)
    assert proto.shape == (NUM_CLASSES, SAE_DIM), f"{m}: bad shape {proto.shape}"
    centroids[m] = l2norm_rows(proto)
    print(f"  [{m}] centroid loaded {proto.shape}")
if not centroids:
    raise SystemExit("No centroids found — check --proto_root / --proto_tag.")
active_methods = [m for m in args.methods if m in centroids]

# ── Load HG-PatchSAE ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

from hgpatchsae import HGPatchSAE, load_imagenet_class_names  # noqa: E402

CLASS_NAMES = load_imagenet_class_names()
print(f"Loading HG-PatchSAE (sae_device={args.sae_device})...")
hg = HGPatchSAE(class_names=CLASS_NAMES, sae_device=args.sae_device)
print("Ready.\n")

# Precompute CLIP text features once for fast zero-shot inference (pred mode).
need_pred = args.mode in ("pred", "both")
text_feats = None
if need_pred:
    print("Precomputing zero-shot text features...")
    prompts = [f"a photo of a {c}" for c in CLASS_NAMES]
    txt = hg.processor(text=prompts, return_tensors="pt", padding=True).to(hg.device)
    with torch.no_grad():
        tf = hg.model.get_text_features(**txt)
        text_feats = (tf / tf.norm(dim=-1, keepdim=True))


def predict_zeroshot(pil_img):
    im = hg.processor(images=pil_img, return_tensors="pt").to(hg.device)
    with torch.no_grad():
        imf = hg.model.get_image_features(**im)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        logits = (imf @ text_feats.T).squeeze(0)
    return int(logits.argmax().item())


# ── Build balanced val subset ─────────────────────────────────────────────────
print("Loading ImageNet val split...")
ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="val")
labels = ds["label"]
c2i = defaultdict(list)
for i, l in enumerate(labels):
    c2i[l].append(i)

val_items = []
for c in range(NUM_CLASSES):
    idxs = c2i[c] if args.n_per_class < 0 else c2i[c][: args.n_per_class]
    for i in idxs:
        val_items.append((i, c))
rng = np.random.default_rng(args.seed)
rng.shuffle(val_items)
print(f"Evaluating on {len(val_items)} val images.\n")

# ── Accumulators ──────────────────────────────────────────────────────────────
modes = ["gt", "pred"] if args.mode == "both" else [args.mode]
acc = {mode: {m: {"top1": 0, "top5": 0, "n": 0} for m in active_methods} for mode in modes}
# Per-class correct/total counts so we can report mean +/- std across classes.
_NC = len(CLASS_NAMES)
percls = {mode: {m: {"c1": np.zeros(_NC), "c5": np.zeros(_NC), "n": np.zeros(_NC)}
                 for m in active_methods} for mode in modes}
zeroshot_correct = 0
zeroshot_n = 0
errors = 0


def score_vec(vec, centroid, true_cls):
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    s = centroid @ v                       # (1000,)
    pred = int(s.argmax())
    top1 = int(pred == true_cls)
    top5 = int(true_cls in np.argpartition(s, -5)[-5:])
    return top1, top5, pred


# Per-image predictions for the NCC-vs-CLIP agreement analysis (ncc_agreement.py).
# For every processed image we log the true class, CLIP's zero-shot prediction,
# and each method's NCC top-1 prediction (per mode). -1 marks a missing value.
perimg_true = []
perimg_clip = []
perimg_pred = {mode: {m: [] for m in active_methods} for mode in modes}


for hf_idx, true_cls in tqdm(val_items, desc="val"):
    try:
        img = ds[hf_idx]["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Decide the conditioning label per regime.
        cond_label = {}
        cond_label["gt"] = CLASS_NAMES[true_cls]
        clip_pred = -1
        if need_pred:
            pred_idx = predict_zeroshot(img)
            cond_label["pred"] = CLASS_NAMES[pred_idx]
            clip_pred = pred_idx
            zeroshot_correct += int(pred_idx == true_cls)
            zeroshot_n += 1

        cur_pred = {mode: {m: -1 for m in active_methods} for mode in modes}
        for mode in modes:
            feats = hg.run_all_methods(
                img,
                true_class=cond_label[mode],
                methods=active_methods,
                activation_mode=args.activation_mode,
                sae_threshold=args.sae_threshold,
                random_seed=args.seed,
            )
            for m in active_methods:
                if m not in feats:
                    continue
                t1, t5, pred = score_vec(feats[m], centroids[m], true_cls)
                acc[mode][m]["top1"] += t1
                acc[mode][m]["top5"] += t5
                acc[mode][m]["n"]    += 1
                percls[mode][m]["c1"][true_cls] += t1
                percls[mode][m]["c5"][true_cls] += t5
                percls[mode][m]["n"][true_cls]  += 1
                cur_pred[mode][m] = pred

        # Record per-image predictions (aligned across all arrays).
        perimg_true.append(true_cls)
        perimg_clip.append(clip_pred)
        for mode in modes:
            for m in active_methods:
                perimg_pred[mode][m].append(cur_pred[mode][m])

    except Exception as e:
        errors += 1
        if errors <= 5:
            tqdm.write(f"[{hf_idx}] {e}")

# ── Report ────────────────────────────────────────────────────────────────────
results = {"proto_tag": args.proto_tag, "n_images": len(val_items),
           "zeroshot_top1": round(100 * zeroshot_correct / max(zeroshot_n, 1), 2)
           if need_pred else None,
           "modes": {}}

print("\n" + "=" * 70)
if need_pred:
    print(f"CLIP zero-shot top-1: {results['zeroshot_top1']:.2f}%")
print("=" * 70)
for mode in modes:
    print(f"\n[{mode.upper()} conditioning]")
    print(f"  {'Method':<28} {'Top-1':>8} {'Top-5':>8}")
    print(f"  {'-'*46}")
    results["modes"][mode] = {}
    for m in active_methods:
        n = max(acc[mode][m]["n"], 1)
        t1 = 100 * acc[mode][m]["top1"] / n
        t5 = 100 * acc[mode][m]["top5"] / n
        # Per-class accuracy (%) over classes that received at least one image,
        # then mean +/- std across those classes.
        pc = percls[mode][m]
        seen = pc["n"] > 0
        pc1 = 100 * pc["c1"][seen] / pc["n"][seen]
        pc5 = 100 * pc["c5"][seen] / pc["n"][seen]
        results["modes"][mode][m] = {
            "top1": round(t1, 2), "top5": round(t5, 2), "n": n,
            "top1_perclass_mean": round(float(pc1.mean()), 2),
            "top1_perclass_std":  round(float(pc1.std()), 2),
            "top5_perclass_mean": round(float(pc5.mean()), 2),
            "top5_perclass_std":  round(float(pc5.std()), 2),
            "n_classes": int(seen.sum()),
        }
        print(f"  {m:<28} {t1:>7.2f}% {t5:>7.2f}%   "
              f"(per-class Top-1 {pc1.mean():.2f} +/- {pc1.std():.2f})")
print("=" * 70)
print(f"Errors: {errors}")

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {args.output}")

# ── Per-image predictions for the agreement analysis ────────────────────────────
# Per-image predictions, saved for downstream analysis.
perimg_path = os.path.splitext(args.output)[0] + "_perimg.npz"
arrays = {
    "true":      np.asarray(perimg_true, dtype=np.int32),
    "clip_pred": np.asarray(perimg_clip, dtype=np.int32),
    "methods":   np.asarray(active_methods),
    "modes":     np.asarray(modes),
}
for mode in modes:
    for m in active_methods:
        arrays[f"pred__{mode}__{m}"] = np.asarray(perimg_pred[mode][m], dtype=np.int32)
np.savez_compressed(perimg_path, **arrays)
print(f"Saved: {perimg_path}  (per-image preds: {len(perimg_true)} images)")
