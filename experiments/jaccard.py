"""
jaccard.py — Intra-class Jaccard@K with on-the-fly feature extraction.

Jaccard measures whether images of the SAME class activate the same top-K SAE
features, so it needs per-image feature vectors, not the class prototype.
This script extracts each image's HG-PatchSAE feature vector via HGPatchSAE
(the same pipeline used to build the prototypes), takes the top-K active SAE
features per image, and computes the mean pairwise Jaccard similarity within
each class, then averages over classes.

For class-conditional methods the heatmap is conditioned on the true class,
matching how the prototypes were built. run_all_methods extracts every method
in one shared forward+backward pass per image, and also returns the uniform
and random baselines for free.

This evaluates on the same data as the NCC script (the ImageNet val split),
so pass the same --n_per_class to both for a like-for-like comparison.

Usage (from the repo root):
    PYTHONPATH="." python experiments/jaccard.py \\
        --n_per_class 10 --topk_k 50 \\
        --output outputs/jaccard.json

The val split has 50 images/class, so --n_per_class maxes out at 50.
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

# Display methods. uniform is emitted by run_all_methods unconditionally;
# random must be requested explicitly (run_all_methods only fabricates the
# random weight vector when "random" is in the methods list).
METHODS = [
    "uniform",
    "random",
    "raw_attention",
    "attention_rollout",
    "gradcam",
    "legrad",
    "transformer_attribution",
    "grad_sam",
    "layercam",
]
REQUEST = [m for m in METHODS if m != "uniform"]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--methods", nargs="+", default=None,
    help="Subset of methods to evaluate. Default = all. Pass e.g. "
         "'--methods random' to compute just one for the missing-row case. "
         "uniform is included automatically when present in METHODS.")
parser.add_argument("--n_per_class", type=int, default=10,
    help="Val images per class. Match the value used for NCC. Max 50 on val.")
parser.add_argument("--topk_k", type=int, default=50,
    help="K for the top-K active SAE features per image.")
parser.add_argument("--max_pairs", type=int, default=300,
    help="Max random image pairs per class (speed vs variance).")
parser.add_argument("--split", default="val",
    help="HF split to sample class images from. val matches the NCC setup.")
parser.add_argument("--activation_mode", default="thresholded_real")
parser.add_argument("--sae_threshold", type=float, default=0.2)
parser.add_argument("--sae_device", default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device for the SAE forward pass.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", default="outputs/jaccard.json")
args = parser.parse_args()


# Apply the user's --methods filter (keeps ordering from METHODS).
if args.methods:
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {unknown}. Allowed: {METHODS}")
    METHODS = [m for m in METHODS if m in set(args.methods)]
    REQUEST = [m for m in METHODS if m != "uniform"]
    print(f"Filtered methods -> {METHODS}")


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

# ── Build balanced class -> image-index map ───────────────────────────────────
print(f"Loading ImageNet {args.split} split label index...")
ds = load_dataset("evanarlian/imagenet_1k_resized_256", split=args.split)
labels = ds["label"]
c2i = defaultdict(list)
for i, l in enumerate(labels):
    c2i[l].append(i)

rng = np.random.default_rng(args.seed)
K = args.topk_k


def topk_set(vec):
    """Indices of the K largest entries of vec, as a Python set."""
    if vec.size <= K:
        return set(np.nonzero(vec)[0].tolist())
    idx = np.argpartition(vec, -K)[-K:]
    return set(idx.tolist())


def jaccard_pairs(sets, max_pairs):
    n = len(sets)
    if n < 2:
        return None
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[p] for p in chosen]
    vals = []
    for i, j in pairs:
        u = len(sets[i] | sets[j])
        vals.append(len(sets[i] & sets[j]) / u if u else 0.0)
    return float(np.mean(vals))


# ── Per-class accumulation ────────────────────────────────────────────────────
per_class_jacc = {m: [] for m in METHODS}
errors = 0

for c in tqdm(range(NUM_CLASSES), desc="classes"):
    idxs = c2i[c][: args.n_per_class]
    name = CLASS_NAMES[c]

    # collect top-K sets per method across this class's images
    sets_by_method = {m: [] for m in METHODS}
    for hf_idx in idxs:
        try:
            img = ds[hf_idx]["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            feats = hg.run_all_methods(
                img,
                true_class=name,
                methods=REQUEST,
                activation_mode=args.activation_mode,
                sae_threshold=args.sae_threshold,
                random_seed=args.seed,
            )
            for m in METHODS:
                if m in feats:
                    sets_by_method[m].append(topk_set(np.asarray(feats[m], dtype=np.float32)))
        except Exception as e:
            errors += 1
            if errors <= 5:
                tqdm.write(f"[cls {c} idx {hf_idx}] {e}")

    for m in METHODS:
        j = jaccard_pairs(sets_by_method[m], args.max_pairs)
        if j is not None:
            per_class_jacc[m].append(j)

# ── Report ────────────────────────────────────────────────────────────────────
uniform_mean = (float(np.mean(per_class_jacc["uniform"]))
                if per_class_jacc.get("uniform") else None)

results = {}
print("\n" + "=" * 60)
print(f"  Jaccard@{K}  ({args.n_per_class} imgs/class, {args.split} split)")
print("=" * 60)
print(f"  {'Method':<26} {'Mean':>8} {'Std':>8} {'Δ vs Uniform':>14}")
print(f"  {'-'*56}")
for m in METHODS:
    vals = per_class_jacc[m]
    if not vals:
        print(f"  {m:<26}   N/A")
        results[m] = None
        continue
    mean, std = float(np.mean(vals)), float(np.std(vals))
    delta = "---" if uniform_mean is None or m == "uniform" else f"{mean - uniform_mean:+.4f}"
    results[m] = {"mean": round(mean, 4), "std": round(std, 4), "delta": delta,
                  "n_classes": len(vals)}
    print(f"  {m:<26} {mean:>8.4f} {std:>8.4f} {delta:>14}")
print("=" * 60)
print(f"Errors: {errors}")

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w") as f:
    json.dump({"topk_k": K, "n_per_class": args.n_per_class, "split": args.split,
               "results": results}, f, indent=2)
print(f"\nSaved: {args.output}")
