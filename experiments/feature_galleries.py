"""
feature_galleries.py
====================
For each requested SAE feature index, save an image grid of the top-k
maximally activating training images. Uses the max-activating image
indices that PatchSAE ships and HGPatchSAE already loads at init, so no
re-computation of activations is needed.

Usage:
    PYTHONPATH=. python experiments/feature_galleries.py \\
        --features 44690 20516 41134 2396 37313 \\
        --top_k 9 \\
        --output_dir outputs/feature_galleries
"""

import argparse
import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Make the repo root (src/, tasks/, hgpatchsae/) importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--features", type=int, nargs="+", required=True,
    help="SAE feature indices to visualise, e.g. 44690 20516 41134")
parser.add_argument("--top_k", type=int, default=9,
    help="Number of top-activating images per feature to show.")
parser.add_argument("--dataset", default="imagenet-sketch",
    help="Which dataset's max-activating images to draw from. HGPatchSAE loads "
         "'imagenet-sketch' by default; other options depend on your setup.")
parser.add_argument("--sae_device", default="cpu",
    help="HGPatchSAE sae_device. cpu is fine — we do no forward passes here.")
parser.add_argument("--output_dir", default="outputs/feature_galleries")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


# ── Load HG-PatchSAE (only for the max_act_imgs table it exposes) ──────────
from hgpatchsae import HGPatchSAE  # noqa: E402

print(f"Loading HG-PatchSAE (sae_device={args.sae_device})...")
hg = HGPatchSAE(sae_device=args.sae_device)
print("Ready.\n")

if args.dataset not in hg.max_act_imgs:
    raise SystemExit(
        f"'{args.dataset}' has no max_act_imgs table. Available: "
        f"{list(hg.max_act_imgs.keys())}"
    )
if args.dataset not in hg.datasets:
    raise SystemExit(
        f"'{args.dataset}' is not among the loaded datasets: "
        f"{list(hg.datasets.keys())}"
    )

max_act_table = hg.max_act_imgs[args.dataset]     # shape (F, N_top) int32
dataset       = hg.datasets[args.dataset]         # HF dataset
print(f"max_act table shape: {tuple(max_act_table.shape)}   "
      f"dataset size: {len(dataset)}\n")


# ── Render one gallery per feature ───────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.titlesize":    9,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
})


def render_gallery(feat_idx, top_k):
    if feat_idx < 0 or feat_idx >= max_act_table.shape[0]:
        print(f"  [{feat_idx}] out of range (0..{max_act_table.shape[0]-1})")
        return
    row = max_act_table[feat_idx].tolist()[:top_k]
    ncols = min(top_k, 5) if top_k >= 5 else top_k
    nrows = (top_k + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.0, nrows * 2.0))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [a for r in axes for a in r]

    labels = []
    for ax, img_idx in zip(axes, row):
        try:
            item = dataset[int(img_idx)]
            img = item["image"] if "image" in item else item[list(item.keys())[0]]
            if img.mode != "RGB":
                img = img.convert("RGB")
            lbl = item.get("label", "?")
            lbl_str = hg.imagenet_classes[lbl] if isinstance(lbl, int) \
                and 0 <= lbl < len(hg.imagenet_classes) else str(lbl)
            labels.append(lbl_str)
            ax.imshow(np.array(img.resize((160, 160))))
            ax.set_title(lbl_str, fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"err\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7)
        ax.set_axis_off()
    for ax in axes[len(row):]:
        ax.set_axis_off()

    fig.suptitle(f"Feature {feat_idx} — top-{top_k} max-activating images ({args.dataset})",
                 fontsize=10, fontweight="bold", y=1.02)
    out = os.path.join(args.output_dir, f"feat_{feat_idx:05d}.pdf")
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}    labels: {labels}")


for f in args.features:
    render_gallery(f, args.top_k)

print("\nDone.")
