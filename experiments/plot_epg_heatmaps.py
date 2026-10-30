"""
plot_epg_heatmaps.py — per-method heatmaps for ONE image with the ground-truth
bounding box drawn on each panel and that panel's EPG in the title. No Original
panel.

Uses a val image (which has a GT bbox) and reuses epg.py's bbox + EPG code, so
the per-image EPG is computed exactly as in the EPG experiment: pool the heatmap
to the 14x14 patch grid and take the fraction of its energy inside the box.

Usage (repo root, GPU):
    PYTHONPATH=. python experiments/plot_epg_heatmaps.py \
        --synset n02099601 --class_name "Golden Retriever" \
        --output_dir outputs/epg_heatmaps_golden
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image

# Make the repo root (src/, tasks/, hgpatchsae/) importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import epg  # bbox parsing, EPG, patch-grid pooling  # noqa: E402

DISPLAY = {
    "raw_attention": "Raw Attention", "attention_rollout": "Attention Rollout",
    "gradcam": "GradCAM", "legrad": "LeGrad", "grad_sam": "Grad-SAM",
    "transformer_attribution": "Transformer Attribution", "layercam": "LayerCAM",
}
METHOD_KEYS = {v: k for k, v in DISPLAY.items()}
GRID_ROW1 = ["raw_attention", "attention_rollout", "gradcam"]
GRID_ROW2 = ["legrad", "grad_sam", "transformer_attribution", "layercam"]

p = argparse.ArgumentParser()
p.add_argument("--synset", default="n02099601", help="WordNet synset (n02099601 = golden retriever).")
p.add_argument("--class_name", default="Golden Retriever", help="Text prompt for the heatmaps.")
p.add_argument("--image_dir", default="data/imagenet/ILSVRC/Data/CLS-LOC/val")
p.add_argument("--bbox_dir", default="data/imagenet/ILSVRC/Annotations/CLS-LOC/val")
p.add_argument("--min_cov", type=float, default=0.2, help="Min bbox coverage of the picked image.")
p.add_argument("--max_cov", type=float, default=0.6, help="Max bbox coverage.")
p.add_argument("--pick", type=int, default=0, help="Skip this many eligible images before starting.")
p.add_argument("--num_images", type=int, default=1, help="How many eligible images to render.")
p.add_argument("--epg_mode", choices=["pixel", "patch"], default="pixel",
    help="pixel = full-resolution EPG (standard Energy Pointing Game); "
         "patch = 14x14 pooled EPG (fractional patch mask, as in epg.py).")
p.add_argument("--style", choices=["epg", "qual"], default="epg",
    help="epg = no Original, GT box + EPG on each panel. "
         "qual = Original on top, heatmaps only (no box, no EPG in titles).")
p.add_argument("--cmap", default="jet")
p.add_argument("--overlay_alpha", type=float, default=0.5)
p.add_argument("--box_color", default="lime")
p.add_argument("--sae_device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--output_dir", default="outputs/epg_heatmaps_golden")
args = p.parse_args()
os.makedirs(args.output_dir, exist_ok=True)


def clip_crop(img, target=224):
    """Replicate CLIP preprocessing (resize short side -> target, center crop),
    matching epg.transform_bbox_to_clip so the box and heatmap align."""
    w, h = img.size
    if w < h:
        neww, newh = target, round(h * target / w)
    else:
        newh, neww = target, round(w * target / h)
    img2 = img.resize((neww, newh), Image.BICUBIC)
    cx, cy = (neww - target) // 2, (newh - target) // 2
    return img2.crop((cx, cy, cx + target, cy + target))


def epg_pixel(hm, clip_boxes):
    """Standard pixel-level EPG: fraction of the relu'd heatmap's energy that
    falls inside the (binary) box, at the heatmap's native resolution. The box
    is given in 224-space (clip crop) and scaled to the heatmap size."""
    hm = np.clip(np.asarray(hm, np.float64), 0, None)
    H, W = hm.shape
    sy, sx = H / 224.0, W / 224.0
    mask = np.zeros((H, W), np.float64)
    for (x0, y0, x1, y1) in clip_boxes:
        mask[int(np.floor(y0 * sy)):int(np.ceil(y1 * sy)),
             int(np.floor(x0 * sx)):int(np.ceil(x1 * sx))] = 1.0
    tot = hm.sum()
    return float((hm * mask).sum() / tot) if tot > 0 else 0.0


# ── Pick eligible val images of the synset (single box, coverage window) ──────
print(f"Searching {args.bbox_dir} for synset {args.synset} ...")
cand = sorted(subprocess.run(["grep", "-rl", args.synset, args.bbox_dir],
                             capture_output=True, text=True).stdout.split())
eligible = []
for xml in cand:
    boxes, w, h, syn = epg.parse_imagenet_bbox(xml)
    if not boxes:
        continue
    cov = epg.bbox_coverage_fraction(boxes, w, h)
    if len(boxes) == 1 and args.min_cov <= cov <= args.max_cov:
        eligible.append((xml, boxes, w, h, cov))
chosen_list = eligible[args.pick: args.pick + args.num_images]
if not chosen_list:
    raise SystemExit(f"No eligible {args.synset} images with 1 box in "
                     f"[{args.min_cov}, {args.max_cov}] coverage found.")
print(f"{len(eligible)} eligible; rendering {len(chosen_list)}.")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.gridspec as gridspec  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

mpl.rcParams.update({"font.family": "serif", "font.size": 11, "axes.titlesize": 11,
                     "axes.titlepad": 5, "savefig.dpi": 300})

# ── HG-PatchSAE (loaded once for all images) ──────────────────────────────────────
print("Loading HG-PatchSAE ...")
model, class_names = epg.load_model(sae_device=args.sae_device)
print("Ready.\n")


def render_one(xml, boxes, img_w, img_h, cov):
    stem = os.path.splitext(os.path.basename(xml))[0]
    print(f"[{stem}] coverage {cov:.2f}")
    pil = Image.open(os.path.join(args.image_dir, stem + ".JPEG")).convert("RGB")
    crop224 = np.array(clip_crop(pil, 224))
    bbox_frac = epg.boxes_to_patch_mask(boxes, img_w, img_h)          # (14,14) fractional
    clip_boxes = [b for b in (epg.transform_bbox_to_clip(bx, img_w, img_h, 224)
                              for bx in boxes) if b is not None]      # 224-space boxes

    heatmaps, epgs, epgs_patch = {}, {}, {}
    for disp, key in METHOD_KEYS.items():
        try:
            hm = model.get_heatmap(image=pil, text_prompt=args.class_name, method=key)
            hm = hm.detach().cpu().numpy().squeeze().astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"  [{key}] heatmap failed: {e}")
            heatmaps[key] = None; epgs[key] = float("nan"); epgs_patch[key] = float("nan")
            continue
        # pixel-level (standard) EPG and the older 14x14-patch EPG for comparison
        ep_pix = epg_pixel(hm, clip_boxes)
        hm14 = epg.pixel_heatmap_to_grid(torch.from_numpy(hm), 14)
        ep_patch = epg.compute_epg(hm14, bbox_frac)
        epgs_patch[key] = ep_patch
        epgs[key] = ep_pix if args.epg_mode == "pixel" else ep_patch
        hp = np.clip(hm, 0, None)
        heatmaps[key] = (hp - hp.min()) / (hp.max() - hp.min() + 1e-8)
        print(f"    {disp:<26} pixel={ep_pix:.3f}  patch={ep_patch:.3f}")

    NCOL = 12
    qual = args.style == "qual"
    if qual:
        fig = plt.figure(figsize=(11.5, 9))
        gs = gridspec.GridSpec(3, NCOL, figure=fig, wspace=0.06, hspace=0.18,
                               height_ratios=[1.5, 1, 1])
        ow = 6
        axo = fig.add_subplot(gs[0, (NCOL - ow) // 2:(NCOL + ow) // 2])
        axo.imshow(crop224, interpolation="lanczos")
        axo.set_title("Original", fontweight="bold"); axo.set_axis_off()
        rows = (1, 2)
    else:
        fig = plt.figure(figsize=(11.5, 6.6))
        gs = gridspec.GridSpec(2, NCOL, figure=fig, wspace=0.06, hspace=0.22)
        rows = (0, 1)

    def _draw(ax, key):
        ax.set_axis_off()
        ax.imshow(crop224, interpolation="lanczos")
        hm = heatmaps.get(key)
        if hm is not None:
            ax.imshow(hm, cmap=args.cmap, alpha=args.overlay_alpha, vmin=0, vmax=1,
                      interpolation="lanczos")
        if not qual:
            for (x0, y0, x1, y1) in clip_boxes:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       edgecolor=args.box_color, linewidth=2.2))
        ax.set_title(DISPLAY[key] if qual
                     else f"{DISPLAY[key]}\nEPG {epgs.get(key, float('nan')):.3f}")

    def _row(row_idx, methods):
        k = len(methods); span = NCOL // k; off = (NCOL - span * k) // 2
        for c, key in enumerate(methods):
            _draw(fig.add_subplot(gs[row_idx, off + c * span: off + c * span + span]), key)

    _row(rows[0], GRID_ROW1)
    _row(rows[1], GRID_ROW2)

    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.5])
    sm = mpl.cm.ScalarMappable(cmap=args.cmap, norm=mpl.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cbar_ax); cb.set_ticks([0, 0.5, 1])
    cb.set_ticklabels(["Low", "Mid", "High"]); cb.ax.tick_params(labelsize=9)

    prefix = "qual_heatmaps" if qual else "epg_heatmaps"
    base = os.path.join(args.output_dir, f"{prefix}_{args.synset}_{stem}")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(base + "." + ext, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {base}.png\n")


for c in chosen_list:
    render_one(*c)
print("Done.")
