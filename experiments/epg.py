"""
epg.py

Energy-based Pointing Game (EPG) and IoU evaluation of the heatmap methods
on ImageNet-1k validation (Table 2).

The heatmaps come from HGPatchSAE.get_heatmap, i.e. the same maps used as
pooling weights in the Feature Faithfulness, NCC and Jaccard experiments.

References:
  Wang, H. et al. "Score-CAM." CVPR Workshops, 2020.
  Zhang, J. et al. "Top-Down Neural Attention by Excitation Backprop." IJCV, 2018.

Default protocol (Score-CAM):
  - 500 random ImageNet val images
  - Exclude images whose bbox union covers > 50% of the image
  - CLIP ViT-B/16 backbone, 14x14 patch grid

Usage (from the repo root):
  PYTHONPATH=. python experiments/epg.py \
      --image_dir data/imagenet/ILSVRC/Data/CLS-LOC/val \
      --bbox_dir  data/imagenet/ILSVRC/Annotations/CLS-LOC/val \
      --output    outputs/epg_iou_results.json
"""

import json
import os
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Make the repo root (src/, tasks/, hgpatchsae/) importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


# =====================================================================
# Section 1. Core metrics
# =====================================================================

def compute_epg(heatmap: torch.Tensor, bbox_mask: torch.Tensor) -> float:
    """
    Energy-based Pointing Game (Wang et al., 2020).

    Args:
        heatmap: (G, G) non-negative tensor on the patch grid.
                 Need not be L1-normalized; we compute a ratio.
        bbox_mask: (G, G) tensor in [0, 1]. Use fractional coverage
                   (each cell = fraction of its 16x16 area in the bbox).

    Returns:
        EPG in [0, 1] = fraction of heatmap mass falling in the bbox.
    """
    assert heatmap.shape == bbox_mask.shape, (
        f"shape mismatch {heatmap.shape} vs {bbox_mask.shape}"
    )
    if (heatmap < 0).any():
        heatmap = F.relu(heatmap)
    total = heatmap.sum()
    if total <= 0:
        return 0.0
    inside = (heatmap * bbox_mask).sum()
    return float(inside / total)


def compute_iou(
    heatmap: torch.Tensor,
    bbox_mask_binary: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    IoU between binarized heatmap and binary bbox mask.

    Args:
        heatmap: (G, G) tensor. Will be min-max normalized to [0, 1].
        bbox_mask_binary: (G, G) tensor in {0, 1}.
        threshold: binarize the normalized heatmap at this value.

    Returns:
        IoU in [0, 1].
    """
    if (heatmap < 0).any():
        heatmap = F.relu(heatmap)
    h_min, h_max = heatmap.min(), heatmap.max()
    if (h_max - h_min) < 1e-8:
        return 0.0
    h_norm = (heatmap - h_min) / (h_max - h_min)
    pred_mask = (h_norm >= threshold).float()

    intersection = (pred_mask * bbox_mask_binary).sum()
    union = ((pred_mask + bbox_mask_binary) > 0).float().sum()
    if union <= 0:
        return 0.0
    return float(intersection / union)


# =====================================================================
# Section 2. Bounding box utilities
# =====================================================================

def parse_imagenet_bbox(xml_path: str):
    """Parse one ImageNet val XML annotation file."""
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    boxes = []
    for obj in root.findall('object'):
        b = obj.find('bndbox')
        boxes.append((
            int(b.find('xmin').text), int(b.find('ymin').text),
            int(b.find('xmax').text), int(b.find('ymax').text),
        ))
    synset = root.find('object/name').text if boxes else None
    return boxes, img_w, img_h, synset


def transform_bbox_to_clip(
    box: Tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    target_size: int = 224,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Apply CLIP preprocessing (Resize smaller edge -> 224, CenterCrop 224)
    to bbox coordinates. Returns the bbox in 224x224 image coordinates, or
    None if the bbox falls entirely outside the crop.
    """
    x0, y0, x1, y1 = box
    if img_w < img_h:
        scale = target_size / img_w
        new_w, new_h = target_size, int(round(img_h * scale))
    else:
        scale = target_size / img_h
        new_h, new_w = target_size, int(round(img_w * scale))
    x0, y0, x1, y1 = x0 * scale, y0 * scale, x1 * scale, y1 * scale

    crop_x = (new_w - target_size) / 2
    crop_y = (new_h - target_size) / 2
    x0 -= crop_x; x1 -= crop_x
    y0 -= crop_y; y1 -= crop_y

    x0 = max(0.0, min(target_size, x0))
    x1 = max(0.0, min(target_size, x1))
    y0 = max(0.0, min(target_size, y0))
    y1 = max(0.0, min(target_size, y1))
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return (x0, y0, x1, y1)


def boxes_to_patch_mask(
    boxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
    grid_size: int = 14,
    target_size: int = 224,
) -> torch.Tensor:
    """
    Convert bboxes (original image coords) into a fractional-coverage mask
    on the patch grid, accounting for CLIP preprocessing. Each grid cell =
    fraction of its 16x16 area covered by the bbox union after
    Resize+CenterCrop. Returns (grid_size, grid_size) in [0, 1].
    """
    pixel_mask = torch.zeros(target_size, target_size)
    for box in boxes:
        clip_box = transform_bbox_to_clip(box, img_w, img_h, target_size)
        if clip_box is None:
            continue
        x0, y0, x1, y1 = clip_box
        ix0, ix1 = int(np.floor(x0)), int(np.ceil(x1))
        iy0, iy1 = int(np.floor(y0)), int(np.ceil(y1))
        pixel_mask[iy0:iy1, ix0:ix1] = 1.0

    patch_size = target_size // grid_size  # 16 for 224/14
    grid = pixel_mask.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    return grid.mean(dim=(-1, -2))


def bbox_coverage_fraction(
    boxes: List[Tuple[int, int, int, int]],
    img_w: int,
    img_h: int,
) -> float:
    """Fraction of original image covered by the bbox union (pre-preprocessing)."""
    mask = np.zeros((img_h, img_w), dtype=bool)
    for (x0, y0, x1, y1) in boxes:
        mask[y0:y1, x0:x1] = True
    return float(mask.sum() / (img_h * img_w))


# =====================================================================
# Section 3. Image sampling (Score-CAM protocol)
# =====================================================================

def sample_eligible_images(
    image_dir: str,
    bbox_dir: str,
    n_samples: int = 500,
    max_coverage: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """
    Sample n_samples val images whose bbox union covers <= max_coverage of
    the original image. Returns metadata for each.
    """
    rng = random.Random(seed)
    all_xmls = sorted(Path(bbox_dir).glob('*.xml'))
    rng.shuffle(all_xmls)

    chosen = []
    for xml_path in all_xmls:
        if len(chosen) >= n_samples:
            break
        try:
            boxes, img_w, img_h, synset = parse_imagenet_bbox(xml_path)
        except Exception:
            continue
        if not boxes:
            continue
        cov = bbox_coverage_fraction(boxes, img_w, img_h)
        if cov > max_coverage:
            continue
        img_path = Path(image_dir) / (xml_path.stem + '.JPEG')
        if not img_path.exists():
            continue
        chosen.append({
            'image_path': str(img_path),
            'xml_path': str(xml_path),
            'synset': synset,
            'boxes': boxes,
            'img_w': img_w,
            'img_h': img_h,
            'coverage': cov,
        })

    print(
        f"Sampled {len(chosen)} images "
        f"(target={n_samples}, max_coverage={max_coverage}) "
        f"from {len(all_xmls)} annotation files."
    )
    return chosen


# =====================================================================
# Section 4. HG-PatchSAE heatmap engine and class names
# =====================================================================

def load_model(sae_device: str = "cpu"):
    """
    Load HGPatchSAE and return (model, class_names). The SAE is not used for
    EPG (only the heatmap extractors are), so sae_device defaults to cpu to
    keep GPU memory free; the gradient/attention methods still run on the GPU
    via the open_clip model inside HGPatchSAE.
    """
    from hgpatchsae import HGPatchSAE

    class_names = load_class_names()
    model = HGPatchSAE(class_names=class_names, sae_device=sae_device)
    return model, class_names


def load_class_names() -> List[str]:
    """Readable ImageNet-1k names in class-id order (tench=0, ...)."""
    path = os.path.join(_REPO_ROOT, "imagenet_classes.json")
    with open(path) as f:
        names = json.load(f)
    assert len(names) == 1000, f"expected 1000 names, got {len(names)}"
    return names


def build_synset_to_name(
    class_names: List[str],
    bbox_dir: str,
    class_mapping_file: Optional[str] = None,
) -> Dict[str, str]:
    """
    Map WordNet synset (e.g. 'n01440764') -> readable class name.

    Preferred: a standard imagenet_class_index.json of the form
        {"0": ["n01440764", "tench"], ...}
    Fallback: the ImageNet-1k convention that the 1000 synsets in ascending
    lexicographic order correspond to class ids 0..999. We collect synsets
    from bbox_dir, sort them, and zip with class_names (imagenet_classes.json).
    """
    if class_mapping_file and os.path.isfile(class_mapping_file):
        with open(class_mapping_file) as f:
            m = json.load(f)
        mapping = {syn: name.replace('_', ' ') for _, (syn, name) in m.items()}
        print(f"Loaded {len(mapping)} synset->name pairs from {class_mapping_file}")
        return mapping

    # Fallback: sorted synsets <-> class id order.
    synsets = set()
    for xml_path in Path(bbox_dir).glob('*.xml'):
        try:
            name = ET.parse(xml_path).getroot().find('object/name')
            if name is not None:
                synsets.add(name.text)
        except Exception:
            continue
    synsets = sorted(synsets)
    if len(synsets) != len(class_names):
        print(
            f"WARNING: found {len(synsets)} unique synsets in bbox_dir but "
            f"{len(class_names)} class names. The sorted-synset fallback assumes "
            f"all 1000 classes are present; labels may be misaligned. Provide "
            f"--class_mapping pointing to imagenet_class_index.json for a robust map."
        )
    mapping = {syn: class_names[i] for i, syn in enumerate(synsets)}
    print(f"Built {len(mapping)} synset->name pairs from sorted-synset convention.")
    return mapping


# =====================================================================
# Section 5. Heatmaps via HGPatchSAE  ->  (14, 14) grid
# =====================================================================
# Display name -> HGPatchSAE method key (HGPatchSAE.get_heatmap).
# 'Uniform' and 'Random' are handled specially (no model call).

METHOD_KEYS: Dict[str, str] = {
    'Raw Attention':           'raw_attention',
    'Attention Rollout':       'attention_rollout',
    'GradCAM':                 'gradcam',
    'LeGrad':                  'legrad',
    'Grad-SAM':                'grad_sam',
    'Grad Rollout':            'grad_rollout',
    'Transformer Attribution': 'transformer_attribution',
    'LayerCAM':                'layercam',
}

# Evaluation order: the paper's 7 heatmap methods + Random, Uniform as a sanity
# check, and Grad Rollout (evaluated but not reported in the paper).
METHODS: List[str] = [
    'Uniform',
    'Random',
    'Raw Attention',
    'Attention Rollout',
    'GradCAM',
    'LeGrad',
    'Grad-SAM',
    'Grad Rollout',
    'Transformer Attribution',
    'LayerCAM',
]


def pixel_heatmap_to_grid(hm: torch.Tensor, grid: int = 14) -> torch.Tensor:
    """
    Average-pool a pixel-space heatmap ([1,H,W] or [H,W]) down to (grid, grid),
    matching how boxes_to_patch_mask pools the bbox mask (mean over 16x16).
    """
    hm = hm.detach().float().cpu()
    while hm.dim() > 2:
        hm = hm.squeeze(0)
    hm = F.relu(hm)
    pooled = F.adaptive_avg_pool2d(hm[None, None], (grid, grid))
    return pooled.squeeze(0).squeeze(0)


def compute_grid_heatmap(
    model,
    display_name: str,
    pil_image: Image.Image,
    class_text: str,
    grid: int = 14,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Return a (grid, grid) non-negative heatmap for one image and method."""
    if display_name == 'Uniform':
        g = torch.ones(grid, grid)
        return g / g.sum()
    if display_name == 'Random':
        r = (rng.random((grid, grid)).astype(np.float32)
             if rng is not None else np.random.rand(grid, grid).astype(np.float32))
        g = torch.from_numpy(r)
        return g / g.sum()

    key = METHOD_KEYS[display_name]
    hm = model.get_heatmap(pil_image, text_prompt=class_text, method=key)
    return pixel_heatmap_to_grid(hm, grid)


# =====================================================================
# Section 6. Main evaluation loop
# =====================================================================

def evaluate(
    image_dir: str,
    bbox_dir: str,
    class_mapping_file: Optional[str] = None,
    output_path: str = 'outputs/epg_iou_results.json',
    n_samples: int = 500,
    iou_threshold: float = 0.5,
    bbox_binarize_threshold: float = 0.5,
    sae_device: str = 'cpu',
    seed: int = 42,
    methods: Optional[List[str]] = None,
) -> Dict:
    """Run EPG + IoU for the HG-PatchSAE heatmap methods on sampled val images."""
    methods = methods or METHODS

    # Uniform / Random heatmaps are synthetic (no model call), so skip the
    # model load entirely when only those are requested.
    needs_model = any(m not in ('Uniform', 'Random') for m in methods)
    if needs_model:
        print("Loading HG-PatchSAE...")
        model, class_names = load_model(sae_device=sae_device)
        print("Ready.")
    else:
        print("Only Uniform/Random requested — skipping model load.")
        model, class_names = None, load_class_names()

    synset2name = build_synset_to_name(class_names, bbox_dir, class_mapping_file)
    images = sample_eligible_images(image_dir, bbox_dir, n_samples, seed=seed)
    rng = np.random.default_rng(seed)

    results: Dict[str, Dict[str, List[float]]] = {
        name: {'epg': [], 'iou': []} for name in methods
    }

    # Sanity check: Uniform EPG over these images should be ~ the mean bbox
    # coverage on the 224x224 patch grid.
    expected_uniform_epg = float(np.mean([item['coverage'] for item in images])) \
        if images else 0.0
    print(f"Expected uniform EPG (mean bbox coverage): {expected_uniform_epg:.3f}")

    for item in tqdm(images, desc="EPG/IoU"):
        try:
            img = Image.open(item['image_path']).convert('RGB')
        except Exception:
            continue

        bbox_frac = boxes_to_patch_mask(item['boxes'], item['img_w'], item['img_h'])
        bbox_binary = (bbox_frac >= bbox_binarize_threshold).float()
        if bbox_binary.sum() == 0:
            continue  # bbox vanished under preprocessing

        class_text = synset2name.get(item['synset'], item['synset'])

        for name in methods:
            try:
                hm = compute_grid_heatmap(model, name, img, class_text, rng=rng).float()
                assert hm.shape == (14, 14), f"{name} returned shape {hm.shape}"
                results[name]['epg'].append(compute_epg(hm, bbox_frac))
                results[name]['iou'].append(
                    compute_iou(hm, bbox_binary, threshold=iou_threshold)
                )
            except Exception as e:
                print(f"[{name}] error on {item['image_path']}: {e}")

    summary: Dict = {
        '_metadata': {
            'n_images': len(images),
            'expected_uniform_epg': expected_uniform_epg,
            'iou_threshold': iou_threshold,
            'bbox_binarize_threshold': bbox_binarize_threshold,
            'seed': seed,
            'backbone': 'openai/clip-vit-base-patch16 (open_clip)',
        }
    }
    for name, sc in results.items():
        if not sc['epg']:
            continue
        summary[name] = {
            'epg_mean': float(np.mean(sc['epg'])),
            'epg_std':  float(np.std(sc['epg'])),
            'iou_mean': float(np.mean(sc['iou'])),
            'iou_std':  float(np.std(sc['iou'])),
            'n':         len(sc['epg']),
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"{'Method':<28} {'EPG (mean ± std)':<22} {'IoU (mean ± std)':<22} N")
    print("-" * 84)
    for name in methods:
        if name not in summary:
            continue
        s = summary[name]
        print(
            f"{name:<28} "
            f"{s['epg_mean']:.3f} ± {s['epg_std']:.3f}        "
            f"{s['iou_mean']:.3f} ± {s['iou_std']:.3f}        "
            f"{s['n']}"
        )
    print()
    print(
        f"Sanity check: Uniform EPG should be approximately "
        f"{expected_uniform_epg:.3f} (mean bbox coverage on the 224x224 patch grid). "
        f"Large deviation indicates a bbox-to-patch transform bug."
    )
    print(f"Saved: {output_path}")
    return summary


# =====================================================================
# Section 7. CLI
# =====================================================================

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', required=True,
                   help='ImageNet val image directory (.JPEG files)')
    p.add_argument('--bbox_dir', required=True,
                   help='ImageNet val bbox directory (.xml files)')
    p.add_argument('--class_mapping', default=None,
                   help='Optional imagenet_class_index.json (synset->name). '
                        'If omitted, falls back to the sorted-synset convention '
                        'with imagenet_classes.json.')
    p.add_argument('--methods', nargs='+', default=None,
                   help='Subset of methods to evaluate (default = all). '
                        'e.g. --methods Uniform Random')
    p.add_argument('--n_samples', type=int, default=500)
    p.add_argument('--iou_threshold', type=float, default=0.5)
    p.add_argument('--bbox_binarize_threshold', type=float, default=0.5)
    p.add_argument('--output', default='outputs/epg_iou_results.json')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sae_device', default='cpu',
                   help="Device for the (unused) SAE in HGPatchSAE. Heatmaps run on "
                        "GPU regardless. Default cpu to save VRAM.")
    args = p.parse_args()

    evaluate(
        image_dir=args.image_dir,
        bbox_dir=args.bbox_dir,
        class_mapping_file=args.class_mapping,
        output_path=args.output,
        n_samples=args.n_samples,
        iou_threshold=args.iou_threshold,
        bbox_binarize_threshold=args.bbox_binarize_threshold,
        sae_device=args.sae_device,
        seed=args.seed,
        methods=args.methods,
    )