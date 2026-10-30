"""
build_prototypes.py
===================
Build HG-PatchSAE class prototypes on ImageNet-1k (Sec. 3-4.4 of the paper).

For every training image the PatchSAE latents Z are computed once, every
requested heatmap method produces patch weights w, and the pooled vector
f = w^T Z is accumulated into a per-class prototype (Eq. 4). The uniform
(w_p = 1/P) and random baselines are produced in the same pass.

Outputs under --output_root:
    prototypes/<method>/cls_hgpatchsae_<tag>.npy  HG-PatchSAE prototype, [1000, 49152]
    prototypes/<method>/cls_patchsae_<tag>.npy    uniform-pooling prototype
                                                  (identical in every method dir)
    probe_features/, topk_indices/                optional per-image artefacts

Multi-GPU: split the classes into ranges, one process per GPU, then merge:

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python experiments/build_prototypes.py \
        --class_start 0   --class_end 500  --output_root outputs/imagenet_130/part0 ...
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python experiments/build_prototypes.py \
        --class_start 500 --class_end 1000 --output_root outputs/imagenet_130/part1 ...

    PYTHONPATH=.:experiments python -c "from build_prototypes import merge_partitions; \
        merge_partitions(['outputs/imagenet_130/part0', 'outputs/imagenet_130/part1'], \
                         'outputs/imagenet_130')"

Paper setting (130 training images per class, tau = 0.2):

    PYTHONPATH=. python experiments/build_prototypes.py \
        --max_per_class 130 \
        --methods legrad grad_sam attention_rollout gradcam raw_attention \
                  transformer_attribution layercam random \
        --activation_mode thresholded_real --sae_threshold 0.2 \
        --output_root outputs/imagenet_130
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hgpatchsae import (  # noqa: E402
    HGPatchSAE,
    ALL_METHODS as _BUILTIN_METHODS,
    ACTIVATION_MODES,
    DEFAULT_SAE_THRESHOLD,
    load_imagenet_class_names,
)

_ALL_CHOICES = _BUILTIN_METHODS + ["random"]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Per-class sizing
    p.add_argument("--max_per_class",    type=int, default=-1,
                   help="Train images per class. -1 = use ALL images for each class.")
    p.add_argument("--class_batch_size", type=int, default=50,
                   help="Classes processed at a time (controls peak RAM).")

    # Multi-GPU: class slicing
    p.add_argument("--class_start",      type=int, default=0,
                   help="First class index (inclusive). Use for multi-GPU splitting.")
    p.add_argument("--class_end",        type=int, default=1000,
                   help="Last class index (exclusive).")

    # Methods + activation framework
    p.add_argument("--methods", nargs="+",
                   default=["legrad", "grad_sam", "attention_rollout", "gradcam",
                            "raw_attention", "transformer_attribution", "random"],
                   choices=_ALL_CHOICES)
    p.add_argument("--activation_mode", type=str, default="thresholded_real",
                   choices=sorted(ACTIVATION_MODES),
                   help="How to process raw SAE outputs before weighting.")
    p.add_argument("--sae_threshold",   type=float, default=DEFAULT_SAE_THRESHOLD,
                   help="τ for binary / thresholded_real modes.")
    p.add_argument("--cls_mode", type=str, default="off",
                   choices=["off", "on", "both"],
                   help="CLS-token handling. off = patch tokens only (paper); "
                        "on = add the CLS-token SAE vector; both = write both "
                        "variants (prototypes/ and prototypes_cls/) in one pass.")
    p.add_argument("--cls_weight",      type=float, default=1.0,
                   help="Scalar multiplier on the CLS vector when CLS is active.")

    # Output
    p.add_argument("--output_root", type=str, default="outputs/imagenet_130")

    # Per-image artefacts (OFF by default — only prototypes are saved)
    p.add_argument("--save_probe_features", action="store_true",
                   help="Save per-image features projected through R "
                        "(shape [N, proj_dim]) for downstream linear probes. "
                        "Disabled by default — set this only if you need them.")
    p.add_argument("--proj_dim",    type=int, default=1024,
                   help="Projection dim used ONLY when --save_probe_features is on.")
    p.add_argument("--proj_seed",   type=int, default=42,
                   help="Seed for the random projection (only used when "
                        "--save_probe_features is on).")
    p.add_argument("--save_topk_indices", action="store_true",
                   help="Save per-image top-K SAE indices (for Jaccard scoring). "
                        "Disabled by default.")
    p.add_argument("--topk_k",      type=int, default=50,
                   help="K for top-K indices (only used when --save_topk_indices is on).")

    # Checkpointing — periodic prototype dump, resumable
    p.add_argument("--checkpoint_interval", type=int, default=50,
                   help="Save partial prototypes every N successfully processed "
                        "images, AND save state so the run is resumable on crash. "
                        "Note: checkpoint I/O is ~4 GB per save in 'both' mode "
                        "with 10 methods; raise this number (e.g. 1000) if disk "
                        "throughput becomes the bottleneck.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="On startup, if a checkpoint state exists in "
                        "<output_root>/_state/, resume from it (skipping already-"
                        "done images per class). Use --no-resume to start fresh.")

    # Optional sparse dump for calibration set
    p.add_argument("--save_sparse_subset", type=int, default=0,
                   help="Save raw per-patch SAE acts (sparse) for the first N "
                        "images of each class. 0 = disabled. 5 ≈ 6 GB total.")

    # Per-layer GradCAM
    p.add_argument("--gradcam_all_layers", action="store_true",
                   help="Compute GradCAM at every ViT-B-16 transformer layer (12 total) "
                        "in one extra backward pass per image. Results saved separately as "
                        "gradcam_L00 … gradcam_L11. Combines with all --cls_mode variants. "
                        "Regular 'gradcam' (default layer -4) is kept unless you remove it "
                        "from --methods.")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Output directory layout
# ─────────────────────────────────────────────────────────────────────────────

def variants_from_mode(mode: str) -> List[Dict[str, str]]:
    """
    Returns the list of output variants implied by --cls_mode.

    Each variant is a dict with:
      name        : "patch" or "cls"   — human label
      key_suffix  : ""        or "_cls" — suffix to look up entries in the
                                          dict returned by run_all_methods
      path_suffix : ""        or "_cls" — suffix on output sub-directories
    """
    v: List[Dict[str, str]] = []
    if mode in ("off", "both"):
        v.append({"name": "patch", "key_suffix": "",
                  "path_suffix": ""})
    if mode in ("on", "both"):
        # In "both" mode run_all_methods uses "_cls"-suffixed keys; in "on"
        # mode it uses bare keys (no suffix), so the suffix differs here.
        v.append({"name": "cls",
                  "key_suffix": "_cls" if mode == "both" else "",
                  "path_suffix": "_cls"})
    return v


def make_dirs(args: argparse.Namespace) -> Dict[str, Dict[str, str]]:
    """
    Returns nested paths: {variant_name: {"proto" [+ optional "probe"/"jaccard"]: path}}.
    Sparse-subset dir under "_sparse", checkpoint state dir under "_state".
    """
    out      = args.output_root
    variants = variants_from_mode(args.cls_mode)
    paths: Dict[str, Dict[str, str]] = {}

    for v in variants:
        sfx = v["path_suffix"]
        sub: Dict[str, str] = {
            "proto": os.path.join(out, f"prototypes{sfx}"),
        }
        if args.save_probe_features:
            sub["probe"] = os.path.join(out, f"probe_features{sfx}")
        if args.save_topk_indices:
            sub["jaccard"] = os.path.join(out, f"topk_indices{sfx}")
        for d in sub.values():
            os.makedirs(d, exist_ok=True)
        for m in args.methods:
            os.makedirs(os.path.join(sub["proto"], m), exist_ok=True)
        paths[v["name"]] = sub

    if args.save_sparse_subset > 0:
        sparse_path = os.path.join(out, "sparse_subset")
        os.makedirs(sparse_path, exist_ok=True)
        paths["_sparse"] = {"sparse": sparse_path}

    state_path = os.path.join(out, "_state")
    os.makedirs(state_path, exist_ok=True)
    paths["_state"] = {"state": state_path}

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint store — partial prototypes + resume state
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointStore:
    """
    Persists the running prototype accumulators every N images so the run is:
      (a) resumable on crash (raw accumulators saved to <state>/)
      (b) inspectable mid-run (normalised prototypes overwritten in their
          final location after every checkpoint)
    """

    def __init__(self,
                 state_dir: str,
                 proto_paths_by_variant: Dict[str, str],
                 method_names: List[str],
                 variant_names: List[str],
                 num_classes: int,
                 sae_dim: int,
                 class_range_tag: str):
        self.state_dir          = state_dir
        self.proto_paths        = proto_paths_by_variant
        self.method_names       = method_names
        self.variant_names      = variant_names
        self.num_classes        = num_classes
        self.sae_dim            = sae_dim
        self.tag                = class_range_tag
        os.makedirs(state_dir, exist_ok=True)

    # ── Paths ─────────────────────────────────────────────────────
    def _acc_path(self, variant: str, method_or_uniform: str) -> str:
        return os.path.join(self.state_dir, f"acc_{variant}_{method_or_uniform}.npy")

    def _count_path(self) -> str:
        return os.path.join(self.state_dir, "class_success_count.npy")

    def _done_path(self) -> str:
        return os.path.join(self.state_dir, "class_done_count.npy")

    def _progress_path(self) -> str:
        return os.path.join(self.state_dir, "progress.json")

    # ── Save ──────────────────────────────────────────────────────
    @staticmethod
    def _atomic_save(path: str, arr: np.ndarray) -> None:
        # numpy auto-appends '.npy' if the path doesn't already end with it,
        # so we have to keep the extension on the tmp file too — otherwise
        # 'x.npy.tmp' becomes 'x.npy.tmp.npy' on disk and os.replace fails.
        base, ext = os.path.splitext(path)
        tmp = base + ".tmp" + ext
        np.save(tmp, arr)
        os.replace(tmp, path)

    def save(self,
             cls_uniform: Dict[str, np.ndarray],
             cls_hg:    Dict[str, Dict[str, np.ndarray]],
             class_success_count: np.ndarray,
             class_done_count: np.ndarray,
             total_processed: int,
             write_normalised: bool = True) -> None:
        """
        Persist current state to <state>/ AND (optionally) compute and write
        normalised prototypes to their final locations.

        cls_uniform : {variant -> [C, sae_dim] running sum for the uniform key}
        cls_hg    : {variant -> {method -> [C, sae_dim] running sum}}
        """
        # 1. Raw accumulators (for resume).
        for v in self.variant_names:
            self._atomic_save(self._acc_path(v, "uniform"), cls_uniform[v])
            for m in self.method_names:
                self._atomic_save(self._acc_path(v, m), cls_hg[v][m])
        self._atomic_save(self._count_path(), class_success_count)
        self._atomic_save(self._done_path(),  class_done_count)

        # 2. Progress JSON.
        prog_tmp = self._progress_path() + ".tmp"
        with open(prog_tmp, "w") as f:
            json.dump({
                "total_processed":  int(total_processed),
                "variants":         self.variant_names,
                "methods":          self.method_names,
                "tag":              self.tag,
            }, f, indent=2)
        os.replace(prog_tmp, self._progress_path())

        # 3. Normalised prototypes (for users to consume mid-run).
        if write_normalised:
            safe = np.maximum(class_success_count, 1.0)[:, None]
            for v in self.variant_names:
                # Uniform / PatchSAE
                proto_uni = cls_uniform[v] / safe
                rs        = proto_uni.sum(axis=1, keepdims=True)
                proto_uni = proto_uni / np.where(rs > 1e-8, rs, 1.0)
                # Uniform pooling also gets its own "method" dir, which the FF
                # and NCC evaluations read like every heatmap method.
                uni_dir = os.path.join(self.proto_paths[v], "uniform")
                os.makedirs(uni_dir, exist_ok=True)
                for fname in (f"cls_patchsae_{self.tag}.npy", f"cls_hgpatchsae_{self.tag}.npy"):
                    self._atomic_save(os.path.join(uni_dir, fname),
                                      proto_uni.astype(np.float32, copy=False))
                # Method prototypes
                for m in self.method_names:
                    hg_arr = cls_hg[v][m] / safe
                    rs       = hg_arr.sum(axis=1, keepdims=True)
                    hg_arr = hg_arr / np.where(rs > 1e-8, rs, 1.0)
                    method_dir = os.path.join(self.proto_paths[v], m)
                    self._atomic_save(
                        os.path.join(method_dir, f"cls_patchsae_{self.tag}.npy"),
                        proto_uni.astype(np.float32, copy=False))
                    self._atomic_save(
                        os.path.join(method_dir, f"cls_hgpatchsae_{self.tag}.npy"),
                        hg_arr.astype(np.float32, copy=False))

    # ── Load ──────────────────────────────────────────────────────
    def exists(self) -> bool:
        return os.path.exists(self._progress_path())

    def load(self) -> Tuple[Dict[str, np.ndarray],
                            Dict[str, Dict[str, np.ndarray]],
                            np.ndarray, np.ndarray, int]:
        cls_uniform: Dict[str, np.ndarray] = {}
        cls_hg:    Dict[str, Dict[str, np.ndarray]] = {}
        for v in self.variant_names:
            cls_uniform[v] = np.load(self._acc_path(v, "uniform"))
            cls_hg[v]    = {m: np.load(self._acc_path(v, m)) for m in self.method_names}
        class_success = np.load(self._count_path())
        class_done    = np.load(self._done_path())
        with open(self._progress_path()) as f:
            prog = json.load(f)
        return cls_uniform, cls_hg, class_success, class_done, int(prog["total_processed"])


# ─────────────────────────────────────────────────────────────────────────────
# Random projection matrix (shared across partitions via the same seed)
# ─────────────────────────────────────────────────────────────────────────────

def get_projection_matrix(out_root: str, sae_dim: int, proj_dim: int, seed: int) -> np.ndarray:
    # Always place the projection at the ROOT of out_root (above partitions)
    # so it's shared across parallel runs.
    parent = os.path.dirname(out_root.rstrip("/")) or out_root
    proj_path = os.path.join(parent, "proj_matrix.npy")

    if os.path.exists(proj_path):
        print(f"Loading random projection from {proj_path}")
        R = np.load(proj_path)
        assert R.shape == (sae_dim, proj_dim), \
            f"proj_matrix shape mismatch: got {R.shape}, expected {(sae_dim, proj_dim)}"
    else:
        print(f"Generating random projection ({sae_dim} → {proj_dim})...")
        rng = np.random.RandomState(seed)
        R = (rng.randn(sae_dim, proj_dim) / np.sqrt(proj_dim)).astype(np.float32)
        np.save(proj_path, R)
        print(f"  Saved to {proj_path}")
    return R


# ─────────────────────────────────────────────────────────────────────────────
# Memmap helpers
# ─────────────────────────────────────────────────────────────────────────────

def alloc_memmap(path: str, n: int, d: int, dtype) -> None:
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(n, d))
    mm.flush()
    del mm


# ─────────────────────────────────────────────────────────────────────────────
# Sparse subset writer
# ─────────────────────────────────────────────────────────────────────────────

class SparseSubsetWriter:
    """
    Appends sparse SAE patch activations to a single CSR-on-disk file
    using scipy.sparse.save_npz. We accumulate in memory by class and
    flush once per class to keep peak RAM low.
    """

    def __init__(self, root: str, sae_dim: int):
        self.root    = root
        self.sae_dim = sae_dim
        os.makedirs(root, exist_ok=True)

    def write_class(self, cls_id: int, rows: List[np.ndarray]):
        """rows: list of [196, sae_dim] dense float32 arrays."""
        if not rows:
            return
        try:
            from scipy import sparse
        except ImportError:
            warnings_once("scipy not available — sparse subset disabled.")
            return
        # Stack to [196 * n_images, sae_dim] then convert to CSR.
        block = np.concatenate(rows, axis=0)
        csr = sparse.csr_matrix(block)
        out = os.path.join(self.root, f"cls_{cls_id:04d}.npz")
        sparse.save_npz(out, csr)


_warned: set = set()
def warnings_once(msg: str):
    if msg in _warned:
        return
    _warned.add(msg)
    print(f"  ⚠ {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Image loading for a batch of classes
# ─────────────────────────────────────────────────────────────────────────────

def load_class_images(ds, idx_map: Dict[int, List[int]]) -> Dict[int, list]:
    """Returns {cls_id: [PIL.Image, ...]} for the requested HF row indices."""
    flat = []
    for cls_id in sorted(idx_map):
        flat.extend(idx_map[cls_id])
    if not flat:
        return {}
    subset = ds.select(flat)
    out = defaultdict(list)
    for item in subset:
        img = item["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        out[item["label"]].append(img)
    return dict(out)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    SAE_DIM          = 49152
    NUM_CLASSES      = 1000
    K                = args.topk_k
    PROJ_DIM         = args.proj_dim
    GRADCAM_N_LAYERS = 12   # ViT-B-16 transformer blocks

    # Expand args.methods with per-layer GradCAM keys BEFORE make_dirs so that
    # output subdirectories are created for each layer.
    gradcam_layer_keys: List[str] = []
    if args.gradcam_all_layers:
        gradcam_layer_keys = [f"gradcam_L{i:02d}" for i in range(GRADCAM_N_LAYERS)]
        for key in gradcam_layer_keys:
            if key not in args.methods:
                args.methods.append(key)

    paths     = make_dirs(args)
    variants  = variants_from_mode(args.cls_mode)          # active variants
    variant_names = [v["name"] for v in variants]
    print(f"\nOutput root:        {args.output_root}")
    print(f"Class range:        [{args.class_start}, {args.class_end})")
    print(f"Activation mode:    {args.activation_mode}  (τ={args.sae_threshold})")
    print(f"CLS mode:           {args.cls_mode}  (variants written: {variant_names})")
    print(f"Methods:            {args.methods}")
    print()

    # ── Class names ───────────────────────────────────────────────
    readable_names = load_imagenet_class_names()

    # ── Label-only HF load ────────────────────────────────────────
    print("Loading HF label column (no images yet)...")
    from datasets import load_dataset
    ds = load_dataset("evanarlian/imagenet_1k_resized_256", split="train", trust_remote_code=True)
    labels_col = ds["label"]

    c2i = defaultdict(list)
    for idx, lbl in enumerate(labels_col):
        c2i[lbl].append(idx)

    # ── Per-class index lists (training split only) ───────────────
    class_range = range(args.class_start, args.class_end)
    if args.max_per_class is None or args.max_per_class < 0:
        train_idx_by_class = {c: list(c2i[c]) for c in class_range}        # all images
    else:
        train_idx_by_class = {c: c2i[c][:args.max_per_class] for c in class_range}

    N_train_est = sum(len(v) for v in train_idx_by_class.values())
    print(f"  Train images (this partition): {N_train_est:,}")

    # ── Init HG-PatchSAE ──────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # TF32 roughly doubles matmul throughput on Ampere and newer GPUs.
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True
        torch.backends.cudnn.benchmark        = True

    hg = HGPatchSAE(class_names=readable_names)
    print("HG-PatchSAE ready.\n")
    if gradcam_layer_keys:
        print(f"GradCAM all-layers: {gradcam_layer_keys[0]} … {gradcam_layer_keys[-1]}")

    # ── Accumulators (one set per variant) ────────────────────────
    ALL_KEYS = ["uniform"] + args.methods

    cls_patchsae = {v: np.zeros((NUM_CLASSES, SAE_DIM), dtype=np.float32) for v in variant_names}
    cls_hg     = {v: {m: np.zeros((NUM_CLASSES, SAE_DIM), dtype=np.float32) for m in args.methods}
                    for v in variant_names}
    class_success_count = np.zeros(NUM_CLASSES, dtype=np.float32)
    class_done_count    = np.zeros(NUM_CLASSES, dtype=np.int64)   # attempts (success + fail)

    # ── Random projection — ONLY when probe features are requested ──
    R: Optional[np.ndarray] = None
    if args.save_probe_features:
        R = get_projection_matrix(args.output_root, SAE_DIM, PROJ_DIM, args.proj_seed)

    # ── Optional per-image memmaps (probe features + top-K) ───────
    probe_tr_paths: Dict[str, Dict[str, str]] = {v: {} for v in variant_names}
    topk_tr_paths:  Dict[str, Dict[str, str]] = {v: {} for v in variant_names}
    y_tr_paths:     Dict[str, str]            = {}

    if args.save_probe_features:
        for v in variant_names:
            for k in ALL_KEYS:
                probe_tr_paths[v][k] = os.path.join(paths[v]["probe"], f"X_{k}_tr.npy")
            y_tr_paths[v] = os.path.join(paths[v]["probe"], "y_tr.npy")
        for v in variant_names:
            for k in ALL_KEYS:
                if not os.path.exists(probe_tr_paths[v][k]):
                    alloc_memmap(probe_tr_paths[v][k], N_train_est, PROJ_DIM, np.float32)
        probe_tr_mm = {
            v: {k: np.load(probe_tr_paths[v][k], mmap_mode="r+") for k in ALL_KEYS}
            for v in variant_names
        }
    else:
        probe_tr_mm = None

    if args.save_topk_indices:
        for v in variant_names:
            for k in ALL_KEYS:
                topk_tr_paths[v][k] = os.path.join(paths[v]["jaccard"], f"topk_{k}_tr.npy")
        for v in variant_names:
            for k in ALL_KEYS:
                if not os.path.exists(topk_tr_paths[v][k]):
                    alloc_memmap(topk_tr_paths[v][k], N_train_est, K, np.int32)
        topk_tr_mm = {
            v: {k: np.load(topk_tr_paths[v][k], mmap_mode="r+") for k in ALL_KEYS}
            for v in variant_names
        }
    else:
        topk_tr_mm = None

    y_tr_buf = np.full(N_train_est, -1, dtype=np.int32)

    sparse_path = paths.get("_sparse", {}).get("sparse")
    sparse_writer = (SparseSubsetWriter(sparse_path, SAE_DIM)
                     if (sparse_path is not None and args.save_sparse_subset > 0) else None)

    # ── Checkpoint store ──────────────────────────────────────────
    tag = f"{args.max_per_class}_train_ImageNet_cls{args.class_start}-{args.class_end}"
    ckpt = CheckpointStore(
        state_dir=paths["_state"]["state"],
        proto_paths_by_variant={v: paths[v]["proto"] for v in variant_names},
        method_names=args.methods,
        variant_names=variant_names,
        num_classes=NUM_CLASSES,
        sae_dim=SAE_DIM,
        class_range_tag=tag,
    )

    # ── Resume if a previous checkpoint exists ────────────────────
    total_processed = 0
    tr_row = 0
    if args.resume and ckpt.exists():
        try:
            cls_uniform_loaded, cls_hg_loaded, class_success_loaded, \
                class_done_loaded, total_processed = ckpt.load()
            # Validate shapes match current config.
            ok = (set(cls_uniform_loaded.keys()) == set(variant_names) and
                  all(set(cls_hg_loaded[v].keys()) >= set(args.methods)
                      for v in variant_names))
            if not ok:
                raise RuntimeError("checkpoint variant/method mismatch")
            for v in variant_names:
                cls_patchsae[v]     = cls_uniform_loaded[v]
                for m in args.methods:
                    cls_hg[v][m] = cls_hg_loaded[v][m]
            class_success_count = class_success_loaded.astype(np.float32, copy=False)
            class_done_count    = class_done_loaded.astype(np.int64,   copy=False)
            print(f"✓ Resumed from checkpoint — {total_processed:,} images already done; "
                  f"{int(class_done_count.sum()):,} attempts logged.")
        except Exception as e:
            print(f"⚠ Could not resume checkpoint ({e}); starting fresh.")
            total_processed = 0
            cls_patchsae = {v: np.zeros((NUM_CLASSES, SAE_DIM), dtype=np.float32) for v in variant_names}
            cls_hg     = {v: {m: np.zeros((NUM_CLASSES, SAE_DIM), dtype=np.float32)
                                for m in args.methods}
                            for v in variant_names}
            class_success_count = np.zeros(NUM_CLASSES, dtype=np.float32)
            class_done_count    = np.zeros(NUM_CLASSES, dtype=np.int64)
    elif ckpt.exists() and not args.resume:
        print("⚠ Checkpoint exists but --no-resume requested. Starting fresh "
              "(checkpoint will be OVERWRITTEN on first save).")

    total_errors = 0
    t0           = time.time()
    images_since_save = 0
    ckpt_interval     = max(1, int(args.checkpoint_interval))

    # ═══════════════════════════════════════════════════════════════
    #  TRAINING PASS  (no val — uses every image in train_idx_by_class)
    # ═══════════════════════════════════════════════════════════════
    print(f"{'='*60}")
    print(f"  TRAINING  ({'all' if args.max_per_class < 0 else args.max_per_class}/class "
          f"× {len(class_range)} classes)")
    print(f"  Class batch size: {args.class_batch_size}   "
          f"Checkpoint every: {ckpt_interval} images")
    print(f"{'='*60}\n")

    batch_starts = list(range(args.class_start, args.class_end, args.class_batch_size))

    for batch_start in tqdm(batch_starts, desc="Class-batches"):
        batch = list(range(batch_start, min(batch_start + args.class_batch_size, args.class_end)))

        # Skip indices already-done in each class (for resume).
        remaining_idx_by_class = {}
        for c in batch:
            done   = int(class_done_count[c])
            remain = train_idx_by_class.get(c, [])[done:]
            if remain:
                remaining_idx_by_class[c] = remain

        if not remaining_idx_by_class:
            continue        # entire batch already processed

        class_images = load_class_images(ds, remaining_idx_by_class)

        for cls_id in batch:
            true_class = readable_names[cls_id]
            images     = class_images.get(cls_id, [])

            sparse_rows: List[np.ndarray] = []

            for img_i_in_remain, img in enumerate(images):
                # img_i_in_remain is index within the REMAINING list; absolute
                # index within the class is done + img_i_in_remain.
                already_done = int(class_done_count[cls_id])

                try:
                    # Exclude gradcam_L* from the standard pipeline — they are
                    # computed separately via gradcam_all_layers_run below.
                    std_methods = [m for m in args.methods if not m.startswith("gradcam_L")]
                    results = hg.run_all_methods(
                        image=img,
                        true_class=true_class,
                        methods=std_methods,
                        sae_threshold=args.sae_threshold,
                        activation_mode=args.activation_mode,
                        include_cls=args.cls_mode,         # "off" | "on" | "both"
                        cls_weight=args.cls_weight,
                    )

                    # ── Per-layer GradCAM (one extra backward pass) ───────
                    if gradcam_layer_keys:
                        # patch_acts still cached from register_image inside run_all_methods
                        _pa_raw = torch.from_numpy(hg.get_patch_sae_activations())
                        _act    = HGPatchSAE._apply_activation_mode(
                            _pa_raw, args.activation_mode, args.sae_threshold)
                        _uni_np = _act.sum(dim=0).numpy().astype(np.float32)

                        # CLS vector (only when a cls variant is needed)
                        _want_cls   = args.cls_mode in ("on", "both")
                        _want_patch = args.cls_mode in ("off", "both")
                        _dual       = args.cls_mode == "both"
                        _cls_np: np.ndarray | None = None
                        if _want_cls:
                            _cls_arr = hg.get_cls_sae_activation()
                            if _cls_arr is not None:
                                _cls_t  = torch.from_numpy(_cls_arr).float()
                                _cls_t  = HGPatchSAE._apply_activation_mode(
                                    _cls_t.unsqueeze(0), args.activation_mode,
                                    args.sae_threshold).squeeze(0)
                                _cls_np = (args.cls_weight * _cls_t).numpy().astype(np.float32)
                            else:
                                _cls_np = np.zeros(_uni_np.shape[0], dtype=np.float32)

                        # Single backward — all 12 GradCAM layers at once
                        _all_hm = hg.gradcam_all_layers_run(
                            img, text_prompt=true_class, threshold=0.0)

                        for _li in range(GRADCAM_N_LAYERS):
                            _key = f"gradcam_L{_li:02d}"
                            _hm  = _all_hm.get(_li)

                            if _hm is None:
                                _vec = _uni_np.copy()
                            else:
                                _idx_info = hg.heatmap_to_vit_token_indices(_hm, patch_size=16)
                                _pidx     = _idx_info.get("patch_indices", [])
                                _wts      = np.array(
                                    _idx_info.get("patch_mean_values", []), dtype=np.float32)
                                if len(_pidx) == 0 or len(_pidx) != len(_wts):
                                    _vec = _uni_np.copy()
                                else:
                                    _w_t = torch.from_numpy(_wts).unsqueeze(1)
                                    _vec = (_act[_pidx] * _w_t).sum(dim=0).numpy().astype(np.float32)

                            # Mirrors run_all_methods' _emit: L1-norm + CLS variants
                            if _want_patch:
                                results[_key] = HGPatchSAE._l1_norm(_vec)
                            if _want_cls:
                                _v_cls = _vec + _cls_np
                                _cls_key = f"{_key}_cls" if _dual else _key
                                results[_cls_key] = HGPatchSAE._l1_norm(_v_cls)

                    for v in variants:
                        vn, ks = v["name"], v["key_suffix"]
                        uni_v  = results[f"uniform{ks}"]
                        cls_patchsae[vn][cls_id] += uni_v
                        for m in args.methods:
                            cls_hg[vn][m][cls_id] += results.get(f"{m}{ks}", uni_v)

                        if probe_tr_mm is not None or topk_tr_mm is not None:
                            for k in ALL_KEYS:
                                feat = results.get(f"{k}{ks}", uni_v)
                                if probe_tr_mm is not None:
                                    probe_tr_mm[vn][k][tr_row] = feat @ R
                                if topk_tr_mm is not None:
                                    idx = np.argpartition(feat, -K)[-K:]
                                    topk_tr_mm[vn][k][tr_row] = idx.astype(np.int32)

                    class_success_count[cls_id] += 1.0
                    if probe_tr_mm is not None or topk_tr_mm is not None:
                        y_tr_buf[tr_row] = cls_id
                        tr_row += 1

                    # Sparse calibration dump (first N images per class)
                    if sparse_writer is not None and (already_done + img_i_in_remain) < args.save_sparse_subset:
                        sparse_rows.append(hg.get_patch_sae_activations())

                except Exception as e:
                    total_errors += 1
                    if total_errors <= 5:
                        import traceback
                        tqdm.write(f"  Error [cls {cls_id} img {img_i_in_remain}]: {e}")
                        tqdm.write(traceback.format_exc())
                    elif total_errors == 6:
                        tqdm.write("  (suppressing further errors)")

                # Count this image as attempted regardless of success/failure
                # — keeps resume index correct.
                class_done_count[cls_id] += 1
                total_processed          += 1
                images_since_save        += 1

                # ── Periodic checkpoint ──────────────────────────
                if images_since_save >= ckpt_interval:
                    if probe_tr_mm is not None:
                        for vn in variant_names:
                            for mm in probe_tr_mm[vn].values(): mm.flush()
                    if topk_tr_mm is not None:
                        for vn in variant_names:
                            for mm in topk_tr_mm[vn].values():  mm.flush()
                    ckpt.save(cls_patchsae, cls_hg, class_success_count,
                              class_done_count, total_processed,
                              write_normalised=True)
                    images_since_save = 0

            if sparse_writer is not None and sparse_rows:
                sparse_writer.write_class(cls_id, sparse_rows)

            del images, sparse_rows

        del class_images
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final flush + checkpoint ──────────────────────────────────
    if probe_tr_mm is not None:
        for vn in variant_names:
            for mm in probe_tr_mm[vn].values(): mm.flush()
    if topk_tr_mm is not None:
        for vn in variant_names:
            for mm in topk_tr_mm[vn].values():  mm.flush()

    # Always write a final checkpoint (includes the normalised prototypes).
    ckpt.save(cls_patchsae, cls_hg, class_success_count,
              class_done_count, total_processed,
              write_normalised=True)

    # Trim per-image memmaps to actual rows written.
    if probe_tr_mm is not None:
        if tr_row < N_train_est:
            for vn in variant_names:
                for k in ALL_KEYS:
                    np.save(probe_tr_paths[vn][k], np.load(probe_tr_paths[vn][k])[:tr_row])
        for vn in variant_names:
            np.save(y_tr_paths[vn], y_tr_buf[:tr_row])
    if topk_tr_mm is not None:
        if tr_row < N_train_est:
            for vn in variant_names:
                for k in ALL_KEYS:
                    np.save(topk_tr_paths[vn][k], np.load(topk_tr_paths[vn][k])[:tr_row])

    total_elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Done. Processed {total_processed:,} images "
          f"(of which {int(class_success_count.sum()):,} succeeded). "
          f"Errors: {total_errors}.  Wall time: {total_elapsed/60:.1f} min")
    for vn in variant_names:
        print(f"  [{vn}] Prototypes:    {paths[vn]['proto']}/")
        if args.save_probe_features:
            print(f"  [{vn}] Probe feats:   {paths[vn]['probe']}/  "
                  f"(shape [{tr_row}, {PROJ_DIM}])")
        if args.save_topk_indices:
            print(f"  [{vn}] Top-K indices: {paths[vn]['jaccard']}/  "
                  f"(shape [{tr_row}, {K}])")
    if sparse_writer is not None:
        print(f"  Sparse subset: {sparse_path}/  "
              f"({args.save_sparse_subset} imgs/class)")
    print(f"  Checkpoint state: {paths['_state']['state']}/   "
          f"(safe to delete after a clean finish)")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Partition merge helper (use after running multi-GPU partitions)
# ─────────────────────────────────────────────────────────────────────────────

def merge_partitions(parts: List[str], dest: str) -> None:
    """
    Merge prototype/probe/jaccard outputs from multiple class-range partitions
    into a single combined output tree.

        merge_partitions(["outputs/imagenet_130/part0", "outputs/imagenet_130/part1"],
                         "outputs/imagenet_130")

    What gets merged:
      - prototypes/<method>/cls_*.npy  → summed (each partition zeroed-out the
        classes it didn't own, so a per-class element-wise sum is the correct
        concatenation; rows are re-L1-normalised afterwards)
      - probe_features/X_*_tr.npy, y_tr.npy (if present)
          → concatenated along axis 0
      - topk_indices/topk_*_tr.npy      (if present)
          → concatenated along axis 0
    """
    import shutil
    from glob import glob

    if not parts:
        raise ValueError("merge_partitions: no input partitions.")

    os.makedirs(dest, exist_ok=True)

    # 1. Prototypes — sum per-class (each partition zeroed-out the others).
    # Find all (method, file_basename) combinations.
    proto_files = set()
    for p in parts:
        for f in glob(os.path.join(p, "prototypes*", "*", "*.npy")):
            rel = os.path.relpath(f, p)
            proto_files.add(rel)

    for rel in proto_files:
        dest_path = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        acc = None
        for p in parts:
            src = os.path.join(p, rel)
            if os.path.exists(src):
                arr = np.load(src)
                acc = arr if acc is None else acc + arr
        # Each partition's classes outside its range are zeros, so the sum is
        # the correct concatenation. (Re-normalise row-wise to be safe.)
        row_sum = acc.sum(axis=1, keepdims=True)
        acc = acc / np.where(row_sum > 1e-8, row_sum, 1.0)
        np.save(dest_path, acc)
        print(f"merged proto: {rel}")

    # 2. Probe features & labels — concatenate along axis 0.
    for sub in ("probe_features", "probe_features_cls",
                "topk_indices", "topk_indices_cls"):
        files: Dict[str, list] = defaultdict(list)
        for p in parts:
            for f in glob(os.path.join(p, sub, "*.npy")):
                files[os.path.basename(f)].append(f)
        if not files:
            continue
        os.makedirs(os.path.join(dest, sub), exist_ok=True)
        for fname, srcs in files.items():
            arrs = [np.load(s) for s in sorted(srcs)]
            cat  = np.concatenate(arrs, axis=0)
            np.save(os.path.join(dest, sub, fname), cat)
            print(f"merged {sub}/{fname}: {[a.shape for a in arrs]} → {cat.shape}")

    # 3. Copy proj_matrix.npy if present.
    for p in parts + [os.path.dirname(parts[0].rstrip("/"))]:
        src = os.path.join(os.path.dirname(p.rstrip("/")) or p, "proj_matrix.npy")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(os.path.dirname(dest.rstrip("/")) or dest,
                                           "proj_matrix.npy"))
            break


if __name__ == "__main__":
    main()