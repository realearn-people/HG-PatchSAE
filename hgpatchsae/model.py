"""
HG-PatchSAE core: PatchSAE patch latents pooled with post-hoc heatmap weights.

`HGPatchSAE` wraps a frozen CLIP ViT-B/16, the pre-trained PatchSAE, and a
set of interchangeable heatmap extractors. For one image it computes

    z_p  = ReLU(W_enc x_p + b_enc)               PatchSAE latent of patch p  (Eq. 1)
    w_p  = mean of the heatmap over patch p      14 x 14 patch grid          (Eq. 2)
    f    = sum_p w_p z_p,  f <- f / ||f||_1      pooled feature vector       (Eq. 3)

Implementation details not spelled out in the paper:
  * activation_mode="thresholded_real" (default) zeroes latents below
    tau = 0.2 before pooling (DEFAULT_SAE_THRESHOLD).
  * Every heatmap is min-max scaled on the 14 x 14 grid. For the dense maps
    (attention rollout, raw attention, grad rollout, ...) patches below
    0.5 x max are zeroed first (DEFAULT_THRESHOLD_DENSE).
  * GradCAM and LayerCAM are computed at transformer block -4.
  * Heatmaps are not L1-normalised before pooling; the pooled vector f is
    L1-normalised instead, which gives the same direction as Eq. 2-3.

Heatmap methods (`ALL_METHODS`): legrad, gradcam, attention_rollout,
raw_attention, grad_rollout, eigencam, layercam, reciprocam, grad_sam,
transformer_attribution. All gradient-based methods share a single
forward+backward pass per image (`_compute_grad_bundle`).

Usage
-----
    from hgpatchsae import HGPatchSAE
    hg = HGPatchSAE()
    heatmap = hg.get_heatmap(image, "a photo of a tabby cat", method="layercam")
    feats   = hg.run_all_methods(image, true_class="tabby cat",
                                 methods=["layercam", "grad_sam"])

Paths to the PatchSAE checkpoint (data/sae_weight/base/out.pt) and its
feature statistics (out/feature_data/) are resolved relative to the repo root.
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
import json
import math
import os
import sys
import warnings
from typing import Dict, Iterable, List, Optional, Union

# Repo root (parent of hgpatchsae/) — makes src/ and tasks/ importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image

import matplotlib.pyplot as plt

# ── Project / open_clip / LeGRAD ─────────────────────────────────────────────
import open_clip
from legrad import LeWrapper, LePreprocess
from legrad.utils import visualize
from transformers import CLIPProcessor, CLIPModel

from src.demo.core import SAETester
from tasks.utils import (
    get_all_classnames,
    get_max_acts_and_images,
    get_sae_and_vit,
    load_datasets,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Methods whose heatmaps are dense (need post-hoc thresholding).
_DENSE_METHODS = {
    "attention_rollout",
    "raw_attention",
    "grad_rollout",
    "eigencam",
    "reciprocam",
}
DEFAULT_THRESHOLD_DENSE = 0.5

# Default per-feature activation threshold for the framework
# (thresholded_real mode). 0.2 is the value used for all paper results.
DEFAULT_SAE_THRESHOLD = 0.2

# All built-in heatmap methods (used as default for run_all_methods).
ALL_METHODS: List[str] = [
    "legrad",
    "gradcam",
    "attention_rollout",
    "raw_attention",
    "grad_rollout",
    "eigencam",
    "layercam",
    "reciprocam",
    "grad_sam",
    "transformer_attribution",
]

# Activation modes for run_all_methods.
#   "binary"           - z_p > τ → 1.0 else 0.0
#   "real"             - keep z_p as-is, no threshold
#   "thresholded_real" - z_p where z_p > τ, else 0.0     (default; used in the paper)
ACTIVATION_MODES = {"binary", "real", "thresholded_real"}


# ─────────────────────────────────────────────────────────────────────────────
# Helper: differentiable-attention context manager (for Chefer et al.)
# ─────────────────────────────────────────────────────────────────────────────

class _DifferentiableAttention:
    """
    Context manager that replaces each ResidualAttentionBlock's
    `block.attn.forward` with a manual implementation in which the softmax
    attention matrix is part of the autograd graph.

    On exit, the original forwards are restored.

    After a backward pass inside the `with` block, each entry in
    `self.handles[i]` exposes `.weights` (the attention matrix) with its
    `.grad` populated.

    Notes
    -----
    Only standard self-attention is supported (query == key == value).
    open_clip's ResidualAttentionBlock satisfies this. The patched forward
    matches the calling convention `attn(x, x, x, need_weights=..., ...)[0]`.
    """

    class _Handle:
        weights: Optional[torch.Tensor] = None

    def __init__(self, visual_encoder):
        self.blocks = list(visual_encoder.transformer.resblocks)
        self.original_forwards: Dict[int, callable] = {}
        self.handles: List[_DifferentiableAttention._Handle] = []

    def __enter__(self):
        self.handles = [self._Handle() for _ in self.blocks]
        for i, block in enumerate(self.blocks):
            self.original_forwards[i] = block.attn.forward
            block.attn.forward = self._make_forward(block.attn, self.handles[i])
        return self.handles

    def __exit__(self, exc_type, exc, tb):
        for i, block in enumerate(self.blocks):
            block.attn.forward = self.original_forwards[i]
        return False

    @staticmethod
    def _make_forward(attn_module: nn.MultiheadAttention, handle: "_DifferentiableAttention._Handle"):
        num_heads      = attn_module.num_heads
        embed_dim      = attn_module.embed_dim
        head_dim       = embed_dim // num_heads
        scale          = head_dim ** -0.5
        in_proj_weight = attn_module.in_proj_weight
        in_proj_bias   = attn_module.in_proj_bias
        out_proj       = attn_module.out_proj
        batch_first    = getattr(attn_module, "batch_first", False)

        def forward(query, key, value, **kwargs):
            # Self-attention: query / key / value are the same tensor.
            x = query
            if not batch_first:
                x = x.transpose(0, 1)        # [N, B, C] -> [B, N, C]
            B, N, C = x.shape

            qkv = F.linear(x, in_proj_weight, in_proj_bias)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]                # [B, H, N, D]

            logits = (q @ k.transpose(-2, -1)) * scale      # [B, H, N, N]
            mask = kwargs.get("attn_mask", None)
            if mask is not None:
                logits = logits + mask
            attn_weights = logits.softmax(dim=-1)           # [B, H, N, N]
            attn_weights.retain_grad()
            handle.weights = attn_weights

            out = attn_weights @ v                          # [B, H, N, D]
            out = out.permute(0, 2, 1, 3).reshape(B, N, C)
            out = F.linear(out, out_proj.weight, out_proj.bias)

            if not batch_first:
                out = out.transpose(0, 1)                   # [B, N, C] -> [N, B, C]

            return out, None                                # match nn.MultiheadAttention signature

        return forward


# ─────────────────────────────────────────────────────────────────────────────
# Helper: image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_imagenet_class_names(path: Optional[str] = None) -> List[str]:
    """Readable ImageNet-1k class names (imagenet_classes.json at the repo root)."""
    if path is None:
        path = os.path.join(_REPO_ROOT, "imagenet_classes.json")
    with open(path, "r") as f:
        if path.endswith(".json"):
            return list(json.load(f))
        return [line.strip() for line in f]


def _load_pil(image: Union[str, Image.Image]) -> Image.Image:
    """Accept a filepath, a URL-like string, or a PIL.Image. Returns RGB PIL."""
    if isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class HGPatchSAE:
    """
    PatchSAE + LeGRAD + 10 heatmap methods, all sharing one forward pass
    per image when invoked via `run_all_methods`.
    """

    # ─────────────────────────────────────────────────────────────────
    # Construction
    # ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        inference_model: str = "openai/clip-vit-base-patch16",
        class_names: Optional[Iterable[str]] = None,
        classnames_file: Optional[str] = None,
        sae_device: str = "cpu",
    ):
        self.inference_model = inference_model
        self.device          = "cuda" if torch.cuda.is_available() else "cpu"
        self.sae_device      = sae_device

        # HF CLIP for `get_inference` only — not used in the run_* pipeline.
        self.processor          = CLIPProcessor.from_pretrained(inference_model)
        self.model              = CLIPModel.from_pretrained(inference_model).to(self.device)
        self.imagenet_classes   = self._load_imagenet_classes(class_names, classnames_file)

        # ── PatchSAE ─────────────────────────────────────────────────
        self.datasets    = load_datasets()
        _cls_root        = os.path.join(_REPO_ROOT, "configs", "classnames")
        self.classnames  = get_all_classnames(self.datasets, data_root=_cls_root)

        self.root        = os.path.join(_REPO_ROOT, "out", "feature_data")
        self.sae_runname = "sae_base"
        self.vit_name    = "base"
        self.sae_path    = os.path.join(_REPO_ROOT, "data", "sae_weight", "base", "out.pt")

        self.sae, self.vit, self.sae_cfg = get_sae_and_vit(
            sae_path=self.sae_path,
            vit_type="base",
            device=self.sae_device,
            backbone=inference_model,
            model_path=None,
            classnames=None,
        )
        self.max_act_imgs, self.mean_acts = get_max_acts_and_images(
            self.datasets, self.root, self.sae_runname, self.vit_name
        )
        self.sae_clip = SAETester(
            self.vit, self.sae_cfg, self.sae,
            self.mean_acts, self.max_act_imgs,
            self.datasets, self.classnames,
            device=self.sae_device,
        )
        self.sae_dim = getattr(self.sae_cfg, "d_sae", 49152)

        # ── LeGRAD ───────────────────────────────────────────────────
        self.legrad_model_name = "ViT-B-16"
        self.legrad_pretrained = "openai"
        self.legrad_device     = self.device

        self.legrad_model, _, self.legrad_preprocess = open_clip.create_model_and_transforms(
            model_name=self.legrad_model_name,
            pretrained=self.legrad_pretrained,
            device=self.legrad_device,
        )
        self.legrad_tokenizer = open_clip.get_tokenizer(model_name=self.legrad_model_name)
        self.legrad_model.eval()
        self.legrad_model     = LeWrapper(self.legrad_model)
        self.legrad_preprocess = LePreprocess(preprocess=self.legrad_preprocess, image_size=224)

        # Custom heatmap plugin registry: name -> fn(image, text) -> ndarray[H,W]
        self._custom_methods: Dict[str, callable] = {}

        # Cache for the most recent full [197, sae_dim] SAE activation —
        # populated by `get_patch_sae_activations`, consumed by
        # `get_cls_sae_activation` so we never run the SAE forward twice.
        self._last_full_sae_acts: Optional[np.ndarray] = None

    # ─────────────────────────────────────────────────────────────────
    # Public helpers
    # ─────────────────────────────────────────────────────────────────

    def _load_imagenet_classes(self, class_names, filename) -> List[str]:
        if class_names is not None:
            return list(class_names)
        return load_imagenet_class_names(filename)

    def register_image(self, image_url) -> None:
        """Register an image with the underlying SAETester."""
        self.sae_clip.register_image(image_url)

    def get_inference(self, image_url, inference_show: bool = True) -> str:
        image = Image.open(image_url)
        text_prompts = [f"a photo of a {c}" for c in self.imagenet_classes]
        inputs = self.processor(
            text=text_prompts, images=image, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = outputs.logits_per_image.softmax(dim=1)
        idx        = probs.argmax().item()
        confidence = probs[0, idx].item()
        predicted  = self.imagenet_classes[idx]
        if inference_show:
            plt.imshow(image); plt.show()
            print(f"Predicted: {predicted} (confidence: {confidence:.2%})")
        return predicted

    # ─────────────────────────────────────────────────────────────────
    # Heatmap → patch-index conversion
    # ─────────────────────────────────────────────────────────────────

    def heatmap_to_vit_token_indices(
        self,
        heatmap,
        patch_size: int = 16,
        threshold: float = 0.0,
    ) -> dict:
        """
        Convert a pixel-space heatmap [1, H, W] (or [H, W]) into the per-patch
        statistics needed to weight 14×14 patch activations.
        """
        if isinstance(heatmap, torch.Tensor):
            heat = heatmap.detach().cpu().numpy()
        else:
            heat = np.asarray(heatmap)

        # squeeze leading 1's
        while heat.ndim > 2 and heat.shape[0] == 1:
            heat = heat[0]

        H, W = heat.shape
        num_p_h, num_p_w = math.ceil(H / patch_size), math.ceil(W / patch_size)

        ys, xs = np.nonzero(heat > threshold)
        if ys.size == 0:
            return dict(patch_coords=[], patch_indices=[], token_indices=[],
                        patch_max_values=[], patch_mean_values=[],
                        num_patches=(num_p_h, num_p_w))

        coords = sorted({(int(y // patch_size), int(x // patch_size)) for y, x in zip(ys, xs)})

        patch_indices = [r * num_p_w + c for r, c in coords]
        token_indices = [i + 1 for i in patch_indices]   # +1 for CLS

        patch_max, patch_mean = [], []
        for r, c in coords:
            y0, y1 = r * patch_size, min((r + 1) * patch_size, H)
            x0, x1 = c * patch_size, min((c + 1) * patch_size, W)
            region = heat[y0:y1, x0:x1]
            mask   = region > threshold
            if np.any(mask):
                vals = region[mask]
                patch_max.append(float(vals.max()))
                patch_mean.append(float(vals.mean()))
            else:
                patch_max.append(0.0)
                patch_mean.append(0.0)

        return dict(patch_coords=coords, patch_indices=patch_indices,
                    token_indices=token_indices,
                    patch_max_values=patch_max, patch_mean_values=patch_mean,
                    num_patches=(num_p_h, num_p_w))

    # ─────────────────────────────────────────────────────────────────
    # Internal model accessors
    # ─────────────────────────────────────────────────────────────────

    def _get_clip_model(self):
        wrapper = self.legrad_model
        for attr in ("model", "wrapped_model", "_model", "net", "clip_model"):
            cand = getattr(wrapper, attr, None)
            if cand is not None and hasattr(cand, "encode_image"):
                return cand
        for val in vars(wrapper).values():
            if hasattr(val, "encode_image") and hasattr(val, "encode_text"):
                return val
        if hasattr(wrapper, "encode_image"):
            return wrapper
        raise RuntimeError("Cannot locate the open_clip model inside LeWrapper.")

    def _get_visual_encoder(self):
        clip_model = self._get_clip_model()
        if hasattr(clip_model, "visual"):
            return clip_model.visual
        raise RuntimeError(f"CLIP model has no .visual attribute: {type(clip_model)}")

    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        pil = _load_pil(image)
        return self.legrad_preprocess(pil).unsqueeze(0).to(self.legrad_device)

    # ─────────────────────────────────────────────────────────────────
    # Attention capture (no-grad — for non-gradient methods)
    # ─────────────────────────────────────────────────────────────────

    def _capture_attn_no_grad(self, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Run the visual encoder and capture per-layer attention weights
        as [1, H, S, S] tensors (detached).
        """
        visual            = self._get_visual_encoder()
        attn_maps         = []
        original_forwards = {}

        for i, block in enumerate(visual.transformer.resblocks):
            orig = block.attn.forward

            def patched(query, key, value,
                        key_padding_mask=None, need_weights=False,
                        attn_mask=None, _orig=orig, _store=attn_maps):
                try:
                    out, w = _orig(query, key, value,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=True, average_attn_weights=False,
                                   attn_mask=attn_mask)
                except TypeError:
                    out, w = _orig(query, key, value,
                                   key_padding_mask=key_padding_mask,
                                   need_weights=True, attn_mask=attn_mask)
                    if w is not None and w.dim() == 3:
                        w = w.unsqueeze(1)
                if w is not None:
                    w = w.detach()
                    # [H, B=1, S, S] → [B, H, S, S]
                    if w.dim() == 4 and w.shape[0] != 1 and w.shape[1] == 1:
                        w = w.permute(1, 0, 2, 3).contiguous()
                    _store.append(w)
                return out, None

            original_forwards[i] = orig
            block.attn.forward = patched

        try:
            with torch.no_grad():
                visual(image_tensor)
        finally:
            for i, block in enumerate(visual.transformer.resblocks):
                if i in original_forwards:
                    block.attn.forward = original_forwards[i]

        if not attn_maps:
            raise RuntimeError("No attention maps captured.")
        return attn_maps

    # ─────────────────────────────────────────────────────────────────
    # Combined forward+backward bundle: features, feat-grads, attn, attn-grads
    # ─────────────────────────────────────────────────────────────────

    def _compute_grad_bundle(
        self,
        image_tensor: torch.Tensor,
        text_prompt: str,
    ) -> dict:
        """
        Single forward+backward pass that captures everything needed for
        EVERY gradient-based heatmap method in one go.

        Returns
        -------
        dict with keys:
            'feat'        : list of [B, N, C] tensors (per resblock output, detached)
            'feat_grad'   : list of [B, N, C] tensors (gradients, detached)
            'attn'        : list of [B, H, N, N] tensors (post-softmax weights, detached)
            'attn_grad'   : list of [B, H, N, N] tensors (gradients, detached)
        """
        visual     = self._get_visual_encoder()
        clip_model = self._get_clip_model()
        text       = self.legrad_tokenizer([text_prompt]).to(self.legrad_device)

        # 1. Register forward hooks on every resblock to capture feat+grad.
        feat_stores: List[dict] = []
        hooks: List = []
        for block in visual.transformer.resblocks:
            d = {}
            feat_stores.append(d)

            def _fwd(mod, inp, out, _d=d):
                out.retain_grad()
                _d["feat"] = out
            hooks.append(block.register_forward_hook(_fwd))

        # 2. Temporarily enable grad on all visual params (some are frozen).
        frozen = [p for blk in visual.transformer.resblocks
                  for p in blk.parameters() if not p.requires_grad]
        for p in frozen:
            p.requires_grad_(True)

        try:
            # 3. Replace attention forwards with differentiable versions.
            with _DifferentiableAttention(visual) as attn_handles:
                clip_model.zero_grad()
                with torch.enable_grad():
                    image_features = clip_model.encode_image(image_tensor, normalize=True)
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text, normalize=True)
                    score = (image_features * text_features).sum()
                    score.backward()

                # Snapshot attention weights + grads BEFORE leaving the context
                # (forwards are restored on __exit__, but the tensors keep their grads).
                attn_list      = [h.weights.detach() if h.weights is not None else None
                                  for h in attn_handles]
                attn_grad_list = [h.weights.grad.detach() if (h.weights is not None
                                                              and h.weights.grad is not None)
                                  else None for h in attn_handles]
        finally:
            for h in hooks:
                h.remove()
            for p in frozen:
                p.requires_grad_(False)

        if not feat_stores or "feat" not in feat_stores[0]:
            raise RuntimeError("_compute_grad_bundle: forward hooks never fired.")

        feat_list      = [d["feat"].detach() if d.get("feat") is not None else None
                          for d in feat_stores]
        feat_grad_list = [d["feat"].grad.detach() if (d.get("feat") is not None
                                                      and d["feat"].grad is not None)
                          else None for d in feat_stores]

        return {
            "feat":      feat_list,
            "feat_grad": feat_grad_list,
            "attn":      attn_list,
            "attn_grad": attn_grad_list,
        }

    @staticmethod
    def _split_patches(feat_or_grad: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Return the patch part of a [B, N, C] or [N, B, C] tensor with CLS dropped,
        normalised to [N_patches, C] for B=1.
        """
        x = feat_or_grad
        if x.shape[0] != batch_size:                # [N, B, C]
            x = x.permute(1, 0, 2).contiguous()     # [B, N, C]
        return x[0, 1:, :]                           # [N_patches, C]

    @staticmethod
    def _patch_to_pixel_heatmap(patch_map_2d: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
        """
        Apply RAW threshold → min-max normalise → upsample to [1, 224, 224].
        """
        pm = patch_map_2d.float()
        if threshold > 0.0:
            cutoff = pm.max() * threshold
            pm = pm.clone()
            pm[pm < cutoff] = 0.0
        pm_min, pm_max = pm.min(), pm.max()
        if pm_max - pm_min > 1e-8:
            pm = (pm - pm_min) / (pm_max - pm_min)
        else:
            pm = torch.zeros_like(pm)
        heatmap_px = F.interpolate(
            pm.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode="bilinear", align_corners=False,
        ).squeeze(0)                                 # [1, 224, 224]
        return heatmap_px

    # ═════════════════════════════════════════════════════════════════
    # Individual heatmap methods
    # All accept either a filepath OR a PIL.Image via `image`.
    # ═════════════════════════════════════════════════════════════════

    # ── Method 0: LeGRAD ────────────────────────────────────────────
    def legrad_run(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        threshold: float = 0.0,
        legrad_visualize: bool = False,
    ) -> torch.Tensor:
        image_t = self._preprocess(image)
        text    = self.legrad_tokenizer([text_prompt]).to(self.legrad_device)
        text_emb = self.legrad_model.encode_text(text, normalize=True)
        emap = self.legrad_model.compute_legrad_clip(image=image_t, text_embedding=text_emb)
        if threshold > 0:
            emap = emap.clone()
            emap[emap <= threshold] = 0
        if legrad_visualize:
            visualize(heatmaps=emap, image=image_t)
        return emap

    # ── Method 1: Attention rollout ─────────────────────────────────
    def attention_rollout_run(
        self,
        image: Union[str, Image.Image],
        text_prompt: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD_DENSE,
        discard_ratio: float = 0.1,
        head_fusion: str = "max",
        start_layer: int = 0,
        legrad_visualize: bool = False,
        _attn_maps: Optional[List[torch.Tensor]] = None,
        _image_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_t   = _image_tensor if _image_tensor is not None else self._preprocess(image)
        attn_maps = _attn_maps    if _attn_maps    is not None else self._capture_attn_no_grad(image_t)

        N = attn_maps[0].shape[-1]
        rollout = torch.eye(N, device=self.legrad_device)

        for attn in attn_maps[start_layer:]:
            attn_sq = attn.squeeze(0)
            while attn_sq.dim() > 3:
                attn_sq = attn_sq.squeeze(0)
            if attn_sq.dim() == 2:
                attn_sq = attn_sq.unsqueeze(0)        # [1, S, S]

            if head_fusion == "max":
                fused = attn_sq.max(dim=0).values
            elif head_fusion == "min":
                fused = attn_sq.min(dim=0).values
            else:
                fused = attn_sq.mean(dim=0)

            while fused.dim() > 2:
                fused = fused.squeeze(0)

            if discard_ratio > 0:
                flat = fused.view(-1)
                k = max(1, int(discard_ratio * flat.numel()))
                cutoff = flat.kthvalue(k).values.item()
                fused = fused.clamp(min=cutoff)

            fused = fused + torch.eye(N, device=self.legrad_device)
            fused = fused / (fused.sum(dim=-1, keepdim=True) + 1e-8)
            rollout = fused @ rollout

        cls_attn = rollout[0, 1:]
        grid = int(cls_attn.numel() ** 0.5)
        heatmap_px = self._patch_to_pixel_heatmap(cls_attn.reshape(grid, grid), threshold=threshold)
        if legrad_visualize:
            visualize(heatmaps=heatmap_px, image=image_t)
        return heatmap_px

    # ── Method 2: Raw CLS attention ─────────────────────────────────
    def raw_attention_run(
        self,
        image: Union[str, Image.Image],
        text_prompt: Optional[str] = None,
        threshold: float = DEFAULT_THRESHOLD_DENSE,
        layer: Optional[int] = None,
        head_fusion: str = "max",
        legrad_visualize: bool = False,
        _attn_maps: Optional[List[torch.Tensor]] = None,
        _image_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        image_t   = _image_tensor if _image_tensor is not None else self._preprocess(image)
        attn_maps = _attn_maps    if _attn_maps    is not None else self._capture_attn_no_grad(image_t)

        L = len(attn_maps)
        if layer is None:
            layer = int(L * 0.75)
        attn_sq = attn_maps[layer].squeeze(0)
        if attn_sq.dim() == 2:
            attn_sq = attn_sq.unsqueeze(0)

        if head_fusion == "max":
            cls_attn = attn_sq.max(dim=0).values[0, 1:]
        elif head_fusion == "min":
            cls_attn = attn_sq.min(dim=0).values[0, 1:]
        else:
            cls_attn = attn_sq.mean(dim=0)[0, 1:]

        grid = int(cls_attn.numel() ** 0.5)
        heatmap_px = self._patch_to_pixel_heatmap(cls_attn.reshape(grid, grid), threshold=threshold)
        if legrad_visualize:
            visualize(heatmaps=heatmap_px, image=image_t)
        return heatmap_px

    # ── Gradient methods from a pre-computed bundle ─────────────────

    def _gradcam_from_bundle(self, bundle: dict, layer: int, threshold: float) -> torch.Tensor:
        feat = bundle["feat"][layer]
        grad = bundle["feat_grad"][layer]
        if feat is None or grad is None:
            raise RuntimeError(f"GradCAM: missing feat/grad for layer {layer}.")
        B = 1
        patch_feat = self._split_patches(feat, B)           # [N, C]
        patch_grad = self._split_patches(grad, B)
        weights = patch_grad.abs().mean(dim=-1, keepdim=True)
        cam = F.relu((weights * patch_feat).sum(dim=-1))
        grid = int(cam.numel() ** 0.5)
        return self._patch_to_pixel_heatmap(cam.reshape(grid, grid), threshold=threshold)

    def _layercam_from_bundle(self, bundle: dict, layer: int, threshold: float) -> torch.Tensor:
        feat = bundle["feat"][layer]
        grad = bundle["feat_grad"][layer]
        if feat is None or grad is None:
            raise RuntimeError(f"LayerCAM: missing feat/grad for layer {layer}.")
        B = 1
        patch_feat = self._split_patches(feat, B)
        patch_grad = self._split_patches(grad, B)
        cam = (F.relu(patch_grad) * F.relu(patch_feat)).sum(dim=-1)
        grid = int(cam.numel() ** 0.5)
        return self._patch_to_pixel_heatmap(cam.reshape(grid, grid), threshold=threshold)

    def _grad_sam_from_bundle(self, bundle: dict, threshold: float) -> torch.Tensor:
        """Per-layer ReLU(grad)*|feat|, averaged across layers."""
        patch_imp = None
        n_used = 0
        for feat, grad in zip(bundle["feat"], bundle["feat_grad"]):
            if feat is None or grad is None:
                continue
            patch_feat = self._split_patches(feat, 1)
            patch_grad = self._split_patches(grad, 1)
            imp = (F.relu(patch_grad) * patch_feat.abs()).sum(dim=-1)
            patch_imp = imp if patch_imp is None else patch_imp + imp
            n_used += 1
        if n_used == 0:
            raise RuntimeError("grad_sam: no valid layers.")
        patch_imp = patch_imp / n_used
        grid = int(patch_imp.numel() ** 0.5)
        return self._patch_to_pixel_heatmap(patch_imp.reshape(grid, grid), threshold=threshold)

    def _grad_rollout_from_bundle(
        self, bundle: dict, threshold: float,
        discard_ratio: float = 0.1, start_layer: int = 0,
    ) -> torch.Tensor:
        attn_list = bundle["attn"]
        grad_list = bundle["attn_grad"]
        if not attn_list:
            raise RuntimeError("grad_rollout: no attention captured.")
        N = attn_list[0].shape[-1]
        rollout = torch.eye(N, device=self.legrad_device)

        for attn, grad in zip(attn_list[start_layer:], grad_list[start_layer:]):
            if attn is None:
                continue
            attn_sq = attn.squeeze(0)
            while attn_sq.dim() > 3:
                attn_sq = attn_sq.squeeze(0)
            if attn_sq.dim() == 2:
                attn_sq = attn_sq.unsqueeze(0)

            if grad is not None:
                grad_sq = grad.squeeze(0)
                while grad_sq.dim() > 3:
                    grad_sq = grad_sq.squeeze(0)
                if grad_sq.dim() == 2:
                    grad_sq = grad_sq.unsqueeze(0)
                cam = F.relu(grad_sq * attn_sq).mean(dim=0)
            else:
                cam = attn_sq.max(dim=0).values

            while cam.dim() > 2:
                cam = cam.squeeze(0)

            if discard_ratio > 0:
                flat = cam.view(-1)
                k = max(1, int(discard_ratio * flat.numel()))
                cutoff = flat.kthvalue(k).values.item()
                cam = cam.clamp(min=cutoff)

            cam = cam + torch.eye(N, device=self.legrad_device)
            cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-8)
            rollout = cam @ rollout

        cls_attn = rollout[0, 1:]
        grid = int(cls_attn.numel() ** 0.5)
        return self._patch_to_pixel_heatmap(cls_attn.reshape(grid, grid), threshold=threshold)

    def _transformer_attribution_from_bundle(self, bundle: dict, threshold: float) -> torch.Tensor:
        """
        Chefer et al. 2021 ("Generic Attention-model Explainability"):

            R = I_n
            for each block b:
                Ā_b = ReLU( grad_b ⊙ A_b ).mean(heads)
                R   = R + Ā_b @ R
            relevance = R[CLS, patches]
        """
        attn_list = bundle["attn"]
        grad_list = bundle["attn_grad"]
        if not attn_list or attn_list[0] is None:
            raise RuntimeError("transformer_attribution: no attention captured.")

        N = attn_list[0].shape[-1]
        R = torch.eye(N, device=self.legrad_device)

        for A, A_grad in zip(attn_list, grad_list):
            if A is None or A_grad is None:
                continue
            # A: [B, H, N, N]; B=1.  Matches Chefer's official `avg_heads`:
            #   cam = (grad * cam).clamp(min=0).mean(dim=heads)
            # NO row-normalisation here — Chefer's code does not normalise.
            Ahat = F.relu(A_grad * A).mean(dim=1).squeeze(0)         # [N, N]
            R = R + Ahat @ R

        relevance = R[0, 1:]
        grid = int(relevance.numel() ** 0.5)
        return self._patch_to_pixel_heatmap(relevance.reshape(grid, grid), threshold=threshold)

    # ── Public single-method gradient wrappers (run their own backward) ──

    def gradcam_run(self, image, text_prompt="", threshold=0.0, layer=-4,
                    legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        L = len(bundle["feat"])
        layer = layer % L
        hm = self._gradcam_from_bundle(bundle, layer, threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    def layercam_run(self, image, text_prompt="", threshold=0.0, layer=-4,
                     legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        L = len(bundle["feat"])
        layer = layer % L
        hm = self._layercam_from_bundle(bundle, layer, threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    def grad_sam_run(self, image, text_prompt="", threshold=0.0,
                     legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        hm = self._grad_sam_from_bundle(bundle, threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    def grad_rollout_run(self, image, text_prompt="", threshold=DEFAULT_THRESHOLD_DENSE,
                         discard_ratio=0.1, start_layer=0,
                         legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        hm = self._grad_rollout_from_bundle(bundle, threshold, discard_ratio, start_layer)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    def transformer_attribution_run(self, image, text_prompt="", threshold=0.0,
                                    legrad_visualize=False) -> torch.Tensor:
        """
        Chefer et al. 2021 — Transformer-Explainability.
        See `_transformer_attribution_from_bundle` for the algorithm.
        """
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        hm = self._transformer_attribution_from_bundle(bundle, threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    # ── GradCAM at EVERY layer (new feature for layer-choice ablation) ──

    def gradcam_all_layers_run(
        self,
        image: Union[str, Image.Image],
        text_prompt: str = "",
        threshold: float = 0.0,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute GradCAM heatmaps for every transformer block in a SINGLE
        forward+backward pass. Returns {layer_idx: heatmap_px [1,224,224]}.

        Useful for measuring how sensitive a downstream metric is to layer
        choice — call this on a small image subset, then compare results.
        """
        image_t = self._preprocess(image)
        bundle  = self._compute_grad_bundle(image_t, text_prompt)
        out = {}
        for li in range(len(bundle["feat"])):
            try:
                out[li] = self._gradcam_from_bundle(bundle, li, threshold)
            except RuntimeError:
                out[li] = None
        return out

    # ── Method 5: EigenCAM (no gradients) ───────────────────────────
    def eigencam_run(self, image, text_prompt=None, threshold=DEFAULT_THRESHOLD_DENSE,
                     layer=-1, legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        visual  = self._get_visual_encoder()
        resblocks = list(visual.transformer.resblocks)
        target = resblocks[layer % len(resblocks)]

        store = {}

        def fwd(mod, inp, out):
            store["feat"] = out.detach()
        h = target.register_forward_hook(fwd)
        try:
            with torch.no_grad():
                self._get_clip_model().encode_image(image_t, normalize=False)
        finally:
            h.remove()

        feat = store["feat"]
        if feat.shape[0] != image_t.shape[0]:
            feat = feat.permute(1, 0, 2).contiguous()
        patch_feat = feat[0, 1:, :]
        patch_feat = patch_feat - patch_feat.mean(dim=0, keepdim=True)
        _, _, Vt = torch.linalg.svd(patch_feat, full_matrices=False)
        first_pc = patch_feat @ Vt[0]
        if first_pc.sum() < 0:
            first_pc = -first_pc
        first_pc = F.relu(first_pc)
        grid = int(first_pc.numel() ** 0.5)
        hm = self._patch_to_pixel_heatmap(first_pc.reshape(grid, grid), threshold=threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    # ── Method 7: ViT-ReciproCAM (text-conditioned, no gradients) ──
    def reciprocam_run(self, image, text_prompt="", threshold=DEFAULT_THRESHOLD_DENSE,
                       layer=-1, legrad_visualize=False) -> torch.Tensor:
        image_t = self._preprocess(image)
        text    = self.legrad_tokenizer([text_prompt]).to(self.legrad_device)
        visual  = self._get_visual_encoder()
        clip_model = self._get_clip_model()
        resblocks = list(visual.transformer.resblocks)
        target = resblocks[layer % len(resblocks)]

        store = {}

        def fwd(mod, inp, out):
            store["feat"] = out.detach()
        h = target.register_forward_hook(fwd)
        try:
            with torch.no_grad():
                clip_model.encode_image(image_t, normalize=False)
                text_features = clip_model.encode_text(text, normalize=True)
        finally:
            h.remove()

        feat = store["feat"]
        if feat.shape[0] != image_t.shape[0]:
            feat = feat.permute(1, 0, 2).contiguous()
        patch_feat = feat[0, 1:, :]
        if hasattr(visual, "ln_post"):
            patch_feat = visual.ln_post(patch_feat)
        if hasattr(visual, "proj") and visual.proj is not None:
            patch_proj = patch_feat @ visual.proj
        else:
            patch_proj = patch_feat
        patch_proj = F.normalize(patch_proj, dim=-1)
        cam = F.relu((patch_proj * text_features).sum(dim=-1))
        grid = int(cam.numel() ** 0.5)
        hm = self._patch_to_pixel_heatmap(cam.reshape(grid, grid), threshold=threshold)
        if legrad_visualize:
            visualize(heatmaps=hm, image=image_t)
        return hm

    # ─────────────────────────────────────────────────────────────────
    # Dispatcher
    # ─────────────────────────────────────────────────────────────────

    def get_heatmap(
        self,
        image: Union[str, Image.Image],
        text_prompt: str = "",
        threshold: float = 0.0,
        method: str = "legrad",
        legrad_visualize: bool = False,
        **method_kwargs,
    ) -> torch.Tensor:
        dispatch = {
            "legrad":                   self.legrad_run,
            "attention_rollout":        self.attention_rollout_run,
            "raw_attention":            self.raw_attention_run,
            "gradcam":                  self.gradcam_run,
            "grad_rollout":             self.grad_rollout_run,
            "eigencam":                 self.eigencam_run,
            "layercam":                 self.layercam_run,
            "reciprocam":               self.reciprocam_run,
            "grad_sam":                 self.grad_sam_run,
            "transformer_attribution":  self.transformer_attribution_run,
        }

        if method in self._custom_methods:
            raw = self._custom_methods[method](image, text_prompt)
            arr = np.array(raw, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            return torch.from_numpy(arr)

        if method not in dispatch:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Built-in: {list(dispatch.keys())}. Custom: {list(self._custom_methods.keys())}."
            )

        call_kwargs = dict(image=image, text_prompt=text_prompt,
                           legrad_visualize=legrad_visualize, **method_kwargs)
        if threshold != 0.0 or method not in _DENSE_METHODS:
            call_kwargs["threshold"] = threshold

        return dispatch[method](**call_kwargs)

    # ─────────────────────────────────────────────────────────────────
    # SAE activation utilities
    # ─────────────────────────────────────────────────────────────────

    def _run_sae_full(self) -> np.ndarray:
        """
        Run ViT + SAE exactly once for the currently registered image and
        return the full [197, sae_dim] activation tensor (CLS at index 0,
        patches at 1..196), AFTER the noise filter.  Result is cached on
        `self._last_full_sae_acts` so `get_cls_sae_activation` reuses it
        without an extra forward pass.

        Calls the SAETester's internal hooks directly (same primitives it
        uses in `get_all_tokens_activations_batch`); avoids running the
        forward twice when both patch + CLS are needed.
        """
        vit_act = self.sae_clip._run_vit_hook()
        sae_act = self.sae_clip._run_sae_hook(vit_act)          # [1, 197, sae_dim]
        full    = sae_act[0, :197, :].detach().cpu().numpy()    # [197, sae_dim]
        full    = self.sae_clip._filter_out_nosiy_activation(full)
        self._last_full_sae_acts = full.astype(np.float32, copy=False)
        return self._last_full_sae_acts

    def get_patch_sae_activations(self) -> np.ndarray:
        """
        Return per-patch SAE activations for the currently registered image
        as a [196, sae_dim] float32 ndarray (CLS dropped).  Side-effect:
        caches the full [197, sae_dim] tensor so `get_cls_sae_activation`
        can return the CLS row without a second forward pass.
        """
        full = self._run_sae_full()                              # [197, sae_dim]
        return full[1:]                                           # [196, sae_dim]

    def get_cls_sae_activation(self) -> Optional[np.ndarray]:
        """
        Return SAE activation for the CLS token as a [sae_dim] float32 ndarray.
        Reuses the cache populated by `get_patch_sae_activations`; if no
        cache is available it runs a fresh forward pass.
        """
        if self._last_full_sae_acts is not None:
            return self._last_full_sae_acts[0]
        return self._run_sae_full()[0]

    @staticmethod
    def _apply_activation_mode(
        patch_fires: torch.Tensor,
        mode: str,
        sae_threshold: float,
    ) -> torch.Tensor:
        """
        Apply the activation processing mode to raw SAE outputs.
            "binary"           : (z > τ).float()
            "real"             : z   (no threshold)
            "thresholded_real" : z where z > τ, else 0
        """
        if mode == "binary":
            return (patch_fires > sae_threshold).float()
        if mode == "real":
            return patch_fires
        if mode == "thresholded_real":
            return torch.where(patch_fires > sae_threshold,
                               patch_fires, torch.zeros_like(patch_fires))
        raise ValueError(f"Unknown activation mode '{mode}'. "
                         f"Expected one of {sorted(ACTIVATION_MODES)}.")

    @staticmethod
    def _l1_norm(arr: np.ndarray) -> np.ndarray:
        s = arr.sum()
        return arr / (s + 1e-8)

    # ─────────────────────────────────────────────────────────────────
    # Framework API — plugin registration + multi-method single-pass
    # ─────────────────────────────────────────────────────────────────

    def register_heatmap_method(self, name: str, fn) -> None:
        """Register fn(image, text_prompt) → ndarray [H, W] as a named method."""
        self._custom_methods[name] = fn

    def run_all_methods(
        self,
        image: Union[str, Image.Image],
        true_class: Optional[str] = None,
        methods: Optional[List[str]] = None,
        sae_threshold: float = DEFAULT_SAE_THRESHOLD,
        random_seed: Optional[int] = None,
        activation_mode: str = "thresholded_real",
        include_cls: Union[bool, str] = False,
        cls_weight: float = 1.0,
    ) -> Dict[str, np.ndarray]:
        """
        Single-image multi-method extraction.

        Pipeline:
            1. Run the visual encoder + SAE ONCE  →  per-patch vectors z_p
            2. Apply `activation_mode` (default: real values, threshold τ)
            3. For each heatmap method:
                 - compute heatmap from a shared grad bundle (gradient methods)
                   or via direct call (non-gradient methods)
                 - extract patch indices + weights from the heatmap
                 - weighted-sum the per-patch vectors → feature vector
                 - (optional) add CLS-token SAE activation
                 - L1-normalise
            4. Return {method_name: ndarray [sae_dim]}

        Parameters
        ----------
        image           : path or PIL.Image
        true_class      : text prompt for gradient methods
        methods         : subset of ALL_METHODS ∪ custom ∪ {"random"};
                          default = ALL_METHODS + custom + ["random"]
        sae_threshold   : τ in the framework, default 0.2
        random_seed     : RNG for the "random" baseline method
        activation_mode : "thresholded_real" (default), "real", or "binary"
        include_cls     : False/True   → standard behaviour (no CLS / +CLS)
                          "both"       → return BOTH variants in one call;
                                         CLS-augmented entries get a "_cls"
                                         suffix (e.g. "legrad" and "legrad_cls").
                                         All the expensive work (heatmap +
                                         patch weighting) is shared.
        cls_weight      : scalar multiplier on the CLS vector when added

        Returns
        -------
        dict {method: ndarray [sae_dim]}, always including "uniform".
        With include_cls="both", the dict also contains "<method>_cls"
        entries for every method.
        """
        if methods is None:
            methods = ALL_METHODS + list(self._custom_methods.keys()) + ["random"]
        if activation_mode not in ACTIVATION_MODES:
            raise ValueError(f"activation_mode must be one of {sorted(ACTIVATION_MODES)}.")

        # Normalise the cls flag.
        if isinstance(include_cls, str):
            if include_cls not in ("off", "on", "both"):
                raise ValueError("include_cls string must be 'off', 'on', or 'both'.")
            cls_mode = include_cls
        else:
            cls_mode = "on" if include_cls else "off"
        want_patch = cls_mode in ("off", "both")
        want_cls   = cls_mode in ("on",  "both")
        dual       = cls_mode == "both"

        # ── 1. SAE forward once ────────────────────────────────────
        # SAETester.register_image accepts both a path and a PIL image.
        self.sae_clip.register_image(image)
        patch_acts = torch.from_numpy(self.get_patch_sae_activations())  # [196, sae_dim]
        act_patches = self._apply_activation_mode(patch_acts, activation_mode, sae_threshold)

        # CLS vector (computed only if any output needs it)
        cls_vec_np: Optional[np.ndarray] = None
        if want_cls:
            cls_arr = self.get_cls_sae_activation()
            if cls_arr is None:
                warnings.warn("include_cls requested but SAETester provides no CLS "
                              "activation; treating CLS as zeros for this image.")
                cls_vec_np = np.zeros(int(act_patches.shape[1]), dtype=np.float32)
            else:
                cls_t = torch.from_numpy(cls_arr).float()
                # Apply the SAME activation mode to CLS for consistency.
                cls_t = self._apply_activation_mode(cls_t.unsqueeze(0),
                                                   activation_mode, sae_threshold).squeeze(0)
                cls_vec_np = (cls_weight * cls_t).numpy().astype(np.float32)

        # Helper: emit one or both variants of a raw (pre-norm, pre-CLS) vec.
        results: Dict[str, np.ndarray] = {}
        def _emit(name: str, vec_patch_only: np.ndarray) -> None:
            v = vec_patch_only.astype(np.float32, copy=False)
            if want_patch:
                results[name] = self._l1_norm(v)
            if want_cls:
                v_cls = v + cls_vec_np
                key = f"{name}_cls" if dual else name
                results[key] = self._l1_norm(v_cls)

        # ── 2. Uniform baseline (PatchSAE — sum over all patches) ──
        uni = act_patches.sum(dim=0).numpy().astype(np.float32)
        _emit("uniform", uni)

        # ── 3. Pre-compute heavy artefacts shared across methods ───
        image_t = self._preprocess(image)

        gradient_methods = {
            "gradcam", "layercam", "grad_sam", "grad_rollout",
            "transformer_attribution",
        }
        attn_methods     = {"attention_rollout", "raw_attention"}

        need_grad_bundle = any(m in gradient_methods for m in methods)
        need_attn_maps   = any(m in attn_methods    for m in methods)

        grad_bundle  = self._compute_grad_bundle(image_t, true_class or "") if need_grad_bundle else None
        attn_no_grad = self._capture_attn_no_grad(image_t)                  if need_attn_maps   else None

        # Cache the L1-normed uniform fallback (correct for whichever variant).
        def _uniform_fallback_for(method: str) -> None:
            if want_patch:
                results[method] = results["uniform"].copy()
            if want_cls:
                key = f"{method}_cls" if dual else method
                results[key] = results["uniform_cls" if dual else "uniform"].copy()

        # ── 4. Apply each heatmap method as a spatial weighting ────
        for method in methods:
            try:
                if method == "random":
                    rng    = np.random.default_rng(random_seed)
                    rand_w = rng.random(act_patches.shape[0]).astype(np.float32)
                    rand_w /= (rand_w.sum() + 1e-8)
                    w_t = torch.from_numpy(rand_w).unsqueeze(1)
                    vec = (act_patches * w_t).sum(dim=0).numpy()
                elif method == "uniform":
                    continue
                else:
                    heatmap = self._heatmap_for_method(
                        method=method,
                        image=image,
                        image_tensor=image_t,
                        text_prompt=true_class or "",
                        grad_bundle=grad_bundle,
                        attn_maps_no_grad=attn_no_grad,
                    )
                    indices  = self.heatmap_to_vit_token_indices(heatmap, patch_size=16)
                    weights  = np.array(indices.get("patch_mean_values", []), dtype=np.float32)
                    patch_idx = indices.get("patch_indices", [])

                    if len(patch_idx) == 0 or len(patch_idx) != len(weights):
                        _uniform_fallback_for(method)
                        continue

                    w_t = torch.from_numpy(weights).unsqueeze(1)              # [n_sel, 1]
                    vec = (act_patches[patch_idx] * w_t).sum(dim=0).numpy()

                _emit(method, vec)

            except Exception as e:
                warnings.warn(f"run_all_methods: '{method}' failed — {e}")
                _uniform_fallback_for(method)

        return results

    # ── Internal: dispatch a single heatmap, reusing cached artefacts ──
    def _heatmap_for_method(
        self,
        method: str,
        image,
        image_tensor: torch.Tensor,
        text_prompt: str,
        grad_bundle: Optional[dict],
        attn_maps_no_grad: Optional[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Return a heatmap for `method`, using cached artefacts where possible."""
        if method == "legrad":
            return self.legrad_run(image=image, text_prompt=text_prompt, threshold=0.0)

        if method == "attention_rollout":
            return self.attention_rollout_run(
                image=image, text_prompt=text_prompt,
                _attn_maps=attn_maps_no_grad, _image_tensor=image_tensor,
            )
        if method == "raw_attention":
            return self.raw_attention_run(
                image=image, text_prompt=text_prompt,
                _attn_maps=attn_maps_no_grad, _image_tensor=image_tensor,
            )

        if method == "gradcam":
            return self._gradcam_from_bundle(grad_bundle, layer=-4 % len(grad_bundle["feat"]),
                                             threshold=0.0)
        if method == "layercam":
            return self._layercam_from_bundle(grad_bundle, layer=-4 % len(grad_bundle["feat"]),
                                              threshold=0.0)
        if method == "grad_sam":
            return self._grad_sam_from_bundle(grad_bundle, threshold=0.0)
        if method == "grad_rollout":
            return self._grad_rollout_from_bundle(grad_bundle, threshold=DEFAULT_THRESHOLD_DENSE)
        if method == "transformer_attribution":
            return self._transformer_attribution_from_bundle(grad_bundle, threshold=0.0)

        if method == "eigencam":
            return self.eigencam_run(image=image)
        if method == "reciprocam":
            return self.reciprocam_run(image=image, text_prompt=text_prompt)

        # Custom plugin
        if method in self._custom_methods:
            raw = self._custom_methods[method](image, text_prompt)
            arr = np.array(raw, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]
            return torch.from_numpy(arr)

        raise ValueError(f"Unknown method '{method}'.")
