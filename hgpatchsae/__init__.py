"""HG-PatchSAE: heatmap-guided pooling of PatchSAE features."""

from .model import (
    ACTIVATION_MODES,
    ALL_METHODS,
    DEFAULT_SAE_THRESHOLD,
    HGPatchSAE,
    load_imagenet_class_names,
)

__all__ = [
    "ACTIVATION_MODES",
    "ALL_METHODS",
    "DEFAULT_SAE_THRESHOLD",
    "HGPatchSAE",
    "load_imagenet_class_names",
]
