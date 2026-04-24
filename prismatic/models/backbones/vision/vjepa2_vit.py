"""
vjepa2_vit.py

Vision backbone wrapping V-JEPA 2 encoder for use in MemoryVLA.
Replaces the DINOv2 + SigLIP dual-stream backbone with a single-stream
V-JEPA 2 ViT encoder that has native temporal/motion understanding.

Key differences from DINOSigLIP backbone:
  - Single-stream (not dual-stream concat)
  - patch_size=16 (not 14), so 224px -> 196 patches (not 256)
  - Uses 3D-RoPE positional encoding
  - Pre-trained on video via self-supervised JEPA objective
  - Weights loaded from Meta .pt format (not timm/HF hub)
"""

import sys
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision import transforms

from prismatic.models.backbones.vision.base_vision import VisionBackbone


# ── V-JEPA 2 model configurations ──
VJEPA2_CONFIGS = {
    "vjepa2-vit-large-224px": {
        "arch": "vit_large",        # embed_dim=1024, depth=24, heads=16
        "img_size": 224,
        "patch_size": 16,
    },
    "vjepa2-vit-huge-224px": {
        "arch": "vit_huge",         # embed_dim=1280, depth=32, heads=16
        "img_size": 224,
        "patch_size": 16,
    },
    "vjepa2-vit-giant-256px": {
        "arch": "vit_giant_xformers",  # embed_dim=1408, depth=40, heads=22
        "img_size": 256,
        "patch_size": 16,
    },
}

VJEPA2_EMBED_DIMS = {
    "vit_large": 1024,
    "vit_huge": 1280,
    "vit_giant_xformers": 1408,
}


class VJEPA2ImageTransform:
    """
    V-JEPA 2 image preprocessing: Resize → ToTensor → Normalize.
    Uses ImageNet normalization statistics (same as V-JEPA 2 pretraining).
    """

    def __init__(self, img_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize(
                (img_size, img_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        # Single-stream: forward returns a plain tensor, not a dict
        self.is_prismatic = False

    def __call__(self, img: Image.Image, **kwargs) -> torch.Tensor:
        return self.transform(img)


def _import_vjepa2_modules():
    """
    Lazily import V-JEPA 2 source modules.
    Assumes the vjepa2 repo is available either:
      (a) installed as a package, or
      (b) accessible via VJEPA2_ROOT env var / sibling directory.
    """
    try:
        from src.models.vision_transformer import VisionTransformer as ViT
        from src.models.utils.modules import Block
        return ViT, Block
    except ImportError:
        import os
        vjepa2_root = os.environ.get(
            "VJEPA2_ROOT",
            str(Path(__file__).resolve().parents[4] / "vjepa2"),
        )
        if vjepa2_root not in sys.path:
            sys.path.insert(0, vjepa2_root)
        from src.models.vision_transformer import VisionTransformer as ViT
        from src.models.utils.modules import Block
        return ViT, Block


class VJEPA2ViTBackbone(VisionBackbone):
    """
    Wraps a V-JEPA 2 ViT encoder as a MemoryVLA-compatible vision backbone.

    The encoder operates in image mode (num_frames=1) and outputs patch-level
    features of shape [B, num_patches, embed_dim].

    Args:
        vision_backbone_id: Key in VJEPA2_CONFIGS, e.g. "vjepa2-vit-large-224px"
        image_resize_strategy: One of "resize-naive" | "resize-crop" | "letterbox"
        default_image_size: Input image resolution
        vjepa2_checkpoint_path: Path to pretrained .pt checkpoint file
    """

    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        vjepa2_checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            vision_backbone_id,
            image_resize_strategy,
            default_image_size=default_image_size,
        )

        cfg = VJEPA2_CONFIGS[vision_backbone_id]
        arch_name = cfg["arch"]
        img_size = cfg["img_size"]
        patch_size = cfg["patch_size"]

        # Import V-JEPA 2 modules
        self._ViTCls, self._BlockCls = _import_vjepa2_modules()

        # Build encoder in image mode
        from src.models import vision_transformer as vit_module

        build_fn = vit_module.__dict__[arch_name]
        self.featurizer = build_fn(
            patch_size=patch_size,
            img_size=(img_size, img_size),
            num_frames=1,       # image mode
            tubelet_size=1,     # image mode
            use_sdpa=True,
            use_rope=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
        )

        # Load pretrained weights if provided
        if vjepa2_checkpoint_path is not None:
            self._load_pretrained(vjepa2_checkpoint_path)

        self.featurizer.eval()

        # Image transform
        self._img_size = img_size
        self.image_transform = VJEPA2ImageTransform(img_size)

        # Cache metadata
        self._embed_dim = VJEPA2_EMBED_DIMS[arch_name]
        self._num_patches = (img_size // patch_size) ** 2

    def _load_pretrained(self, checkpoint_path: str) -> None:
        """Load V-JEPA 2 pretrained weights from a .pt checkpoint."""
        state_dict = torch.load(checkpoint_path, map_location="cpu")

        # V-JEPA 2 checkpoints store encoder weights under "target_encoder"
        # or "encoder" key, with "module." and "backbone." prefixes
        key = "target_encoder" if "target_encoder" in state_dict else "encoder"
        raw_sd = state_dict[key]

        cleaned_sd = {}
        for k, v in raw_sd.items():
            k = k.replace("module.", "").replace("backbone.", "")
            cleaned_sd[k] = v

        # strict=False because RoPE models don't have pos_embed in state_dict
        missing, unexpected = self.featurizer.load_state_dict(
            cleaned_sd, strict=False
        )
        if missing:
            print(f"[VJEPA2ViTBackbone] Missing keys: {missing}")
        if unexpected:
            print(f"[VJEPA2ViTBackbone] Unexpected keys: {unexpected}")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP policy that wraps each transformer block."""
        vit_wrap = partial(
            _module_wrap_policy, module_classes={type(self.featurizer)}
        )
        block_wrap = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={self._BlockCls},
        )
        return partial(_or_policy, policies=[vit_wrap, block_wrap])

    def forward(
        self, pixel_values: Union[torch.Tensor, dict]
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [B, 3, H, W] tensor (single-stream, not dict)

        Returns:
            Patch features [B, num_patches, embed_dim]
        """
        # V-JEPA 2 with tubelet_size=1 uses Conv2d patch_embed,
        # so it expects 4D [B, C, H, W] directly — no unsqueeze needed.
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self._img_size, self._img_size)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
