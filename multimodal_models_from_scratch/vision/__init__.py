"""
Vision encoding components.

Modules:
- patch_embedding: Patch Embedding implementation
- vit: Vision Transformer model
- image_processor: Image preprocessing utilities
- backbone: CNN Backbone (ResNet) for DETR
"""

from multimodal_models_from_scratch.vision.image_processor import ImageProcessor
from multimodal_models_from_scratch.vision.patch_embedding import PatchEmbedding
from multimodal_models_from_scratch.vision.vit import ViTModel
from multimodal_models_from_scratch.vision.backbone import (
    FrozenBatchNorm2d,
    BasicBlock,
    Bottleneck,
    PositionEmbeddingSine,
    ResNet,
    ResNetBackbone,
    build_backbone,
)

__all__ = [
    'ImageProcessor',
    'PatchEmbedding',
    'ViTModel',
    'FrozenBatchNorm2d',
    'BasicBlock',
    'Bottleneck',
    'PositionEmbeddingSine',
    'ResNet',
    'ResNetBackbone',
    'build_backbone',
]
