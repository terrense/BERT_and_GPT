"""
Object detection components.

Modules:
- detr: DETR model
- hungarian: Hungarian Matching algorithm
- losses: Detection loss functions
"""

from .hungarian import (
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_area,
    box_iou,
    generalized_box_iou,
)
from .losses import DETRLoss
from .detr import (
    DETR,
    build_detr,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    MLP,
)

__all__ = [
    # Hungarian matching
    'HungarianMatcher',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_area',
    'box_iou',
    'generalized_box_iou',
    # Loss
    'DETRLoss',
    # DETR model
    'DETR',
    'build_detr',
    'TransformerEncoder',
    'TransformerDecoder',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'MLP',
]
