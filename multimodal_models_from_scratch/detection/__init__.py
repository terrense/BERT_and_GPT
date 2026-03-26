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

__all__ = [
    'HungarianMatcher',
    'box_cxcywh_to_xyxy',
    'box_xyxy_to_cxcywh',
    'box_area',
    'box_iou',
    'generalized_box_iou',
]
