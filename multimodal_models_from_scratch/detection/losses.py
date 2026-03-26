"""
DETR Loss Functions.

This module implements the loss functions for DETR object detection model,
including classification loss, L1 bounding box loss, and GIoU loss.

Validates: Requirements 3.8, 14.3, 14.4, 14.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from .hungarian import (
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    generalized_box_iou,
)


class DETRLoss(nn.Module):
    """
    DETR Loss Function.
    
    Computes the total loss for DETR object detection, which consists of:
    - Classification loss: Cross-entropy loss for class predictions
    - L1 bounding box loss: L1 distance between predicted and target boxes
    - GIoU loss: Generalized IoU loss for better box regression
    
    The loss is computed only for matched predictions using Hungarian matching.
    Unmatched predictions are assigned to the background class.
    
    Validates: Requirements 3.8, 14.3, 14.4, 14.5
    
    Args:
        num_classes: Number of object classes (excluding background)
        matcher: HungarianMatcher instance for bipartite matching
        weight_dict: Dictionary of loss weights with keys:
            - 'loss_ce': Weight for classification loss (default: 1.0)
            - 'loss_bbox': Weight for L1 bounding box loss (default: 5.0)
            - 'loss_giou': Weight for GIoU loss (default: 2.0)
        eos_coef: Weight for the background class in classification loss (default: 0.1)
    """
    
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Optional[Dict[str, float]] = None,
        eos_coef: float = 0.1
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        
        # Default loss weights
        if weight_dict is None:
            weight_dict = {
                'loss_ce': 1.0,
                'loss_bbox': 5.0,
                'loss_giou': 2.0
            }
        self.weight_dict = weight_dict
        
        # Background class weight (lower weight for background to handle class imbalance)
        self.eos_coef = eos_coef
        
        # Create class weights: background class has lower weight
        # Background class is the last class (index = num_classes)
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the total DETR loss.
        
        Args:
            outputs: Dictionary containing:
                - 'pred_logits': (batch_size, num_queries, num_classes + 1) - class logits
                - 'pred_boxes': (batch_size, num_queries, 4) - predicted boxes in (cx, cy, w, h) format
            targets: List of dictionaries (one per image), each containing:
                - 'labels': (num_targets,) - class labels for each target
                - 'boxes': (num_targets, 4) - target boxes in (cx, cy, w, h) format
        
        Returns:
            Dictionary containing:
                - 'loss': Total weighted loss
                - 'loss_ce': Classification loss
                - 'loss_bbox': L1 bounding box loss
                - 'loss_giou': GIoU loss
        """
        # Perform Hungarian matching
        indices = self.matcher(outputs, targets)
        
        # Compute individual losses
        loss_ce = self.loss_labels(outputs, targets, indices)
        loss_bbox = self.loss_boxes(outputs, targets, indices)
        loss_giou = self.loss_giou(outputs, targets, indices)
        
        # Compute total weighted loss
        total_loss = (
            self.weight_dict.get('loss_ce', 1.0) * loss_ce +
            self.weight_dict.get('loss_bbox', 5.0) * loss_bbox +
            self.weight_dict.get('loss_giou', 2.0) * loss_giou
        )
        
        return {
            'loss': total_loss,
            'loss_ce': loss_ce,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        }
    
    def loss_labels(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute classification loss using cross-entropy.
        
        Matched predictions are assigned their target class labels.
        Unmatched predictions are assigned to the background class (num_classes).
        
        Args:
            outputs: Model outputs with 'pred_logits'
            targets: List of target dictionaries with 'labels'
            indices: Matching indices from Hungarian matcher
            
        Returns:
            Classification loss (scalar tensor)
        """
        pred_logits = outputs['pred_logits']  # (batch_size, num_queries, num_classes + 1)
        
        batch_size, num_queries = pred_logits.shape[:2]
        device = pred_logits.device
        
        # Initialize all targets as background class
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_classes,  # Background class index
            dtype=torch.int64,
            device=device
        )
        
        # Assign matched predictions to their target classes
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[batch_idx, pred_idx] = targets[batch_idx]['labels'][target_idx]
        
        # Compute cross-entropy loss with class weights
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2),  # (batch_size, num_classes + 1, num_queries)
            target_classes,  # (batch_size, num_queries)
            weight=self.empty_weight
        )
        
        return loss_ce
    
    def loss_boxes(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute L1 bounding box loss for matched predictions.
        
        Args:
            outputs: Model outputs with 'pred_boxes'
            targets: List of target dictionaries with 'boxes'
            indices: Matching indices from Hungarian matcher
            
        Returns:
            L1 bounding box loss (scalar tensor)
        """
        # Get matched predictions and targets
        pred_boxes_list = []
        target_boxes_list = []
        
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_list.append(outputs['pred_boxes'][batch_idx, pred_idx])
                target_boxes_list.append(targets[batch_idx]['boxes'][target_idx])
        
        # If no matches, return zero loss
        if len(pred_boxes_list) == 0:
            return outputs['pred_boxes'].sum() * 0.0
        
        # Concatenate all matched boxes
        pred_boxes = torch.cat(pred_boxes_list, dim=0)  # (num_matched, 4)
        target_boxes = torch.cat(target_boxes_list, dim=0)  # (num_matched, 4)
        
        # Compute L1 loss
        loss_bbox = F.l1_loss(pred_boxes, target_boxes, reduction='mean')
        
        return loss_bbox
    
    def loss_giou(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Compute GIoU loss for matched predictions.
        
        GIoU loss = 1 - GIoU, which is in range [0, 2].
        
        Args:
            outputs: Model outputs with 'pred_boxes'
            targets: List of target dictionaries with 'boxes'
            indices: Matching indices from Hungarian matcher
            
        Returns:
            GIoU loss (scalar tensor)
        """
        # Get matched predictions and targets
        pred_boxes_list = []
        target_boxes_list = []
        
        for batch_idx, (pred_idx, target_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_list.append(outputs['pred_boxes'][batch_idx, pred_idx])
                target_boxes_list.append(targets[batch_idx]['boxes'][target_idx])
        
        # If no matches, return zero loss
        if len(pred_boxes_list) == 0:
            return outputs['pred_boxes'].sum() * 0.0
        
        # Concatenate all matched boxes
        pred_boxes = torch.cat(pred_boxes_list, dim=0)  # (num_matched, 4)
        target_boxes = torch.cat(target_boxes_list, dim=0)  # (num_matched, 4)
        
        # Convert to xyxy format for GIoU computation
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        
        # Compute GIoU (diagonal elements since we want pairwise GIoU for matched pairs)
        giou_matrix = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        giou = torch.diag(giou_matrix)  # (num_matched,)
        
        # GIoU loss = 1 - GIoU
        loss_giou = (1 - giou).mean()
        
        return loss_giou
    
    def get_loss(
        self,
        loss_name: str,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Get a specific loss by name.
        
        Args:
            loss_name: One of 'labels', 'boxes', 'giou'
            outputs: Model outputs
            targets: Target dictionaries
            indices: Matching indices
            
        Returns:
            The requested loss
        """
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'giou': self.loss_giou
        }
        
        if loss_name not in loss_map:
            raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_map.keys())}")
        
        return loss_map[loss_name](outputs, targets, indices)
