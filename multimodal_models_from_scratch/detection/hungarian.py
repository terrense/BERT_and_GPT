"""
Hungarian Matching for DETR.

This module implements the Hungarian matching algorithm for optimal bipartite
matching between predictions and ground truth targets in object detection.

Validates: Requirements 3.7, 14.1, 14.2
"""

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Dict


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).
    
    Args:
        boxes: Tensor of shape (..., 4) in (cx, cy, w, h) format
        
    Returns:
        Tensor of shape (..., 4) in (x1, y1, x2, y2) format
    """
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from corner format (x1, y1, x2, y2) to center format (cx, cy, w, h).
    
    Args:
        boxes: Tensor of shape (..., 4) in (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape (..., 4) in (cx, cy, w, h) format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Compute the area of bounding boxes.
    
    Args:
        boxes: Tensor of shape (..., 4) in (x1, y1, x2, y2) format
        
    Returns:
        Tensor of shape (...) containing box areas
    """
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute IoU (Intersection over Union) between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format
        
    Returns:
        iou: Tensor of shape (N, M) containing pairwise IoU values
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute union
    union = area1[:, None] + area2[None, :] - inter  # (N, M)
    
    # Compute IoU
    iou = inter / (union + 1e-6)
    
    return iou


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Compute Generalized IoU (GIoU) between two sets of boxes.
    
    GIoU = IoU - (area of enclosing box - union) / area of enclosing box
    
    Reference: https://arxiv.org/abs/1902.09630
    
    Args:
        boxes1: Tensor of shape (N, 4) in (x1, y1, x2, y2) format
        boxes2: Tensor of shape (M, 4) in (x1, y1, x2, y2) format
        
    Returns:
        giou: Tensor of shape (N, M) containing pairwise GIoU values
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)
    
    # Compute intersection
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    
    wh = (rb - lt).clamp(min=0)  # (N, M, 2)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)
    
    # Compute union
    union = area1[:, None] + area2[None, :] - inter  # (N, M)
    
    # Compute IoU
    iou = inter / (union + 1e-6)
    
    # Compute enclosing box
    enclose_lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    enclose_rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)
    
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)  # (N, M, 2)
    enclose_area = enclose_wh[:, :, 0] * enclose_wh[:, :, 1]  # (N, M)
    
    # Compute GIoU
    giou = iou - (enclose_area - union) / (enclose_area + 1e-6)
    
    return giou


class HungarianMatcher(nn.Module):
    """
    Hungarian Matcher for DETR.
    
    Computes optimal bipartite matching between predictions and ground truth
    targets using the Hungarian algorithm (scipy.optimize.linear_sum_assignment).
    
    The cost matrix is computed as a weighted sum of:
    - Classification cost: negative log probability of the target class
    - L1 bounding box cost: L1 distance between predicted and target boxes
    - GIoU cost: 1 - GIoU between predicted and target boxes
    
    Validates: Requirements 3.7, 14.1, 14.2
    
    Args:
        cost_class: Weight for classification cost (default: 1.0)
        cost_bbox: Weight for L1 bounding box cost (default: 5.0)
        cost_giou: Weight for GIoU cost (default: 2.0)
    """
    
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        
        # Ensure at least one cost is non-zero
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, \
            "At least one cost weight must be non-zero"
    
    @torch.no_grad()
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute optimal bipartite matching between predictions and targets.
        
        Args:
            outputs: Dictionary containing:
                - 'pred_logits': (batch_size, num_queries, num_classes) - class logits
                - 'pred_boxes': (batch_size, num_queries, 4) - predicted boxes in (cx, cy, w, h) format
            targets: List of dictionaries (one per image), each containing:
                - 'labels': (num_targets,) - class labels for each target
                - 'boxes': (num_targets, 4) - target boxes in (cx, cy, w, h) format
        
        Returns:
            List of tuples (pred_indices, target_indices) for each batch element,
            where pred_indices and target_indices are 1D tensors of matching indices.
        """
        batch_size, num_queries = outputs['pred_logits'].shape[:2]
        
        # Flatten predictions across batch for efficient computation
        # (batch_size * num_queries, num_classes)
        pred_logits = outputs['pred_logits'].flatten(0, 1)
        # (batch_size * num_queries, 4)
        pred_boxes = outputs['pred_boxes'].flatten(0, 1)
        
        # Apply softmax to get class probabilities
        pred_probs = pred_logits.softmax(-1)
        
        # Concatenate target labels and boxes
        target_labels = torch.cat([t['labels'] for t in targets])
        target_boxes = torch.cat([t['boxes'] for t in targets])
        
        # Compute classification cost
        # Cost is negative log probability of the target class
        # Shape: (batch_size * num_queries, total_num_targets)
        cost_class = -pred_probs[:, target_labels]
        
        # Compute L1 bounding box cost
        # Shape: (batch_size * num_queries, total_num_targets)
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        
        # Compute GIoU cost
        # Convert boxes to xyxy format for GIoU computation
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        # Shape: (batch_size * num_queries, total_num_targets)
        cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        
        # Compute total cost matrix
        cost_matrix = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        
        # Reshape cost matrix to (batch_size, num_queries, total_num_targets)
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1)
        
        # Get number of targets per image
        sizes = [len(t['labels']) for t in targets]
        
        # Perform Hungarian matching for each image in the batch
        indices = []
        cost_matrix_cpu = cost_matrix.cpu()
        
        for i, c in enumerate(cost_matrix_cpu.split(sizes, -1)):
            # c has shape (num_queries, num_targets_i)
            c_i = c[i]  # Get cost matrix for image i
            
            if c_i.shape[1] == 0:
                # No targets in this image
                indices.append((
                    torch.tensor([], dtype=torch.int64),
                    torch.tensor([], dtype=torch.int64)
                ))
            else:
                # Use scipy's linear_sum_assignment for Hungarian algorithm
                pred_idx, target_idx = linear_sum_assignment(c_i.numpy())
                indices.append((
                    torch.as_tensor(pred_idx, dtype=torch.int64),
                    torch.as_tensor(target_idx, dtype=torch.int64)
                ))
        
        return indices
    
    def compute_cost_matrix(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        target_labels: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cost matrix for a single image.
        
        This is a helper method for computing the cost matrix without batching.
        
        Args:
            pred_logits: (num_queries, num_classes) - class logits
            pred_boxes: (num_queries, 4) - predicted boxes in (cx, cy, w, h) format
            target_labels: (num_targets,) - class labels for each target
            target_boxes: (num_targets, 4) - target boxes in (cx, cy, w, h) format
            
        Returns:
            cost_matrix: (num_queries, num_targets) - total cost matrix
        """
        # Apply softmax to get class probabilities
        pred_probs = pred_logits.softmax(-1)
        
        # Classification cost: negative log probability of target class
        cost_class = -pred_probs[:, target_labels]
        
        # L1 bounding box cost
        cost_bbox = torch.cdist(pred_boxes, target_boxes, p=1)
        
        # GIoU cost
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
        target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes)
        cost_giou = -generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)
        
        # Total cost
        cost_matrix = (
            self.cost_class * cost_class +
            self.cost_bbox * cost_bbox +
            self.cost_giou * cost_giou
        )
        
        return cost_matrix
