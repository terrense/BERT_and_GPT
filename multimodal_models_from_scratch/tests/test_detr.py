"""
DETR 模型单元测试

测试 DETR 目标检测模型的各个组件和功能：
- 匈牙利匹配 (Hungarian Matching)
- 损失计算 (DETR Loss)
- 前向传播 (Forward Pass)

需求: 3.7, 3.8
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.config import DETRConfig
from multimodal_models_from_scratch.detection.hungarian import (
    HungarianMatcher,
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    box_iou,
    generalized_box_iou,
    box_area,
)
from multimodal_models_from_scratch.detection.losses import DETRLoss
from multimodal_models_from_scratch.detection.detr import (
    DETR,
    build_detr,
    TransformerEncoder,
    TransformerDecoder,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    MLP,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_detr_config():
    """小型 DETR 配置（用于快速测试）"""
    return DETRConfig(
        num_classes=10,
        num_queries=10,
        d_model=64,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        d_ff=128,
        dropout_rate=0.1
    )


@pytest.fixture
def detr_model(small_detr_config):
    """创建 DETR 模型实例"""
    model = DETR(small_detr_config, backbone_name='resnet18', frozen_bn=True)
    model.eval()
    return model


@pytest.fixture
def hungarian_matcher():
    """创建匈牙利匹配器"""
    return HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)


@pytest.fixture
def detr_loss(small_detr_config, hungarian_matcher):
    """创建 DETR 损失函数"""
    return DETRLoss(
        num_classes=small_detr_config.num_classes,
        matcher=hungarian_matcher,
        weight_dict={'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0},
        eos_coef=0.1
    )


@pytest.fixture
def sample_outputs(small_detr_config):
    """创建示例模型输出"""
    batch_size = 2
    num_queries = small_detr_config.num_queries
    num_classes = small_detr_config.num_classes
    
    pred_logits = torch.randn(batch_size, num_queries, num_classes + 1)
    pred_boxes = torch.sigmoid(torch.randn(batch_size, num_queries, 4))
    
    return {
        'pred_logits': pred_logits,
        'pred_boxes': pred_boxes
    }


@pytest.fixture
def sample_targets():
    """创建示例目标"""
    # 第一张图像有 3 个目标
    targets_0 = {
        'labels': torch.tensor([0, 2, 5]),
        'boxes': torch.tensor([
            [0.3, 0.4, 0.2, 0.3],  # (cx, cy, w, h)
            [0.6, 0.5, 0.3, 0.4],
            [0.2, 0.7, 0.15, 0.2]
        ])
    }
    # 第二张图像有 2 个目标
    targets_1 = {
        'labels': torch.tensor([1, 3]),
        'boxes': torch.tensor([
            [0.5, 0.5, 0.4, 0.4],
            [0.8, 0.2, 0.2, 0.3]
        ])
    }
    return [targets_0, targets_1]
