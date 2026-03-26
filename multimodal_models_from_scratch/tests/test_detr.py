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


# ============================================================================
# 测试边界框格式转换
# ============================================================================

class TestBoxConversions:
    """测试边界框格式转换函数"""
    
    def test_cxcywh_to_xyxy_single_box(self):
        """测试单个边界框从 cxcywh 转换为 xyxy"""
        # (cx=0.5, cy=0.5, w=0.4, h=0.6) -> (x1=0.3, y1=0.2, x2=0.7, y2=0.8)
        boxes_cxcywh = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
        
        expected = torch.tensor([[0.3, 0.2, 0.7, 0.8]])
        assert torch.allclose(boxes_xyxy, expected, atol=1e-5)
    
    def test_xyxy_to_cxcywh_single_box(self):
        """测试单个边界框从 xyxy 转换为 cxcywh"""
        # (x1=0.3, y1=0.2, x2=0.7, y2=0.8) -> (cx=0.5, cy=0.5, w=0.4, h=0.6)
        boxes_xyxy = torch.tensor([[0.3, 0.2, 0.7, 0.8]])
        boxes_cxcywh = box_xyxy_to_cxcywh(boxes_xyxy)
        
        expected = torch.tensor([[0.5, 0.5, 0.4, 0.6]])
        assert torch.allclose(boxes_cxcywh, expected, atol=1e-5)
    
    def test_roundtrip_conversion(self):
        """测试往返转换的一致性"""
        boxes_original = torch.tensor([
            [0.3, 0.4, 0.2, 0.3],
            [0.6, 0.5, 0.3, 0.4],
            [0.2, 0.7, 0.15, 0.2]
        ])
        
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_original)
        boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
        
        assert torch.allclose(boxes_original, boxes_back, atol=1e-5)
    
    def test_batch_conversion(self):
        """测试批量边界框转换"""
        batch_size = 4
        num_boxes = 10
        boxes_cxcywh = torch.rand(batch_size, num_boxes, 4)
        
        boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
        boxes_back = box_xyxy_to_cxcywh(boxes_xyxy)
        
        assert boxes_xyxy.shape == (batch_size, num_boxes, 4)
        assert torch.allclose(boxes_cxcywh, boxes_back, atol=1e-5)


# ============================================================================
# 测试 IoU 和 GIoU 计算
# ============================================================================

class TestIoUComputation:
    """测试 IoU 和 GIoU 计算"""
    
    def test_box_area(self):
        """测试边界框面积计算"""
        # (x1=0, y1=0, x2=1, y2=1) -> area = 1
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        areas = box_area(boxes)
        
        assert torch.allclose(areas, torch.tensor([1.0]))
    
    def test_box_area_multiple(self):
        """测试多个边界框面积计算"""
        boxes = torch.tensor([
            [0.0, 0.0, 1.0, 1.0],  # area = 1
            [0.0, 0.0, 0.5, 0.5],  # area = 0.25
            [0.2, 0.3, 0.6, 0.8],  # area = 0.4 * 0.5 = 0.2
        ])
        areas = box_area(boxes)
        
        expected = torch.tensor([1.0, 0.25, 0.2])
        assert torch.allclose(areas, expected, atol=1e-5)
    
    def test_iou_identical_boxes(self):
        """测试相同边界框的 IoU（应为 1）"""
        boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        
        iou = box_iou(boxes1, boxes2)
        
        assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-5)
    
    def test_iou_no_overlap(self):
        """测试无重叠边界框的 IoU（应接近 0）"""
        boxes1 = torch.tensor([[0.0, 0.0, 0.3, 0.3]])
        boxes2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou[0, 0].item() < 1e-5
    
    def test_iou_partial_overlap(self):
        """测试部分重叠边界框的 IoU"""
        boxes1 = torch.tensor([[0.0, 0.0, 0.5, 0.5]])  # area = 0.25
        boxes2 = torch.tensor([[0.25, 0.25, 0.75, 0.75]])  # area = 0.25
        # intersection: (0.25, 0.25) to (0.5, 0.5) = 0.25 * 0.25 = 0.0625
        # union: 0.25 + 0.25 - 0.0625 = 0.4375
        # iou: 0.0625 / 0.4375 ≈ 0.143
        
        iou = box_iou(boxes1, boxes2)
        
        expected_iou = 0.0625 / 0.4375
        assert torch.allclose(iou, torch.tensor([[expected_iou]]), atol=1e-3)
    
    def test_iou_pairwise(self):
        """测试成对 IoU 计算"""
        boxes1 = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, 1.0]
        ])
        boxes2 = torch.tensor([
            [0.0, 0.0, 0.5, 0.5],
            [0.25, 0.25, 0.75, 0.75],
            [0.5, 0.5, 1.0, 1.0]
        ])
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou.shape == (2, 3)
        # boxes1[0] 与 boxes2[0] 相同，IoU = 1
        assert torch.allclose(iou[0, 0], torch.tensor(1.0), atol=1e-5)
        # boxes1[1] 与 boxes2[2] 相同，IoU = 1
        assert torch.allclose(iou[1, 2], torch.tensor(1.0), atol=1e-5)

    
    def test_giou_identical_boxes(self):
        """测试相同边界框的 GIoU（应为 1）"""
        boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        boxes2 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        assert torch.allclose(giou, torch.tensor([[1.0]]), atol=1e-5)
    
    def test_giou_no_overlap(self):
        """测试无重叠边界框的 GIoU（应小于 0）"""
        boxes1 = torch.tensor([[0.0, 0.0, 0.2, 0.2]])
        boxes2 = torch.tensor([[0.8, 0.8, 1.0, 1.0]])
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        # GIoU 对于无重叠的框应该小于 0
        assert giou[0, 0].item() < 0
    
    def test_giou_range(self):
        """测试 GIoU 值范围在 [-1, 1]"""
        boxes1 = torch.rand(10, 4)
        boxes2 = torch.rand(10, 4)
        
        # 确保有效的边界框格式
        boxes1[:, 2:] = boxes1[:, :2] + torch.abs(boxes1[:, 2:]) + 0.1
        boxes2[:, 2:] = boxes2[:, :2] + torch.abs(boxes2[:, 2:]) + 0.1
        
        giou = generalized_box_iou(boxes1, boxes2)
        
        assert (giou >= -1.0 - 1e-5).all()
        assert (giou <= 1.0 + 1e-5).all()


# ============================================================================
# 测试匈牙利匹配
# ============================================================================

class TestHungarianMatcher:
    """测试匈牙利匹配器"""
    
    def test_matcher_initialization(self):
        """测试匹配器初始化"""
        matcher = HungarianMatcher(cost_class=1.0, cost_bbox=5.0, cost_giou=2.0)
        
        assert matcher.cost_class == 1.0
        assert matcher.cost_bbox == 5.0
        assert matcher.cost_giou == 2.0
    
    def test_matcher_output_format(self, hungarian_matcher, sample_outputs, sample_targets):
        """测试匹配器输出格式"""
        indices = hungarian_matcher(sample_outputs, sample_targets)
        
        # 应该返回与 batch_size 相同数量的匹配结果
        assert len(indices) == 2
        
        # 每个匹配结果是 (pred_indices, target_indices) 元组
        for pred_idx, target_idx in indices:
            assert isinstance(pred_idx, torch.Tensor)
            assert isinstance(target_idx, torch.Tensor)
            assert pred_idx.shape == target_idx.shape
    
    def test_matcher_indices_valid(self, hungarian_matcher, sample_outputs, sample_targets):
        """测试匹配索引的有效性"""
        indices = hungarian_matcher(sample_outputs, sample_targets)
        
        num_queries = sample_outputs['pred_logits'].shape[1]
        
        for i, (pred_idx, target_idx) in enumerate(indices):
            num_targets = len(sample_targets[i]['labels'])
            
            # 预测索引应该在有效范围内
            if len(pred_idx) > 0:
                assert (pred_idx >= 0).all()
                assert (pred_idx < num_queries).all()
            
            # 目标索引应该在有效范围内
            if len(target_idx) > 0:
                assert (target_idx >= 0).all()
                assert (target_idx < num_targets).all()
            
            # 匹配数量不应超过目标数量
            assert len(pred_idx) <= num_targets

    
    def test_matcher_with_empty_targets(self, hungarian_matcher, sample_outputs):
        """测试空目标的匹配"""
        # 创建空目标
        empty_targets = [
            {'labels': torch.tensor([], dtype=torch.int64), 
             'boxes': torch.zeros(0, 4)},
            {'labels': torch.tensor([], dtype=torch.int64), 
             'boxes': torch.zeros(0, 4)}
        ]
        
        indices = hungarian_matcher(sample_outputs, empty_targets)
        
        # 空目标应该返回空匹配
        for pred_idx, target_idx in indices:
            assert len(pred_idx) == 0
            assert len(target_idx) == 0
    
    def test_matcher_with_single_target(self, hungarian_matcher):
        """测试单个目标的匹配"""
        batch_size = 1
        num_queries = 5
        num_classes = 10
        
        outputs = {
            'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
            'pred_boxes': torch.sigmoid(torch.randn(batch_size, num_queries, 4))
        }
        
        targets = [{
            'labels': torch.tensor([3]),
            'boxes': torch.tensor([[0.5, 0.5, 0.3, 0.3]])
        }]
        
        indices = hungarian_matcher(outputs, targets)
        
        assert len(indices) == 1
        pred_idx, target_idx = indices[0]
        assert len(pred_idx) == 1
        assert len(target_idx) == 1
        assert target_idx[0] == 0  # 唯一的目标索引
    
    def test_matcher_with_various_batch_sizes(self, hungarian_matcher):
        """测试不同批次大小的匹配"""
        for batch_size in [1, 2, 4, 8]:
            num_queries = 10
            num_classes = 10
            
            outputs = {
                'pred_logits': torch.randn(batch_size, num_queries, num_classes + 1),
                'pred_boxes': torch.sigmoid(torch.randn(batch_size, num_queries, 4))
            }
            
            # 每张图像有不同数量的目标
            targets = []
            for i in range(batch_size):
                num_targets = (i % 5) + 1  # 1 到 5 个目标
                targets.append({
                    'labels': torch.randint(0, num_classes, (num_targets,)),
                    'boxes': torch.rand(num_targets, 4)
                })
            
            indices = hungarian_matcher(outputs, targets)
            
            assert len(indices) == batch_size
    
    def test_cost_matrix_computation(self, hungarian_matcher):
        """测试代价矩阵计算"""
        num_queries = 5
        num_targets = 3
        num_classes = 10
        
        pred_logits = torch.randn(num_queries, num_classes + 1)
        pred_boxes = torch.sigmoid(torch.randn(num_queries, 4))
        target_labels = torch.randint(0, num_classes, (num_targets,))
        target_boxes = torch.rand(num_targets, 4)
        
        cost_matrix = hungarian_matcher.compute_cost_matrix(
            pred_logits, pred_boxes, target_labels, target_boxes
        )
        
        assert cost_matrix.shape == (num_queries, num_targets)


# ============================================================================
# 测试 DETR 损失函数
# ============================================================================

class TestDETRLoss:
    """测试 DETR 损失函数"""
    
    def test_loss_initialization(self, small_detr_config, hungarian_matcher):
        """测试损失函数初始化"""
        loss_fn = DETRLoss(
            num_classes=small_detr_config.num_classes,
            matcher=hungarian_matcher
        )
        
        assert loss_fn.num_classes == small_detr_config.num_classes
        assert loss_fn.matcher is hungarian_matcher
    
    def test_loss_output_format(self, detr_loss, sample_outputs, sample_targets):
        """测试损失函数输出格式"""
        losses = detr_loss(sample_outputs, sample_targets)
        
        assert 'loss' in losses
        assert 'loss_ce' in losses
        assert 'loss_bbox' in losses
        assert 'loss_giou' in losses
        
        # 所有损失应该是标量
        assert losses['loss'].dim() == 0
        assert losses['loss_ce'].dim() == 0
        assert losses['loss_bbox'].dim() == 0
        assert losses['loss_giou'].dim() == 0
    
    def test_loss_positive_values(self, detr_loss, sample_outputs, sample_targets):
        """测试损失值为正"""
        losses = detr_loss(sample_outputs, sample_targets)
        
        assert losses['loss'].item() >= 0
        assert losses['loss_ce'].item() >= 0
        assert losses['loss_bbox'].item() >= 0
        assert losses['loss_giou'].item() >= 0
    
    def test_classification_loss(self, detr_loss, sample_outputs, sample_targets):
        """测试分类损失计算"""
        indices = detr_loss.matcher(sample_outputs, sample_targets)
        loss_ce = detr_loss.loss_labels(sample_outputs, sample_targets, indices)
        
        assert loss_ce.dim() == 0
        assert loss_ce.item() >= 0
    
    def test_bbox_loss(self, detr_loss, sample_outputs, sample_targets):
        """测试 L1 边界框损失计算"""
        indices = detr_loss.matcher(sample_outputs, sample_targets)
        loss_bbox = detr_loss.loss_boxes(sample_outputs, sample_targets, indices)
        
        assert loss_bbox.dim() == 0
        assert loss_bbox.item() >= 0
    
    def test_giou_loss(self, detr_loss, sample_outputs, sample_targets):
        """测试 GIoU 损失计算"""
        indices = detr_loss.matcher(sample_outputs, sample_targets)
        loss_giou = detr_loss.loss_giou(sample_outputs, sample_targets, indices)
        
        assert loss_giou.dim() == 0
        assert loss_giou.item() >= 0

    
    def test_loss_with_empty_targets(self, detr_loss, sample_outputs):
        """测试空目标的损失计算"""
        empty_targets = [
            {'labels': torch.tensor([], dtype=torch.int64), 
             'boxes': torch.zeros(0, 4)},
            {'labels': torch.tensor([], dtype=torch.int64), 
             'boxes': torch.zeros(0, 4)}
        ]
        
        losses = detr_loss(sample_outputs, empty_targets)
        
        # 空目标时损失应该仍然有效
        assert 'loss' in losses
        assert not torch.isnan(losses['loss'])
        assert not torch.isinf(losses['loss'])
    
    def test_total_loss_weighted_sum(self, small_detr_config, hungarian_matcher, sample_outputs, sample_targets):
        """测试总损失是加权和"""
        weight_dict = {'loss_ce': 2.0, 'loss_bbox': 3.0, 'loss_giou': 1.0}
        
        loss_fn = DETRLoss(
            num_classes=small_detr_config.num_classes,
            matcher=hungarian_matcher,
            weight_dict=weight_dict
        )
        
        losses = loss_fn(sample_outputs, sample_targets)
        
        expected_total = (
            weight_dict['loss_ce'] * losses['loss_ce'] +
            weight_dict['loss_bbox'] * losses['loss_bbox'] +
            weight_dict['loss_giou'] * losses['loss_giou']
        )
        
        assert torch.allclose(losses['loss'], expected_total, atol=1e-5)
    
    def test_get_loss_method(self, detr_loss, sample_outputs, sample_targets):
        """测试 get_loss 方法"""
        indices = detr_loss.matcher(sample_outputs, sample_targets)
        
        loss_labels = detr_loss.get_loss('labels', sample_outputs, sample_targets, indices)
        loss_boxes = detr_loss.get_loss('boxes', sample_outputs, sample_targets, indices)
        loss_giou = detr_loss.get_loss('giou', sample_outputs, sample_targets, indices)
        
        assert loss_labels.dim() == 0
        assert loss_boxes.dim() == 0
        assert loss_giou.dim() == 0
    
    def test_background_class_weight(self, small_detr_config, hungarian_matcher):
        """测试背景类权重"""
        eos_coef = 0.1
        loss_fn = DETRLoss(
            num_classes=small_detr_config.num_classes,
            matcher=hungarian_matcher,
            eos_coef=eos_coef
        )
        
        # 检查 empty_weight 的最后一个元素（背景类）
        assert abs(loss_fn.empty_weight[-1].item() - eos_coef) < 1e-5
        # 其他类的权重应该为 1
        assert (loss_fn.empty_weight[:-1] == 1.0).all()


# ============================================================================
# 测试 DETR 模型
# ============================================================================

class TestDETRModel:
    """测试 DETR 模型"""
    
    def test_model_initialization(self, small_detr_config):
        """测试模型初始化"""
        model = DETR(small_detr_config, backbone_name='resnet18')
        
        assert model.num_classes == small_detr_config.num_classes
        assert model.num_queries == small_detr_config.num_queries
        assert model.d_model == small_detr_config.d_model
    
    def test_model_components_exist(self, detr_model):
        """测试模型包含所有必要组件"""
        # 需求 3.1: CNN backbone
        assert hasattr(detr_model, 'backbone')
        
        # 需求 3.2: Transformer Encoder
        assert hasattr(detr_model, 'encoder')
        
        # 需求 3.3: Transformer Decoder
        assert hasattr(detr_model, 'decoder')
        
        # 需求 3.4: Object Queries
        assert hasattr(detr_model, 'query_embed')
        
        # 需求 3.5: Classification Head
        assert hasattr(detr_model, 'class_head')
        
        # 需求 3.6: Bounding Box Head
        assert hasattr(detr_model, 'bbox_head')
    
    def test_forward_output_shape(self, detr_model, small_detr_config):
        """测试前向传播输出形状"""
        batch_size = 2
        image_size = 224
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        with torch.no_grad():
            outputs = detr_model(pixel_values)
        
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        
        # pred_logits: (batch, num_queries, num_classes + 1)
        assert outputs['pred_logits'].shape == (
            batch_size, 
            small_detr_config.num_queries, 
            small_detr_config.num_classes + 1
        )
        
        # pred_boxes: (batch, num_queries, 4)
        assert outputs['pred_boxes'].shape == (
            batch_size, 
            small_detr_config.num_queries, 
            4
        )
    
    def test_forward_with_different_image_sizes(self, detr_model, small_detr_config):
        """测试不同图像尺寸的前向传播"""
        batch_size = 2
        
        for image_size in [224, 256, 320]:
            pixel_values = torch.randn(batch_size, 3, image_size, image_size)
            
            with torch.no_grad():
                outputs = detr_model(pixel_values)
            
            # 输出形状应该与图像尺寸无关
            assert outputs['pred_logits'].shape == (
                batch_size, 
                small_detr_config.num_queries, 
                small_detr_config.num_classes + 1
            )
            assert outputs['pred_boxes'].shape == (
                batch_size, 
                small_detr_config.num_queries, 
                4
            )

    
    def test_forward_with_padding_mask(self, detr_model, small_detr_config):
        """测试带 padding mask 的前向传播"""
        batch_size = 2
        image_size = 224
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        # 创建 padding mask（True 表示被掩码的位置）
        mask = torch.zeros(batch_size, image_size, image_size, dtype=torch.bool)
        mask[:, :, image_size//2:] = True  # 掩码右半部分
        
        with torch.no_grad():
            outputs = detr_model(pixel_values, mask=mask)
        
        assert 'pred_logits' in outputs
        assert 'pred_boxes' in outputs
        assert outputs['pred_logits'].shape[0] == batch_size
        assert outputs['pred_boxes'].shape[0] == batch_size
    
    def test_pred_boxes_normalized(self, detr_model):
        """测试预测边界框是否归一化到 [0, 1]"""
        batch_size = 2
        image_size = 224
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        with torch.no_grad():
            outputs = detr_model(pixel_values)
        
        pred_boxes = outputs['pred_boxes']
        
        # 边界框坐标应该在 [0, 1] 范围内（经过 sigmoid）
        assert (pred_boxes >= 0).all()
        assert (pred_boxes <= 1).all()
    
    def test_different_backbone_configurations(self, small_detr_config):
        """测试不同 backbone 配置"""
        batch_size = 1
        image_size = 224
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        for backbone_name in ['resnet18', 'resnet50']:
            model = DETR(small_detr_config, backbone_name=backbone_name)
            model.eval()
            
            with torch.no_grad():
                outputs = model(pixel_values)
            
            assert outputs['pred_logits'].shape == (
                batch_size, 
                small_detr_config.num_queries, 
                small_detr_config.num_classes + 1
            )
    
    def test_different_num_queries(self):
        """测试不同 num_queries 配置"""
        batch_size = 1
        image_size = 224
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        for num_queries in [10, 50, 100]:
            config = DETRConfig(
                num_classes=10,
                num_queries=num_queries,
                d_model=64,
                num_heads=4,
                num_encoder_layers=2,
                num_decoder_layers=2,
                d_ff=128,
                dropout_rate=0.1
            )
            model = DETR(config, backbone_name='resnet18')
            model.eval()
            
            with torch.no_grad():
                outputs = model(pixel_values)
            
            assert outputs['pred_logits'].shape[1] == num_queries
            assert outputs['pred_boxes'].shape[1] == num_queries

    
    def test_freeze_backbone(self, detr_model):
        """测试冻结 backbone"""
        detr_model.freeze_backbone()
        
        for param in detr_model.backbone.parameters():
            assert not param.requires_grad
    
    def test_unfreeze_backbone(self, detr_model):
        """测试解冻 backbone"""
        detr_model.freeze_backbone()
        detr_model.unfreeze_backbone()
        
        for param in detr_model.backbone.parameters():
            assert param.requires_grad
    
    def test_get_num_parameters(self, detr_model):
        """测试获取参数数量"""
        num_params = detr_model.get_num_parameters()
        
        assert num_params > 0
        assert isinstance(num_params, int)


class TestBuildDETR:
    """测试 build_detr 辅助函数"""
    
    def test_build_detr_default_config(self):
        """测试使用默认配置构建 DETR"""
        model = build_detr()
        
        assert isinstance(model, DETR)
        # 默认配置
        assert model.num_classes == 91  # COCO 类别数
        assert model.num_queries == 100
    
    def test_build_detr_custom_config(self, small_detr_config):
        """测试使用自定义配置构建 DETR"""
        model = build_detr(config=small_detr_config, backbone_name='resnet18')
        
        assert model.num_classes == small_detr_config.num_classes
        assert model.num_queries == small_detr_config.num_queries
    
    def test_build_detr_different_backbones(self, small_detr_config):
        """测试使用不同 backbone 构建 DETR"""
        for backbone_name in ['resnet18', 'resnet50']:
            model = build_detr(config=small_detr_config, backbone_name=backbone_name)
            assert isinstance(model, DETR)


# ============================================================================
# 测试 Transformer 组件
# ============================================================================

class TestTransformerComponents:
    """测试 Transformer 编码器和解码器组件"""
    
    def test_encoder_layer_output_shape(self):
        """测试编码器层输出形状"""
        batch_size = 2
        seq_len = 49
        d_model = 64
        
        layer = TransformerEncoderLayer(d_model=d_model, num_heads=4, d_ff=128)
        
        src = torch.randn(batch_size, seq_len, d_model)
        pos_embed = torch.randn(batch_size, seq_len, d_model)
        
        output = layer(src, pos_embed)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_encoder_layer_with_mask(self):
        """测试带掩码的编码器层"""
        batch_size = 2
        seq_len = 49
        d_model = 64
        
        layer = TransformerEncoderLayer(d_model=d_model, num_heads=4, d_ff=128)
        
        src = torch.randn(batch_size, seq_len, d_model)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        mask[:, -10:] = True  # 掩码最后 10 个位置
        
        output = layer(src, src_key_padding_mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_decoder_layer_output_shape(self):
        """测试解码器层输出形状"""
        batch_size = 2
        num_queries = 10
        seq_len = 49
        d_model = 64
        
        layer = TransformerDecoderLayer(d_model=d_model, num_heads=4, d_ff=128)
        
        tgt = torch.randn(batch_size, num_queries, d_model)
        memory = torch.randn(batch_size, seq_len, d_model)
        query_pos = torch.randn(batch_size, num_queries, d_model)
        memory_pos = torch.randn(batch_size, seq_len, d_model)
        
        output = layer(tgt, memory, query_pos, memory_pos)
        
        assert output.shape == (batch_size, num_queries, d_model)
    
    def test_encoder_output_shape(self):
        """测试完整编码器输出形状"""
        batch_size = 2
        seq_len = 49
        d_model = 64
        
        encoder = TransformerEncoder(
            num_layers=2, d_model=d_model, num_heads=4, d_ff=128
        )
        
        src = torch.randn(batch_size, seq_len, d_model)
        pos_embed = torch.randn(batch_size, seq_len, d_model)
        
        output = encoder(src, pos_embed)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_decoder_output_shape(self):
        """测试完整解码器输出形状"""
        batch_size = 2
        num_queries = 10
        seq_len = 49
        d_model = 64
        
        decoder = TransformerDecoder(
            num_layers=2, d_model=d_model, num_heads=4, d_ff=128
        )
        
        tgt = torch.randn(batch_size, num_queries, d_model)
        memory = torch.randn(batch_size, seq_len, d_model)
        query_pos = torch.randn(batch_size, num_queries, d_model)
        memory_pos = torch.randn(batch_size, seq_len, d_model)
        
        output = decoder(tgt, memory, query_pos, memory_pos)
        
        assert output.shape == (batch_size, num_queries, d_model)


class TestMLP:
    """测试 MLP 模块"""
    
    def test_mlp_output_shape(self):
        """测试 MLP 输出形状"""
        batch_size = 2
        num_queries = 10
        input_dim = 64
        hidden_dim = 128
        output_dim = 4
        
        mlp = MLP(input_dim, hidden_dim, output_dim, num_layers=3)
        
        x = torch.randn(batch_size, num_queries, input_dim)
        output = mlp(x)
        
        assert output.shape == (batch_size, num_queries, output_dim)
    
    def test_mlp_different_num_layers(self):
        """测试不同层数的 MLP"""
        input_dim = 64
        hidden_dim = 128
        output_dim = 4
        
        for num_layers in [1, 2, 3, 4]:
            mlp = MLP(input_dim, hidden_dim, output_dim, num_layers=num_layers)
            
            x = torch.randn(2, 10, input_dim)
            output = mlp(x)
            
            assert output.shape == (2, 10, output_dim)


# ============================================================================
# 测试梯度流
# ============================================================================

class TestDETRGradients:
    """测试 DETR 模型梯度流"""
    
    def test_gradients_flow(self, small_detr_config, hungarian_matcher):
        """测试梯度是否正确流动"""
        model = DETR(small_detr_config, backbone_name='resnet18')
        loss_fn = DETRLoss(
            num_classes=small_detr_config.num_classes,
            matcher=hungarian_matcher
        )
        
        batch_size = 2
        image_size = 224
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size, requires_grad=True)
        
        targets = [
            {'labels': torch.tensor([0, 2]), 'boxes': torch.rand(2, 4)},
            {'labels': torch.tensor([1]), 'boxes': torch.rand(1, 4)}
        ]
        
        outputs = model(pixel_values)
        losses = loss_fn(outputs, targets)
        losses['loss'].backward()
        
        # 检查梯度是否存在
        assert pixel_values.grad is not None
        assert model.query_embed.weight.grad is not None
        assert model.class_head.weight.grad is not None
    
    def test_backbone_gradients_when_unfrozen(self, small_detr_config, hungarian_matcher):
        """测试解冻 backbone 时的梯度"""
        model = DETR(small_detr_config, backbone_name='resnet18')
        model.unfreeze_backbone()
        
        loss_fn = DETRLoss(
            num_classes=small_detr_config.num_classes,
            matcher=hungarian_matcher
        )
        
        batch_size = 1
        image_size = 224
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        targets = [{'labels': torch.tensor([0]), 'boxes': torch.rand(1, 4)}]
        
        outputs = model(pixel_values)
        losses = loss_fn(outputs, targets)
        losses['loss'].backward()
        
        # 检查 backbone 梯度
        backbone_has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0 
            for p in model.backbone.parameters() if p.requires_grad
        )
        assert backbone_has_grad
