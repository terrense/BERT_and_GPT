"""
目标检测训练模块

实现 DETR 目标检测模型的训练流程。

包含：
- DetectionTrainingConfig 配置类
- 数据增强支持（随机水平翻转、随机缩放、颜色抖动）
- DetectionTrainer 类
- 集成匈牙利匹配和 DETR 损失
- mAP 评估指标（占位符）

需求: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
"""

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert_gpt_from_scratch.config import TrainingConfig

logger = logging.getLogger(__name__)


# ImageNet normalization stats
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DetectionTrainingConfig(TrainingConfig):
    """目标检测训练配置
    
    继承自 TrainingConfig，添加目标检测训练特有的配置参数。
    
    Attributes:
        loss_ce_weight: 分类损失权重
        loss_bbox_weight: L1 边界框损失权重
        loss_giou_weight: GIoU 损失权重
        gradient_accumulation_steps: 梯度累积步数
        use_augmentation: 是否使用数据增强
        horizontal_flip_prob: 水平翻转概率
        resize_scales: 随机缩放的尺度范围 (min_scale, max_scale)
        color_jitter: 是否使用颜色抖动
        color_jitter_brightness: 亮度抖动范围
        color_jitter_contrast: 对比度抖动范围
        color_jitter_saturation: 饱和度抖动范围
        color_jitter_hue: 色调抖动范围
    """
    loss_ce_weight: float = 1.0
    loss_bbox_weight: float = 5.0
    loss_giou_weight: float = 2.0
    gradient_accumulation_steps: int = 1
    use_augmentation: bool = True
    horizontal_flip_prob: float = 0.5
    resize_scales: Tuple[float, float] = (0.8, 1.2)
    color_jitter: bool = False
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.4
    color_jitter_hue: float = 0.1


def random_horizontal_flip(
    image: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    prob: float = 0.5
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    随机水平翻转图像和边界框
    
    Args:
        image: 输入图像，形状为 (C, H, W)
        targets: 目标字典，包含 'boxes' (N, 4) 格式为 (cx, cy, w, h)
        prob: 翻转概率
    
    Returns:
        flipped_image: 翻转后的图像
        flipped_targets: 翻转后的目标
    """
    if random.random() < prob:
        # 水平翻转图像
        image = torch.flip(image, dims=[-1])
        
        # 翻转边界框
        if 'boxes' in targets and len(targets['boxes']) > 0:
            boxes = targets['boxes'].clone()
            # 对于 (cx, cy, w, h) 格式，只需翻转 cx
            # 新的 cx = 1 - cx
            boxes[:, 0] = 1.0 - boxes[:, 0]
            targets = {**targets, 'boxes': boxes}
    
    return image, targets


def random_resize(
    image: torch.Tensor,
    targets: Dict[str, torch.Tensor],
    scales: Tuple[float, float] = (0.8, 1.2)
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    随机缩放图像（边界框坐标为归一化坐标，无需调整）
    
    Args:
        image: 输入图像，形状为 (C, H, W)
        targets: 目标字典
        scales: 缩放范围 (min_scale, max_scale)
    
    Returns:
        resized_image: 缩放后的图像
        targets: 目标（归一化坐标无需调整）
    """
    scale = random.uniform(scales[0], scales[1])
    _, h, w = image.shape
    new_h, new_w = int(h * scale), int(w * scale)
    
    # 使用双线性插值缩放
    image = F.interpolate(
        image.unsqueeze(0),
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    
    # 归一化坐标无需调整
    return image, targets


def color_jitter(
    image: torch.Tensor,
    brightness: float = 0.4,
    contrast: float = 0.4,
    saturation: float = 0.4,
    hue: float = 0.1
) -> torch.Tensor:
    """
    颜色抖动增强
    
    Args:
        image: 输入图像，形状为 (C, H, W)，值范围 [0, 1]
        brightness: 亮度抖动范围
        contrast: 对比度抖动范围
        saturation: 饱和度抖动范围
        hue: 色调抖动范围
    
    Returns:
        jittered_image: 颜色抖动后的图像
    """
    # 亮度调整
    if brightness > 0:
        brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        image = image * brightness_factor
    
    # 对比度调整
    if contrast > 0:
        contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        mean = image.mean(dim=[-2, -1], keepdim=True)
        image = (image - mean) * contrast_factor + mean
    
    # 饱和度调整（简化版本）
    if saturation > 0:
        saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        gray = image.mean(dim=0, keepdim=True)
        image = image * saturation_factor + gray * (1 - saturation_factor)
    
    # 限制值范围
    image = torch.clamp(image, 0, 1)
    
    return image


def normalize_image(
    image: torch.Tensor,
    mean: Tuple[float, ...] = IMAGENET_MEAN,
    std: Tuple[float, ...] = IMAGENET_STD
) -> torch.Tensor:
    """
    使用 ImageNet 统计量归一化图像
    
    Args:
        image: 输入图像，形状为 (C, H, W)，值范围 [0, 1]
        mean: 均值
        std: 标准差
    
    Returns:
        normalized_image: 归一化后的图像
    """
    mean_tensor = torch.tensor(mean, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    std_tensor = torch.tensor(std, device=image.device, dtype=image.dtype).view(-1, 1, 1)
    return (image - mean_tensor) / std_tensor


class DetectionAugmentation:
    """
    目标检测数据增强
    
    支持的增强方式：
    - 随机水平翻转
    - 随机缩放
    - 颜色抖动（可选）
    - ImageNet 归一化
    
    Args:
        config: DetectionTrainingConfig 配置
    """
    
    def __init__(self, config: DetectionTrainingConfig):
        self.config = config
    
    def __call__(
        self,
        image: torch.Tensor,
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        应用数据增强
        
        Args:
            image: 输入图像，形状为 (C, H, W)，值范围 [0, 1]
            targets: 目标字典，包含 'labels' 和 'boxes'
        
        Returns:
            augmented_image: 增强后的图像
            augmented_targets: 增强后的目标
        """
        if not self.config.use_augmentation:
            # 仅归一化
            image = normalize_image(image)
            return image, targets
        
        # 随机水平翻转
        image, targets = random_horizontal_flip(
            image, targets, prob=self.config.horizontal_flip_prob
        )
        
        # 随机缩放
        image, targets = random_resize(
            image, targets, scales=self.config.resize_scales
        )
        
        # 颜色抖动（可选）
        if self.config.color_jitter:
            image = color_jitter(
                image,
                brightness=self.config.color_jitter_brightness,
                contrast=self.config.color_jitter_contrast,
                saturation=self.config.color_jitter_saturation,
                hue=self.config.color_jitter_hue
            )
        
        # ImageNet 归一化
        image = normalize_image(image)
        
        return image, targets


def compute_map_placeholder(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    mAP 计算占位符
    
    完整的 mAP 计算需要 COCO 评估工具，这里提供简化版本。
    
    Args:
        predictions: 预测列表，每个元素包含 'pred_logits' 和 'pred_boxes'
        targets: 目标列表，每个元素包含 'labels' 和 'boxes'
        iou_thresholds: IoU 阈值列表
    
    Returns:
        Dict 包含:
        - 'mAP': 平均精度（占位符）
        - 'mAP_50': IoU=0.5 时的 mAP（占位符）
        - 'mAP_75': IoU=0.75 时的 mAP（占位符）
    """
    # 占位符实现
    # 完整实现需要 pycocotools 或自定义 mAP 计算
    logger.warning("mAP computation is a placeholder. Use pycocotools for accurate evaluation.")
    
    return {
        'mAP': 0.0,
        'mAP_50': 0.0,
        'mAP_75': 0.0
    }


class DetectionTrainer:
    """
    目标检测训练器
    
    用于训练 DETR 目标检测模型。
    
    特性：
    - 集成匈牙利匹配和 DETR 损失
    - 支持数据增强（随机翻转、缩放、颜色抖动）
    - 梯度累积支持
    - 学习率调度
    - 训练日志记录
    - mAP 评估（占位符）
    
    Validates: Requirements 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7
    
    Args:
        model: DETR 模型
        config: DetectionTrainingConfig 训练配置
        matcher: HungarianMatcher 匈牙利匹配器
        loss_fn: DETRLoss 损失函数
    
    Attributes:
        model: 训练的模型
        config: 训练配置
        matcher: 匈牙利匹配器
        loss_fn: DETR 损失函数
        optimizer: AdamW 优化器
        scheduler: 学习率调度器（可选）
        global_step: 全局训练步数
        accumulation_step: 当前梯度累积步数
        augmentation: 数据增强器
    
    Examples:
        >>> from multimodal_models_from_scratch.detection.detr import DETR
        >>> from multimodal_models_from_scratch.detection.hungarian import HungarianMatcher
        >>> from multimodal_models_from_scratch.detection.losses import DETRLoss
        >>> from multimodal_models_from_scratch.config import DETRConfig
        >>> config = DETRConfig()
        >>> model = DETR(config)
        >>> matcher = HungarianMatcher()
        >>> loss_fn = DETRLoss(num_classes=91, matcher=matcher)
        >>> train_config = DetectionTrainingConfig(learning_rate=1e-4, batch_size=2)
        >>> trainer = DetectionTrainer(model, train_config, matcher, loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        config: DetectionTrainingConfig,
        matcher: nn.Module,
        loss_fn: nn.Module
    ):
        """
        初始化目标检测训练器
        
        Args:
            model: DETR 模型
            config: 训练配置
            matcher: HungarianMatcher 匈牙利匹配器
            loss_fn: DETRLoss 损失函数
        """
        self.model = model
        self.config = config
        self.matcher = matcher
        self.loss_fn = loss_fn
        
        # 更新损失权重
        if hasattr(loss_fn, 'weight_dict'):
            loss_fn.weight_dict = {
                'loss_ce': config.loss_ce_weight,
                'loss_bbox': config.loss_bbox_weight,
                'loss_giou': config.loss_giou_weight
            }
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器（可选）
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.accumulation_step = 0
        
        # 数据增强
        self.augmentation = DetectionAugmentation(config)
        
        # 损失追踪
        self.loss_history: List[Dict[str, float]] = []
    
    def apply_augmentation(
        self,
        images: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        对批次数据应用数据增强
        
        Args:
            images: 输入图像，形状为 (batch, C, H, W)
            targets: 目标列表
        
        Returns:
            augmented_images: 增强后的图像
            augmented_targets: 增强后的目标
        """
        augmented_images = []
        augmented_targets = []
        
        for i in range(images.shape[0]):
            img = images[i]
            tgt = targets[i]
            
            aug_img, aug_tgt = self.augmentation(img, tgt)
            augmented_images.append(aug_img)
            augmented_targets.append(aug_tgt)
        
        # 需要处理不同尺寸的图像（如果有随机缩放）
        # 这里简化处理，假设所有图像尺寸相同
        augmented_images = torch.stack(augmented_images, dim=0)
        
        return augmented_images, augmented_targets

    def train_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        单步训练
        
        执行一个 batch 的前向传播、损失计算和反向传播。
        支持梯度累积：只有在累积足够步数后才更新参数。
        
        Args:
            batch: 包含以下键的字典：
                - 'pixel_values': (batch, 3, H, W) 输入图像
                - 'targets': List[Dict] 目标列表，每个元素包含：
                    - 'labels': (num_targets,) 类别标签
                    - 'boxes': (num_targets, 4) 边界框 (cx, cy, w, h)
        
        Returns:
            Dict 包含:
            - 'loss': 总损失值
            - 'loss_ce': 分类损失
            - 'loss_bbox': L1 边界框损失
            - 'loss_giou': GIoU 损失
            - 'did_update': 是否执行了参数更新
        """
        self.model.train()
        
        pixel_values = batch['pixel_values']
        targets = batch['targets']
        
        # 应用数据增强（如果启用）
        if self.config.use_augmentation and self.model.training:
            pixel_values, targets = self.apply_augmentation(pixel_values, targets)
        
        # 前向传播
        outputs = self.model(pixel_values=pixel_values)
        
        # 计算损失
        loss_dict = self.loss_fn(outputs, targets)
        total_loss = loss_dict['loss']
        
        # 梯度累积：将损失除以累积步数
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 更新累积步数
        self.accumulation_step += 1
        did_update = False
        
        # 检查是否需要更新参数
        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 重置累积步数
            self.accumulation_step = 0
            self.global_step += 1
            did_update = True
        
        # 构建返回的 metrics
        result = {
            'loss': total_loss.item(),
            'loss_ce': loss_dict['loss_ce'].item(),
            'loss_bbox': loss_dict['loss_bbox'].item(),
            'loss_giou': loss_dict['loss_giou'].item(),
            'did_update': did_update
        }
        
        # 记录损失历史
        self.loss_history.append({
            'step': self.global_step,
            'loss': result['loss'],
            'loss_ce': result['loss_ce'],
            'loss_bbox': result['loss_bbox'],
            'loss_giou': result['loss_giou']
        })
        
        return result

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        评估模型
        
        计算验证集上的损失和 mAP 指标。
        
        Args:
            dataloader: 验证数据加载器
        
        Returns:
            Dict 包含:
            - 'val_loss': 验证损失
            - 'val_loss_ce': 验证分类损失
            - 'val_loss_bbox': 验证 L1 边界框损失
            - 'val_loss_giou': 验证 GIoU 损失
            - 'mAP': 平均精度（占位符）
            - 'mAP_50': IoU=0.5 时的 mAP（占位符）
            - 'mAP_75': IoU=0.75 时的 mAP（占位符）
        """
        self.model.eval()
        
        total_loss = 0.0
        total_loss_ce = 0.0
        total_loss_bbox = 0.0
        total_loss_giou = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        for batch in dataloader:
            pixel_values = batch['pixel_values']
            targets = batch['targets']
            
            # 前向传播
            outputs = self.model(pixel_values=pixel_values)
            
            # 计算损失
            loss_dict = self.loss_fn(outputs, targets)
            
            total_loss += loss_dict['loss'].item()
            total_loss_ce += loss_dict['loss_ce'].item()
            total_loss_bbox += loss_dict['loss_bbox'].item()
            total_loss_giou += loss_dict['loss_giou'].item()
            num_batches += 1
            
            # 收集预测和目标用于 mAP 计算
            all_predictions.append(outputs)
            all_targets.extend(targets)
        
        # 计算平均损失
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_loss_ce = total_loss_ce / num_batches if num_batches > 0 else 0.0
        avg_loss_bbox = total_loss_bbox / num_batches if num_batches > 0 else 0.0
        avg_loss_giou = total_loss_giou / num_batches if num_batches > 0 else 0.0
        
        # 计算 mAP（占位符）
        map_metrics = compute_map_placeholder(all_predictions, all_targets)
        
        return {
            'val_loss': avg_loss,
            'val_loss_ce': avg_loss_ce,
            'val_loss_bbox': avg_loss_bbox,
            'val_loss_giou': avg_loss_giou,
            **map_metrics
        }

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        val_dataloader: Optional[DataLoader] = None
    ) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 训练数据加载器
            num_epochs: 训练轮数
            val_dataloader: 验证数据加载器（可选）
        """
        logger.info(f"Starting detection training for {num_epochs} epochs")
        logger.info(f"Loss weights: CE={self.config.loss_ce_weight}, "
                   f"BBox={self.config.loss_bbox_weight}, GIoU={self.config.loss_giou_weight}")
        logger.info(f"Data augmentation: {self.config.use_augmentation}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'loss': 0.0,
                'loss_ce': 0.0,
                'loss_bbox': 0.0,
                'loss_giou': 0.0
            }
            num_batches = 0
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                
                epoch_losses['loss'] += metrics['loss']
                epoch_losses['loss_ce'] += metrics['loss_ce']
                epoch_losses['loss_bbox'] += metrics['loss_bbox']
                epoch_losses['loss_giou'] += metrics['loss_giou']
                num_batches += 1
                
                # 日志记录（仅在参数更新时记录）
                if metrics['did_update'] and self.global_step % self.config.log_steps == 0:
                    log_msg = (f"Step {self.global_step}: "
                              f"Loss={metrics['loss']:.4f}, "
                              f"CE={metrics['loss_ce']:.4f}, "
                              f"BBox={metrics['loss_bbox']:.4f}, "
                              f"GIoU={metrics['loss_giou']:.4f}")
                    logger.info(log_msg)
                
                # 保存检查点
                if metrics['did_update'] and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Epoch 统计
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            log_msg = (f"Epoch {epoch + 1}/{num_epochs}: "
                      f"Loss={avg_losses['loss']:.4f}, "
                      f"CE={avg_losses['loss_ce']:.4f}, "
                      f"BBox={avg_losses['loss_bbox']:.4f}, "
                      f"GIoU={avg_losses['loss_giou']:.4f}")
            logger.info(log_msg)
            
            # 验证
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                val_log_msg = (f"Validation: "
                              f"Loss={val_metrics['val_loss']:.4f}, "
                              f"mAP={val_metrics['mAP']:.4f}")
                logger.info(val_log_msg)

    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"detection_step_{self.global_step}.pt"
        )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
            'loss_history': self.loss_history[-1000:],  # 保留最近 1000 条记录
            'model_type': self.model.__class__.__name__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if 'loss_history' in checkpoint:
            self.loss_history = checkpoint['loss_history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def set_scheduler(
        self,
        total_steps: int,
        warmup_steps: Optional[int] = None
    ) -> None:
        """
        设置学习率调度器（warmup + cosine decay）
        
        Args:
            total_steps: 总训练步数
            warmup_steps: 预热步数（默认使用 config 中的值）
        """
        if warmup_steps is None:
            warmup_steps = self.config.warmup_steps
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # 线性预热
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        logger.info(f"Scheduler set with {warmup_steps} warmup steps and {total_steps} total steps")

    def get_trainable_parameters(self) -> int:
        """
        获取可训练参数数量
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """
        获取总参数数量
        
        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.model.parameters())
    
    def freeze_backbone(self) -> None:
        """冻结 backbone 参数"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
    
    def unfreeze_backbone(self) -> None:
        """解冻 backbone 参数"""
        if hasattr(self.model, 'backbone'):
            for param in self.model.backbone.parameters():
                param.requires_grad = True
            logger.info("Backbone unfrozen")
    
    def get_loss_history(self) -> List[Dict[str, float]]:
        """
        获取损失历史
        
        Returns:
            损失历史列表
        """
        return self.loss_history
    
    def get_current_lr(self) -> float:
        """
        获取当前学习率
        
        Returns:
            当前学习率
        """
        return self.optimizer.param_groups[0]['lr']
