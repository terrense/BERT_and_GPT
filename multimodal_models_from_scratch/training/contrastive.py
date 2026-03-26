"""
对比学习训练模块

实现 CLIP 等图文对比学习模型的训练流程。

包含：
- InfoNCE Loss 计算
- ContrastiveTrainer 类
- ContrastiveTrainingConfig 配置类

需求: 11.1, 11.2, 11.3, 11.4, 11.6, 11.7, 17.6
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert_gpt_from_scratch.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class ContrastiveTrainingConfig(TrainingConfig):
    """对比学习训练配置
    
    继承自 TrainingConfig，添加对比学习特有的配置参数。
    
    Attributes:
        temperature: 初始温度参数
        gradient_accumulation_steps: 梯度累积步数
        use_hard_negatives: 是否使用困难负样本挖掘（可选）
    """
    temperature: float = 0.07
    gradient_accumulation_steps: int = 1
    use_hard_negatives: bool = False


def info_nce_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor
) -> torch.Tensor:
    """
    计算 InfoNCE Loss（对比损失）
    
    双向对比损失：(image->text + text->image) / 2
    
    对于 batch 中的每个图文对 (i, t_i)：
    - 正样本：对角线上的配对 (i, t_i)
    - 负样本：非对角线上的配对 (i, t_j) where j != i
    
    Args:
        image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
        text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
        temperature: scalar 温度参数（可学习）
    
    Returns:
        loss: scalar InfoNCE 损失值
    
    Examples:
        >>> image_embeds = F.normalize(torch.randn(4, 512), dim=-1)
        >>> text_embeds = F.normalize(torch.randn(4, 512), dim=-1)
        >>> temperature = torch.tensor([14.28])  # 1/0.07
        >>> loss = info_nce_loss(image_embeds, text_embeds, temperature)
    """
    batch_size = image_embeds.shape[0]
    
    # 计算相似度矩阵
    # logits[i, j] = similarity(image_i, text_j) * temperature
    logits_per_image = temperature * (image_embeds @ text_embeds.T)  # (batch, batch)
    logits_per_text = logits_per_image.T  # (batch, batch)
    
    # 创建标签：对角线上的元素是正样本
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    # 图像到文本的损失
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    
    # 文本到图像的损失
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    # 双向平均
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss


def compute_contrastive_accuracy(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor
) -> Dict[str, float]:
    """
    计算对比学习的准确率
    
    Args:
        image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
        text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
        temperature: scalar 温度参数
    
    Returns:
        Dict 包含:
        - 'i2t_accuracy': 图像到文本的准确率
        - 't2i_accuracy': 文本到图像的准确率
        - 'mean_accuracy': 平均准确率
    """
    batch_size = image_embeds.shape[0]
    
    # 计算相似度矩阵
    logits_per_image = temperature * (image_embeds @ text_embeds.T)
    logits_per_text = logits_per_image.T
    
    # 创建标签
    labels = torch.arange(batch_size, device=image_embeds.device)
    
    # 计算准确率
    i2t_pred = logits_per_image.argmax(dim=-1)
    t2i_pred = logits_per_text.argmax(dim=-1)
    
    i2t_accuracy = (i2t_pred == labels).float().mean().item()
    t2i_accuracy = (t2i_pred == labels).float().mean().item()
    mean_accuracy = (i2t_accuracy + t2i_accuracy) / 2
    
    return {
        'i2t_accuracy': i2t_accuracy,
        't2i_accuracy': t2i_accuracy,
        'mean_accuracy': mean_accuracy
    }


class ContrastiveTrainer:
    """
    对比学习训练器
    
    用于训练 CLIP 等图文对比学习模型。
    
    特性：
    - InfoNCE Loss 计算
    - 可学习的温度参数
    - 梯度累积支持
    - 混合精度训练支持（可选）
    - 训练日志记录
    
    Args:
        model: CLIP 模型（或其他支持 encode_image/encode_text 的模型）
        config: ContrastiveTrainingConfig 训练配置
        tokenizer: 分词器（可选，用于数据预处理）
        image_processor: 图像预处理器（可选）
    
    Attributes:
        model: 训练的模型
        config: 训练配置
        optimizer: AdamW 优化器
        scheduler: 学习率调度器（可选）
        global_step: 全局训练步数
        accumulation_step: 当前梯度累积步数
    
    Examples:
        >>> from multimodal_models_from_scratch.multimodal.clip import CLIPModel
        >>> from multimodal_models_from_scratch.config import CLIPConfig
        >>> config = CLIPConfig()
        >>> model = CLIPModel(config)
        >>> train_config = ContrastiveTrainingConfig(
        ...     learning_rate=1e-4,
        ...     batch_size=32,
        ...     gradient_accumulation_steps=4
        ... )
        >>> trainer = ContrastiveTrainer(model, train_config)
        >>> # trainer.train(dataloader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ContrastiveTrainingConfig,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None
    ):
        """
        初始化对比学习训练器
        
        Args:
            model: CLIP 模型
            config: 训练配置
            tokenizer: 分词器（可选）
            image_processor: 图像预处理器（可选）
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器（可选，使用 warmup + cosine decay）
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.accumulation_step = 0
    
    def compute_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 InfoNCE Loss
        
        Args:
            image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
            text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
            temperature: scalar 温度参数
        
        Returns:
            loss: scalar 对比损失
        """
        return info_nce_loss(image_embeds, text_embeds, temperature)
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        单步训练
        
        执行一个 batch 的前向传播、损失计算和反向传播。
        支持梯度累积：只有在累积足够步数后才更新参数。
        
        Args:
            batch: 包含以下键的字典：
                - 'pixel_values': (batch, 3, H, W) 输入图像
                - 'input_ids': (batch, seq_len) 输入 token IDs
                - 'attention_mask': (batch, seq_len) 注意力掩码（可选）
        
        Returns:
            Dict 包含:
            - 'loss': 当前 batch 的损失值
            - 'i2t_accuracy': 图像到文本的准确率
            - 't2i_accuracy': 文本到图像的准确率
            - 'mean_accuracy': 平均准确率
            - 'temperature': 当前温度值
            - 'did_update': 是否执行了参数更新
        """
        self.model.train()
        
        # 前向传播
        outputs = self.model(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        image_embeds = outputs['image_embeds']
        text_embeds = outputs['text_embeds']
        temperature = outputs['temperature']
        
        # 计算损失
        loss = self.compute_loss(image_embeds, text_embeds, temperature)
        
        # 梯度累积：将损失除以累积步数
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 计算准确率（不需要梯度）
        with torch.no_grad():
            accuracy_metrics = compute_contrastive_accuracy(
                image_embeds, text_embeds, temperature
            )
        
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
        
        return {
            'loss': loss.item(),
            'i2t_accuracy': accuracy_metrics['i2t_accuracy'],
            't2i_accuracy': accuracy_metrics['t2i_accuracy'],
            'mean_accuracy': accuracy_metrics['mean_accuracy'],
            'temperature': temperature.item() if temperature.numel() == 1 else temperature[0].item(),
            'did_update': did_update
        }
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int
    ) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        logger.info(f"Starting contrastive learning training for {num_epochs} epochs")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_accuracy = 0.0
            num_batches = 0
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                epoch_accuracy += metrics['mean_accuracy']
                num_batches += 1
                
                # 日志记录（仅在参数更新时记录）
                if metrics['did_update'] and self.global_step % self.config.log_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"Loss={metrics['loss']:.4f}, "
                        f"Accuracy={metrics['mean_accuracy']:.4f}, "
                        f"Temperature={metrics['temperature']:.4f}"
                    )
                
                # 保存检查点
                if metrics['did_update'] and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            avg_loss = epoch_loss / num_batches
            avg_accuracy = epoch_accuracy / num_batches
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: "
                f"Average Loss={avg_loss:.4f}, "
                f"Average Accuracy={avg_accuracy:.4f}"
            )
    
    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"contrastive_step_{self.global_step}.pt"
        )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
            'model_type': 'clip'
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
