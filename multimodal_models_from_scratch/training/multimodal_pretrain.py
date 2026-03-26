"""
多模态预训练模块

实现 BLIP/BLIP-2 等多模态模型的预训练流程。

包含：
- ITC Loss (Image-Text Contrastive)
- ITM Loss (Image-Text Matching) with Hard Negative Mining
- ITG Loss (Image-grounded Text Generation)
- MultimodalPreTrainer 类
- MultimodalPreTrainingConfig 配置类

需求: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert_gpt_from_scratch.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class MultimodalPreTrainingConfig(TrainingConfig):
    """多模态预训练配置
    
    继承自 TrainingConfig，添加多模态预训练特有的配置参数。
    
    Attributes:
        lambda_itc: ITC 损失权重
        lambda_itm: ITM 损失权重
        lambda_itg: ITG 损失权重
        hard_negative_ratio: 困难负样本比例 (0.0-1.0)
        freeze_vision_encoder: 是否冻结视觉编码器
        gradient_accumulation_steps: 梯度累积步数
        temperature: 对比学习温度参数
    """
    lambda_itc: float = 1.0
    lambda_itm: float = 1.0
    lambda_itg: float = 1.0
    hard_negative_ratio: float = 0.5
    freeze_vision_encoder: bool = False
    gradient_accumulation_steps: int = 1
    temperature: float = 0.07


def compute_itc_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor
) -> torch.Tensor:
    """
    计算 Image-Text Contrastive (ITC) 损失
    
    双向对比损失：(image->text + text->image) / 2
    
    Args:
        image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
        text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
        temperature: scalar 温度参数（可学习）
    
    Returns:
        loss: scalar ITC 损失值
    """
    batch_size = image_embeds.shape[0]
    
    # 计算相似度矩阵
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


def compute_itm_loss(
    itm_logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算 Image-Text Matching (ITM) 损失
    
    二分类交叉熵损失。
    
    Args:
        itm_logits: (batch, 2) ITM logits
        labels: (batch,) 标签，1=匹配，0=不匹配
    
    Returns:
        loss: scalar ITM 损失
    """
    return F.cross_entropy(itm_logits, labels)


def compute_itg_loss(
    lm_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    计算 Image-grounded Text Generation (ITG) 损失
    
    语言模型损失（交叉熵）。
    
    Args:
        lm_logits: (batch, seq_len, vocab_size) 语言模型 logits
        labels: (batch, seq_len) 目标 token IDs
        ignore_index: 忽略的标签索引
    
    Returns:
        loss: scalar ITG 损失
    """
    # Shift logits and labels for next token prediction
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    # 计算交叉熵损失
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    
    return loss


def sample_hard_negatives(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    hard_negative_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    采样困难负样本用于 ITM 任务
    
    困难负样本是指在对比学习中相似度较高但实际不匹配的样本。
    
    策略：
    1. 计算图文相似度矩阵
    2. 对于每个图像，选择相似度最高的非匹配文本作为困难负样本
    3. 对于每个文本，选择相似度最高的非匹配图像作为困难负样本
    4. 按比例混合困难负样本和随机负样本
    
    Args:
        image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
        text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
        hard_negative_ratio: 困难负样本比例
    
    Returns:
        neg_image_indices: (batch,) 负样本图像索引
        neg_text_indices: (batch,) 负样本文本索引
        labels: (batch * 3,) ITM 标签 (正样本 + 图像负样本 + 文本负样本)
    """
    batch_size = image_embeds.shape[0]
    device = image_embeds.device
    
    # 计算相似度矩阵
    similarity = image_embeds @ text_embeds.T  # (batch, batch)
    
    # 将对角线（正样本）设为负无穷，避免选中
    mask = torch.eye(batch_size, device=device, dtype=torch.bool)
    similarity_masked = similarity.masked_fill(mask, float('-inf'))
    
    # 计算需要的困难负样本数量
    num_hard = int(batch_size * hard_negative_ratio)
    num_random = batch_size - num_hard
    
    # 对于每个图像，选择相似度最高的非匹配文本
    # hard_text_indices[i] = 与图像 i 最相似的非匹配文本索引
    hard_text_indices = similarity_masked.argmax(dim=1)  # (batch,)
    
    # 对于每个文本，选择相似度最高的非匹配图像
    # hard_image_indices[i] = 与文本 i 最相似的非匹配图像索引
    hard_image_indices = similarity_masked.argmax(dim=0)  # (batch,)
    
    # 生成随机负样本索引
    if num_random > 0:
        # 随机选择不同的索引
        random_text_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        random_image_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            # 排除自身
            candidates = [j for j in range(batch_size) if j != i]
            random_text_indices[i] = candidates[torch.randint(len(candidates), (1,)).item()]
            random_image_indices[i] = candidates[torch.randint(len(candidates), (1,)).item()]
    else:
        random_text_indices = hard_text_indices
        random_image_indices = hard_image_indices
    
    # 混合困难负样本和随机负样本
    # 前 num_hard 个使用困难负样本，后 num_random 个使用随机负样本
    neg_text_indices = torch.cat([
        hard_text_indices[:num_hard],
        random_text_indices[num_hard:]
    ]) if num_hard < batch_size else hard_text_indices
    
    neg_image_indices = torch.cat([
        hard_image_indices[:num_hard],
        random_image_indices[num_hard:]
    ]) if num_hard < batch_size else hard_image_indices
    
    # 创建 ITM 标签
    # 正样本: batch 个 1
    # 图像负样本（错误文本）: batch 个 0
    # 文本负样本（错误图像）: batch 个 0
    labels = torch.cat([
        torch.ones(batch_size, dtype=torch.long, device=device),   # 正样本
        torch.zeros(batch_size, dtype=torch.long, device=device),  # 图像负样本
        torch.zeros(batch_size, dtype=torch.long, device=device)   # 文本负样本
    ])
    
    return neg_image_indices, neg_text_indices, labels


def compute_itc_accuracy(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor
) -> Dict[str, float]:
    """
    计算 ITC 准确率
    
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


def compute_itm_accuracy(
    itm_logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    计算 ITM 准确率
    
    Args:
        itm_logits: (batch, 2) ITM logits
        labels: (batch,) 标签
    
    Returns:
        accuracy: ITM 准确率
    """
    predictions = itm_logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return accuracy



class MultimodalPreTrainer:
    """
    多模态预训练器
    
    用于训练 BLIP/BLIP-2 等多模态模型。
    
    特性：
    - ITC Loss (Image-Text Contrastive)
    - ITM Loss (Image-Text Matching) with Hard Negative Mining
    - ITG Loss (Image-grounded Text Generation)
    - 多任务联合训练
    - 可配置的损失权重
    - 梯度累积支持
    - 训练日志记录
    
    Args:
        model: BLIP 或 BLIP-2 模型
        config: MultimodalPreTrainingConfig 训练配置
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
        >>> from multimodal_models_from_scratch.multimodal.blip import BLIPModel
        >>> from multimodal_models_from_scratch.config import BLIPConfig
        >>> config = BLIPConfig()
        >>> model = BLIPModel(config)
        >>> train_config = MultimodalPreTrainingConfig(
        ...     learning_rate=1e-4,
        ...     batch_size=32,
        ...     lambda_itc=1.0,
        ...     lambda_itm=1.0,
        ...     lambda_itg=1.0
        ... )
        >>> trainer = MultimodalPreTrainer(model, train_config)
        >>> # trainer.train(dataloader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: MultimodalPreTrainingConfig,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None
    ):
        """
        初始化多模态预训练器
        
        Args:
            model: BLIP 或 BLIP-2 模型
            config: 训练配置
            tokenizer: 分词器（可选）
            image_processor: 图像预处理器（可选）
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # 冻结视觉编码器（如果配置）
        if config.freeze_vision_encoder:
            self._freeze_vision_encoder()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器（可选）
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.accumulation_step = 0
    
    def _freeze_vision_encoder(self):
        """冻结视觉编码器的所有参数"""
        if hasattr(self.model, 'vision_encoder'):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
    
    def compute_itc_loss(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        temperature: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 ITC 损失
        
        Args:
            image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
            text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
            temperature: scalar 温度参数（可选，默认使用配置值）
        
        Returns:
            loss: scalar ITC 损失
        """
        if temperature is None:
            temperature = torch.tensor(
                [1.0 / self.config.temperature],
                device=image_embeds.device
            )
        return compute_itc_loss(image_embeds, text_embeds, temperature)
    
    def compute_itm_loss(
        self,
        itm_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 ITM 损失
        
        Args:
            itm_logits: (batch, 2) ITM logits
            labels: (batch,) 标签
        
        Returns:
            loss: scalar ITM 损失
        """
        return compute_itm_loss(itm_logits, labels)
    
    def compute_itg_loss(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 ITG 损失
        
        Args:
            lm_logits: (batch, seq_len, vocab_size) 语言模型 logits
            labels: (batch, seq_len) 目标 token IDs
        
        Returns:
            loss: scalar ITG 损失
        """
        return compute_itg_loss(lm_logits, labels)
    
    def sample_hard_negatives(
        self,
        image_embeds: torch.Tensor,
        text_embeds: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样困难负样本用于 ITM
        
        Args:
            image_embeds: (batch, embed_dim) L2 归一化后的图像嵌入
            text_embeds: (batch, embed_dim) L2 归一化后的文本嵌入
        
        Returns:
            neg_image_indices: (batch,) 负样本图像索引
            neg_text_indices: (batch,) 负样本文本索引
            labels: (batch * 3,) ITM 标签
        """
        return sample_hard_negatives(
            image_embeds,
            text_embeds,
            self.config.hard_negative_ratio
        )
    
    def _forward_blip(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        BLIP 模型前向传播
        
        Args:
            batch: 包含 pixel_values, input_ids, attention_mask, labels 的字典
        
        Returns:
            Dict 包含各项损失和中间结果
        """
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        output = {}
        
        # 1. ITC 前向传播
        itc_output = self.model.forward_itc(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        image_embeds = itc_output['image_embeds']
        text_embeds = itc_output['text_embeds']
        logits_per_image = itc_output['logits_per_image']
        logits_per_text = itc_output['logits_per_text']
        
        # 计算 ITC 损失
        itc_labels = torch.arange(batch_size, device=device)
        itc_loss = (
            F.cross_entropy(logits_per_image, itc_labels) +
            F.cross_entropy(logits_per_text, itc_labels)
        ) / 2
        output['itc_loss'] = itc_loss
        output['image_embeds'] = image_embeds
        output['text_embeds'] = text_embeds
        
        # 2. ITM 前向传播（使用困难负样本）
        with torch.no_grad():
            neg_image_indices, neg_text_indices, itm_labels = self.sample_hard_negatives(
                image_embeds, text_embeds
            )
        
        # 构建 ITM 输入
        # 正样本：原始图文对
        # 图像负样本：原始图像 + 错误文本
        # 文本负样本：错误图像 + 原始文本
        
        # 正样本
        pos_itm_logits = self.model.forward_itm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 图像负样本（原始图像 + 错误文本）
        neg_text_input_ids = input_ids[neg_text_indices]
        neg_text_attention_mask = attention_mask[neg_text_indices] if attention_mask is not None else None
        neg_text_itm_logits = self.model.forward_itm(
            pixel_values=pixel_values,
            input_ids=neg_text_input_ids,
            attention_mask=neg_text_attention_mask
        )
        
        # 文本负样本（错误图像 + 原始文本）
        neg_image_pixel_values = pixel_values[neg_image_indices]
        neg_image_itm_logits = self.model.forward_itm(
            pixel_values=neg_image_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 合并 ITM logits
        all_itm_logits = torch.cat([
            pos_itm_logits,
            neg_text_itm_logits,
            neg_image_itm_logits
        ], dim=0)
        
        # 计算 ITM 损失
        itm_loss = self.compute_itm_loss(all_itm_logits, itm_labels)
        output['itm_loss'] = itm_loss
        output['itm_logits'] = all_itm_logits
        output['itm_labels'] = itm_labels
        
        # 3. ITG 前向传播（如果提供 labels）
        if labels is not None:
            itg_loss = self.model.forward_itg(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            output['itg_loss'] = itg_loss
        
        return output
    
    def _forward_blip2(
        self,
        batch: Dict[str, torch.Tensor],
        stage: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        BLIP-2 模型前向传播
        
        Args:
            batch: 包含 pixel_values, input_ids, attention_mask, labels 的字典
            stage: 训练阶段 (1 或 2)
        
        Returns:
            Dict 包含各项损失和中间结果
        """
        pixel_values = batch['pixel_values']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch.get('labels')
        
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        output = {}
        
        if stage == 1:
            # 第一阶段：训练 Q-Former (ITC, ITM, ITG)
            stage1_output = self.model.forward_stage1(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                itg_labels=labels
            )
            
            output['itc_loss'] = stage1_output['itc_loss']
            output['image_embeds'] = stage1_output['image_embeds']
            output['text_embeds'] = stage1_output['text_embeds']
            
            # ITM with hard negative mining
            with torch.no_grad():
                neg_image_indices, neg_text_indices, itm_labels = self.sample_hard_negatives(
                    stage1_output['image_embeds'],
                    stage1_output['text_embeds']
                )
            
            # 正样本 ITM logits 已在 stage1_output 中
            pos_itm_logits = stage1_output['itm_logits']
            
            # 图像负样本
            neg_text_input_ids = input_ids[neg_text_indices]
            neg_text_attention_mask = attention_mask[neg_text_indices] if attention_mask is not None else None
            neg_text_output = self.model.forward_stage1(
                pixel_values=pixel_values,
                input_ids=neg_text_input_ids,
                attention_mask=neg_text_attention_mask
            )
            neg_text_itm_logits = neg_text_output['itm_logits']
            
            # 文本负样本
            neg_image_pixel_values = pixel_values[neg_image_indices]
            neg_image_output = self.model.forward_stage1(
                pixel_values=neg_image_pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            neg_image_itm_logits = neg_image_output['itm_logits']
            
            # 合并 ITM logits
            all_itm_logits = torch.cat([
                pos_itm_logits,
                neg_text_itm_logits,
                neg_image_itm_logits
            ], dim=0)
            
            itm_loss = self.compute_itm_loss(all_itm_logits, itm_labels)
            output['itm_loss'] = itm_loss
            output['itm_logits'] = all_itm_logits
            output['itm_labels'] = itm_labels
            
            if 'itg_loss' in stage1_output:
                output['itg_loss'] = stage1_output['itg_loss']
        
        else:
            # 第二阶段：训练 Visual Projection
            if labels is None:
                raise ValueError("Labels are required for stage 2 training")
            
            lm_loss = self.model.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            output['lm_loss'] = lm_loss
        
        return output
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        stage: int = 1
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
                - 'labels': (batch, seq_len) 目标 token IDs（可选，用于 ITG）
            stage: 训练阶段 (1 或 2，仅用于 BLIP-2)
        
        Returns:
            Dict 包含:
            - 'itc_loss': ITC 损失值
            - 'itm_loss': ITM 损失值
            - 'itg_loss': ITG 损失值（如果有）
            - 'total_loss': 总损失值
            - 'itc_accuracy': ITC 准确率
            - 'itm_accuracy': ITM 准确率
            - 'did_update': 是否执行了参数更新
        """
        self.model.train()
        
        # 判断模型类型并执行前向传播
        model_name = self.model.__class__.__name__
        
        if 'BLIP2' in model_name:
            output = self._forward_blip2(batch, stage)
        else:
            output = self._forward_blip(batch)
        
        # 计算总损失
        total_loss = torch.tensor(0.0, device=batch['pixel_values'].device)
        
        if 'itc_loss' in output:
            total_loss = total_loss + self.config.lambda_itc * output['itc_loss']
        
        if 'itm_loss' in output:
            total_loss = total_loss + self.config.lambda_itm * output['itm_loss']
        
        if 'itg_loss' in output:
            total_loss = total_loss + self.config.lambda_itg * output['itg_loss']
        
        if 'lm_loss' in output:
            total_loss = total_loss + output['lm_loss']
        
        # 梯度累积：将损失除以累积步数
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 计算准确率（不需要梯度）
        metrics = {}
        with torch.no_grad():
            if 'image_embeds' in output and 'text_embeds' in output:
                temperature = torch.tensor(
                    [1.0 / self.config.temperature],
                    device=output['image_embeds'].device
                )
                itc_acc = compute_itc_accuracy(
                    output['image_embeds'],
                    output['text_embeds'],
                    temperature
                )
                metrics['itc_i2t_accuracy'] = itc_acc['i2t_accuracy']
                metrics['itc_t2i_accuracy'] = itc_acc['t2i_accuracy']
                metrics['itc_accuracy'] = itc_acc['mean_accuracy']
            
            if 'itm_logits' in output and 'itm_labels' in output:
                metrics['itm_accuracy'] = compute_itm_accuracy(
                    output['itm_logits'],
                    output['itm_labels']
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
        
        # 构建返回的 metrics
        result = {
            'total_loss': total_loss.item(),
            'did_update': did_update
        }
        
        if 'itc_loss' in output:
            result['itc_loss'] = output['itc_loss'].item()
        
        if 'itm_loss' in output:
            result['itm_loss'] = output['itm_loss'].item()
        
        if 'itg_loss' in output:
            result['itg_loss'] = output['itg_loss'].item()
        
        if 'lm_loss' in output:
            result['lm_loss'] = output['lm_loss'].item()
        
        result.update(metrics)
        
        return result
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        stage: int = 1
    ) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            stage: 训练阶段 (1 或 2，仅用于 BLIP-2)
        """
        logger.info(f"Starting multimodal pretraining for {num_epochs} epochs (stage {stage})")
        logger.info(f"Loss weights: ITC={self.config.lambda_itc}, ITM={self.config.lambda_itm}, ITG={self.config.lambda_itg}")
        logger.info(f"Hard negative ratio: {self.config.hard_negative_ratio}")
        logger.info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'total_loss': 0.0,
                'itc_loss': 0.0,
                'itm_loss': 0.0,
                'itg_loss': 0.0
            }
            epoch_accuracies = {
                'itc_accuracy': 0.0,
                'itm_accuracy': 0.0
            }
            num_batches = 0
            
            for batch in dataloader:
                metrics = self.train_step(batch, stage)
                
                epoch_losses['total_loss'] += metrics['total_loss']
                if 'itc_loss' in metrics:
                    epoch_losses['itc_loss'] += metrics['itc_loss']
                if 'itm_loss' in metrics:
                    epoch_losses['itm_loss'] += metrics['itm_loss']
                if 'itg_loss' in metrics:
                    epoch_losses['itg_loss'] += metrics['itg_loss']
                
                if 'itc_accuracy' in metrics:
                    epoch_accuracies['itc_accuracy'] += metrics['itc_accuracy']
                if 'itm_accuracy' in metrics:
                    epoch_accuracies['itm_accuracy'] += metrics['itm_accuracy']
                
                num_batches += 1
                
                # 日志记录（仅在参数更新时记录）
                if metrics['did_update'] and self.global_step % self.config.log_steps == 0:
                    log_msg = f"Step {self.global_step}: Total Loss={metrics['total_loss']:.4f}"
                    if 'itc_loss' in metrics:
                        log_msg += f", ITC={metrics['itc_loss']:.4f}"
                    if 'itm_loss' in metrics:
                        log_msg += f", ITM={metrics['itm_loss']:.4f}"
                    if 'itg_loss' in metrics:
                        log_msg += f", ITG={metrics['itg_loss']:.4f}"
                    if 'itc_accuracy' in metrics:
                        log_msg += f", ITC Acc={metrics['itc_accuracy']:.4f}"
                    if 'itm_accuracy' in metrics:
                        log_msg += f", ITM Acc={metrics['itm_accuracy']:.4f}"
                    logger.info(log_msg)
                
                # 保存检查点
                if metrics['did_update'] and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Epoch 统计
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            avg_accuracies = {k: v / num_batches for k, v in epoch_accuracies.items()}
            
            log_msg = f"Epoch {epoch + 1}/{num_epochs}: Total Loss={avg_losses['total_loss']:.4f}"
            if avg_losses['itc_loss'] > 0:
                log_msg += f", ITC={avg_losses['itc_loss']:.4f}"
            if avg_losses['itm_loss'] > 0:
                log_msg += f", ITM={avg_losses['itm_loss']:.4f}"
            if avg_losses['itg_loss'] > 0:
                log_msg += f", ITG={avg_losses['itg_loss']:.4f}"
            if avg_accuracies['itc_accuracy'] > 0:
                log_msg += f", ITC Acc={avg_accuracies['itc_accuracy']:.4f}"
            if avg_accuracies['itm_accuracy'] > 0:
                log_msg += f", ITM Acc={avg_accuracies['itm_accuracy']:.4f}"
            logger.info(log_msg)
    
    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"multimodal_pretrain_step_{self.global_step}.pt"
        )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config,
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
