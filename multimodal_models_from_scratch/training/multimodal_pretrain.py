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
