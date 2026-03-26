"""
预训练 Trainer 实现

包含：
- MLM 数据预处理
- NSP 数据预处理
- NWP 数据预处理
- BERTPreTrainer
- GPTPreTrainer
"""

import logging
import os
import random
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import TrainingConfig
from ..models.bert import BERTModel
from ..models.gpt import GPTModel
from ..tokenizer.simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


# ==================== MLM 数据预处理 ====================

def prepare_mlm_data(
    input_ids: torch.Tensor,
    tokenizer: SimpleTokenizer,
    mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备 MLM 训练数据
    
    随机选择 15% 的 token 进行掩码处理：
    - 80% 概率替换为 [MASK]
    - 10% 概率替换为随机 token
    - 10% 概率保持不变
    
    Args:
        input_ids: (batch, seq_len) 原始 token IDs
        tokenizer: Tokenizer 实例
        mask_prob: 掩码概率，默认 0.15
    
    Returns:
        masked_input_ids: 掩码后的输入
        mlm_labels: MLM 标签（非掩码位置为 -100）
        mlm_mask: 掩码位置的布尔张量
    """
    masked_input_ids = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, -100)
    
    # 特殊 token 不参与掩码
    special_token_ids = {
        tokenizer.pad_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    }
    
    # 创建可掩码位置的 mask
    can_mask = torch.ones_like(input_ids, dtype=torch.bool)
    for special_id in special_token_ids:
        can_mask &= (input_ids != special_id)
    
    # 随机选择掩码位置
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_positions = (rand < mask_prob) & can_mask
    
    # 设置 MLM 标签
    mlm_labels[mask_positions] = input_ids[mask_positions]
    
    # 对掩码位置进行处理
    # 80% 替换为 [MASK]
    mask_token_positions = mask_positions & (torch.rand_like(rand) < 0.8)
    masked_input_ids[mask_token_positions] = tokenizer.mask_token_id
    
    # 10% 替换为随机 token（在剩余 20% 中的 50%）
    remaining_positions = mask_positions & ~mask_token_positions
    random_token_positions = remaining_positions & (torch.rand_like(rand) < 0.5)
    random_tokens = torch.randint(
        tokenizer.NUM_SPECIAL_TOKENS,  # 跳过特殊 token
        tokenizer.vocab_size,
        random_token_positions.sum().unsqueeze(0)
    )
    masked_input_ids[random_token_positions] = random_tokens.squeeze()
    
    # 剩余 10% 保持不变（不需要额外处理）
    
    return masked_input_ids, mlm_labels, mask_positions


# ==================== NSP 数据预处理 ====================

def prepare_nsp_data(
    sentences: List[str],
    tokenizer: SimpleTokenizer,
    max_length: int = 512
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    准备 NSP 训练数据
    
    50% 概率选择真实下一句，50% 概率选择随机句子。
    
    Args:
        sentences: 句子列表
        tokenizer: Tokenizer 实例
        max_length: 最大序列长度
    
    Returns:
        input_ids: 拼接后的句子对 (batch, seq_len)
        segment_ids: 段落标识 (batch, seq_len)
        nsp_labels: NSP 标签 (batch,)，0=真实下一句，1=随机句子
    """
    batch_input_ids = []
    batch_segment_ids = []
    batch_nsp_labels = []
    
    num_sentences = len(sentences)
    
    for i in range(0, num_sentences - 1, 2):
        sentence_a = sentences[i]
        
        # 50% 概率选择真实下一句
        if random.random() < 0.5 and i + 1 < num_sentences:
            sentence_b = sentences[i + 1]
            nsp_label = 0  # 真实下一句
        else:
            # 随机选择一个句子
            random_idx = random.randint(0, num_sentences - 1)
            while random_idx == i or random_idx == i + 1:
                random_idx = random.randint(0, num_sentences - 1)
            sentence_b = sentences[random_idx]
            nsp_label = 1  # 随机句子
        
        # 编码句子对
        # [CLS] sentence_a [SEP] sentence_b [SEP]
        tokens_a = tokenizer.encode(sentence_a, add_special_tokens=False)
        tokens_b = tokenizer.encode(sentence_b, add_special_tokens=False)
        
        # 截断以适应 max_length
        # 预留 3 个位置给 [CLS], [SEP], [SEP]
        max_tokens = max_length - 3
        while len(tokens_a) + len(tokens_b) > max_tokens:
            if len(tokens_a) > len(tokens_b):
                tokens_a = tokens_a[:-1]
            else:
                tokens_b = tokens_b[:-1]
        
        # 构建 input_ids
        input_ids = (
            [tokenizer.cls_token_id] +
            tokens_a +
            [tokenizer.sep_token_id] +
            tokens_b +
            [tokenizer.sep_token_id]
        )
        
        # 构建 segment_ids
        # segment 0: [CLS] + sentence_a + [SEP]
        # segment 1: sentence_b + [SEP]
        segment_ids = (
            [0] * (len(tokens_a) + 2) +
            [1] * (len(tokens_b) + 1)
        )
        
        # Padding
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
        segment_ids = segment_ids + [0] * padding_length
        
        batch_input_ids.append(input_ids)
        batch_segment_ids.append(segment_ids)
        batch_nsp_labels.append(nsp_label)
    
    return (
        torch.tensor(batch_input_ids),
        torch.tensor(batch_segment_ids),
        torch.tensor(batch_nsp_labels)
    )


# ==================== NWP 数据预处理 ====================

def prepare_nwp_data(
    input_ids: torch.Tensor,
    pad_token_id: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    准备 NWP（Next Word Prediction）训练数据
    
    将输入序列右移一位作为目标序列。
    
    Args:
        input_ids: (batch, seq_len) 输入 token IDs
        pad_token_id: padding token ID
    
    Returns:
        input_ids: 输入序列（不变）
        labels: 目标序列（右移一位），padding 位置为 -100
    """
    # 目标序列：input_ids 右移一位
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = pad_token_id  # 最后一个位置没有目标
    
    # 将 padding 位置的 label 设为 -100（忽略）
    labels[labels == pad_token_id] = -100
    
    return input_ids, labels


# ==================== BERT PreTrainer ====================

class BERTPreTrainer:
    """BERT 预训练器"""
    
    def __init__(
        self,
        model: BERTModel,
        config: TrainingConfig,
        tokenizer: SimpleTokenizer
    ):
        """
        初始化 BERT 预训练器
        
        Args:
            model: BERT 模型
            config: 训练配置
            tokenizer: Tokenizer 实例
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # 损失函数
        self.mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.nsp_loss_fn = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.global_step = 0
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 包含 input_ids, segment_ids, attention_mask, mlm_labels, nsp_labels
        
        Returns:
            {'mlm_loss': float, 'nsp_loss': float, 'total_loss': float}
        """
        self.model.train()
        
        # 前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            segment_ids=batch['segment_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # MLM 损失
        mlm_logits = outputs['mlm_logits']
        mlm_labels = batch['mlm_labels']
        mlm_loss = self.mlm_loss_fn(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1)
        )
        
        # NSP 损失
        nsp_logits = outputs['nsp_logits']
        nsp_labels = batch['nsp_labels']
        nsp_loss = self.nsp_loss_fn(nsp_logits, nsp_labels)
        
        # 总损失
        total_loss = mlm_loss + nsp_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.global_step += 1
        
        return {
            'mlm_loss': mlm_loss.item(),
            'nsp_loss': nsp_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                losses = self.train_step(batch)
                epoch_loss += losses['total_loss']
                num_batches += 1
                
                # 日志记录
                if self.global_step % self.config.log_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: "
                        f"MLM Loss={losses['mlm_loss']:.4f}, "
                        f"NSP Loss={losses['nsp_loss']:.4f}, "
                        f"Total Loss={losses['total_loss']:.4f}"
                    )
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Average Loss={avg_loss:.4f}")
    
    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"bert_step_{self.global_step}.pt"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'global_step': self.global_step,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


# ==================== GPT PreTrainer ====================

class GPTPreTrainer:
    """GPT 预训练器"""
    
    def __init__(
        self,
        model: GPTModel,
        config: TrainingConfig,
        tokenizer: SimpleTokenizer
    ):
        """
        初始化 GPT 预训练器
        
        Args:
            model: GPT 模型
            config: 训练配置
            tokenizer: Tokenizer 实例
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.global_step = 0
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 包含 input_ids, attention_mask, labels
        
        Returns:
            {'loss': float}
        """
        self.model.train()
        
        # 前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # NWP 损失
        logits = outputs['logits']
        labels = batch['labels']
        loss = self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        self.optimizer.step()
        self.global_step += 1
        
        return {'loss': loss.item()}
    
    def train(self, dataloader: DataLoader, num_epochs: int) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                losses = self.train_step(batch)
                epoch_loss += losses['loss']
                num_batches += 1
                
                # 日志记录
                if self.global_step % self.config.log_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: Loss={losses['loss']:.4f}"
                    )
                
                # 保存检查点
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: Average Loss={avg_loss:.4f}")
    
    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"gpt_step_{self.global_step}.pt"
        )
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.model.config,
            'global_step': self.global_step,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
