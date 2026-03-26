"""
监督微调（SFT）Trainer 实现

包含：
- SFTTrainer: 基础 SFT 功能
- BERT 文本分类微调
- GPT 指令微调
"""

import logging
import os
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config import SFTConfig
from ..models.bert import BERTModel
from ..models.gpt import GPTModel
from ..tokenizer.simple_tokenizer import SimpleTokenizer

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """分类头：用于 BERT 文本分类"""
    
    def __init__(self, d_model: int, num_classes: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, cls_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_hidden_state: (batch, d_model)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.dropout(cls_hidden_state)
        return self.classifier(x)


class SFTTrainer:
    """监督微调训练器"""
    
    def __init__(
        self,
        model: Union[BERTModel, GPTModel],
        config: SFTConfig,
        tokenizer: SimpleTokenizer
    ):
        """
        初始化 SFT 训练器
        
        Args:
            model: BERT 或 GPT 模型
            config: SFT 配置
            tokenizer: Tokenizer 实例
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        
        self.is_bert = isinstance(model, BERTModel)
        self.classification_head = None
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # 优化器（稍后在训练时初始化，以便包含分类头参数）
        self.optimizer = None
        self.global_step = 0
    
    def load_pretrained(self, checkpoint_path: str) -> None:
        """
        加载预训练检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded pretrained checkpoint from {checkpoint_path}")
    
    def freeze_layers(self, num_layers: int) -> None:
        """
        冻结前 N 层参数
        
        Args:
            num_layers: 要冻结的层数
        """
        if num_layers <= 0:
            return
        
        # 冻结 embedding 层
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False
        
        if self.is_bert:
            for param in self.model.segment_embedding.parameters():
                param.requires_grad = False
            for param in self.model.position_embedding.parameters():
                param.requires_grad = False
            
            # 冻结前 N 个 encoder 层
            layers_to_freeze = min(num_layers, len(self.model.encoder_layers))
            for i in range(layers_to_freeze):
                for param in self.model.encoder_layers[i].parameters():
                    param.requires_grad = False
        else:
            for param in self.model.position_embedding.parameters():
                param.requires_grad = False
            
            # 冻结前 N 个 decoder 层
            layers_to_freeze = min(num_layers, len(self.model.decoder_layers))
            for i in range(layers_to_freeze):
                for param in self.model.decoder_layers[i].parameters():
                    param.requires_grad = False
        
        logger.info(f"Frozen {num_layers} layers")
    
    def _init_optimizer(self):
        """初始化优化器"""
        params = list(self.model.parameters())
        if self.classification_head is not None:
            params += list(self.classification_head.parameters())
        
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, params),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
    
    def train_classification(
        self,
        dataloader: DataLoader,
        num_classes: int
    ) -> None:
        """
        BERT 文本分类微调
        
        Args:
            dataloader: 数据加载器，batch 包含 input_ids, segment_ids, attention_mask, labels
            num_classes: 分类类别数
        """
        if not self.is_bert:
            raise ValueError("Classification training is only supported for BERT models")
        
        # 初始化分类头
        self.classification_head = ClassificationHead(
            self.model.config.d_model,
            num_classes,
            self.model.config.dropout_rate
        )
        
        # 冻结层
        self.freeze_layers(self.config.freeze_layers)
        
        # 初始化优化器
        self._init_optimizer()
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            self.classification_head.train()
            
            for batch in dataloader:
                loss = self._classification_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if self.global_step % self.config.log_steps == 0:
                    logger.info(f"Step {self.global_step}: Loss={loss:.4f}")
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}: Average Loss={avg_loss:.4f}")
    
    def _classification_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """分类训练单步"""
        # 前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            segment_ids=batch['segment_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # 获取 [CLS] 隐藏状态
        cls_hidden_state = outputs['hidden_states'][:, 0, :]
        
        # 分类
        logits = self.classification_head(cls_hidden_state)
        
        # 计算损失
        loss = self.loss_fn(logits, batch['labels'])
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        self.global_step += 1
        
        return loss.item()
    
    def train_instruction(
        self,
        dataloader: DataLoader
    ) -> None:
        """
        GPT 指令微调
        
        仅对 response 部分计算损失，忽略 instruction 部分。
        
        Args:
            dataloader: 数据加载器，batch 包含 input_ids, attention_mask, labels
                        labels 中 instruction 部分应设为 -100
        """
        if self.is_bert:
            raise ValueError("Instruction training is only supported for GPT models")
        
        # 冻结层
        self.freeze_layers(self.config.freeze_layers)
        
        # 初始化优化器
        self._init_optimizer()
        
        # 训练循环
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            self.model.train()
            
            for batch in dataloader:
                loss = self._instruction_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                if self.global_step % self.config.log_steps == 0:
                    logger.info(f"Step {self.global_step}: Loss={loss:.4f}")
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}: Average Loss={avg_loss:.4f}")
    
    def _instruction_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """指令微调训练单步"""
        # 前向传播
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask')
        )
        
        # 计算损失（仅对 response 部分，instruction 部分的 label 为 -100）
        logits = outputs['logits']
        labels = batch['labels']
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = self.loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        self.global_step += 1
        
        return loss.item()
    
    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """保存检查点"""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            model_type = "bert" if self.is_bert else "gpt"
            path = os.path.join(
                self.config.checkpoint_dir,
                f"{model_type}_sft_step_{self.global_step}.pt"
            )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.model.config,
            'global_step': self.global_step,
        }
        
        if self.classification_head is not None:
            checkpoint['classification_head_state_dict'] = self.classification_head.state_dict()
        
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")


def prepare_instruction_labels(
    input_ids: torch.Tensor,
    instruction_lengths: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """
    准备指令微调的标签
    
    将 instruction 部分的标签设为 -100，仅保留 response 部分。
    
    Args:
        input_ids: (batch, seq_len) 输入 token IDs
        instruction_lengths: (batch,) 每个样本的 instruction 长度
        pad_token_id: padding token ID
    
    Returns:
        labels: (batch, seq_len) 标签，instruction 和 padding 部分为 -100
    """
    labels = input_ids.clone()
    
    batch_size, seq_len = input_ids.shape
    
    for i in range(batch_size):
        # 将 instruction 部分设为 -100
        inst_len = instruction_lengths[i].item()
        labels[i, :inst_len] = -100
    
    # 将 padding 部分设为 -100
    labels[labels == pad_token_id] = -100
    
    return labels
