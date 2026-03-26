"""
BERT 模型实现（Encoder-Only 架构）

包含：
- BERTModel: BERT 主模型
- MLMHead: 掩码语言模型预测头
- NSPHead: 下一句预测头
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import BERTConfig
from ..core.layers import EncoderLayer
from ..core.position import LearnablePositionEmbedding


class MLMHead(nn.Module):
    """MLM 预测头：隐藏状态 -> 词表 logits"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class NSPHead(nn.Module):
    """NSP 预测头：[CLS] 隐藏状态 -> 二分类 logits"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, cls_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_hidden_state: (batch, d_model)
        Returns:
            logits: (batch, 2)
        """
        return self.classifier(cls_hidden_state)


class BERTModel(nn.Module):
    """
    BERT 模型（Encoder-Only）
    
    组件:
    - Token Embedding
    - Segment Embedding
    - Position Embedding (Learnable)
    - N x Encoder Layer
    - MLM Head
    - NSP Head
    """
    
    def __init__(self, config: BERTConfig):
        """
        初始化 BERT 模型
        
        Args:
            config: BERT 配置
        """
        super().__init__()
        
        self.config = config
        
        # Embedding 层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.segment_embedding = nn.Embedding(config.num_segments, config.d_model)
        self.position_embedding = LearnablePositionEmbedding(
            config.d_model, config.max_seq_len
        )
        
        # 嵌入层后的 LayerNorm 和 Dropout
        self.embedding_layer_norm = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        
        # N 个 Encoder Layer
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])
        
        # 预测头
        self.mlm_head = MLMHead(config.d_model, config.vocab_size)
        self.nsp_head = NSPHead(config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            segment_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)，1 表示有效位置，0 表示 padding
        
        Returns:
            {
                'hidden_states': (batch, seq_len, d_model),
                'mlm_logits': (batch, seq_len, vocab_size),
                'nsp_logits': (batch, 2)
            }
        """
        # Embedding
        token_emb = self.token_embedding(input_ids)
        segment_emb = self.segment_embedding(segment_ids)
        
        # 位置嵌入（LearnablePositionEmbedding 会自动加上位置编码）
        embeddings = token_emb + segment_emb
        embeddings = self.position_embedding(embeddings)
        
        # LayerNorm + Dropout
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # 转换 attention_mask 为 padding_mask
        # attention_mask: 1=有效, 0=padding -> padding_mask: 0=有效, 1=padding
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (1 - attention_mask).float()
        
        # 通过 Encoder 层
        hidden_states = embeddings
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, padding_mask=padding_mask)
        
        # MLM logits
        mlm_logits = self.mlm_head(hidden_states)
        
        # NSP logits（使用 [CLS] token 的隐藏状态，即第一个位置）
        cls_hidden_state = hidden_states[:, 0, :]
        nsp_logits = self.nsp_head(cls_hidden_state)
        
        return {
            'hidden_states': hidden_states,
            'mlm_logits': mlm_logits,
            'nsp_logits': nsp_logits
        }
    
    def get_mlm_head(self) -> nn.Module:
        """返回 MLM 预测头"""
        return self.mlm_head
    
    def get_nsp_head(self) -> nn.Module:
        """返回 NSP 预测头"""
        return self.nsp_head
