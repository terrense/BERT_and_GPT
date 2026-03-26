"""
GPT 模型实现（Decoder-Only 架构）

包含：
- GPTModel: GPT 主模型
- LMHead: 语言模型预测头
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from ..config import GPTConfig
from ..core.layers import DecoderLayer
from ..core.position import LearnablePositionEmbedding


class LMHead(nn.Module):
    """语言模型预测头：隐藏状态 -> 词表 logits"""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, d_model)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.decoder(hidden_states)


class GPTModel(nn.Module):
    """
    GPT 模型（Decoder-Only）
    
    组件:
    - Token Embedding
    - Position Embedding (Learnable)
    - N x Decoder Layer
    - LM Head (可选权重绑定)
    """
    
    def __init__(self, config: GPTConfig):
        """
        初始化 GPT 模型
        
        Args:
            config: GPT 配置
        """
        super().__init__()
        
        self.config = config
        
        # Embedding 层
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = LearnablePositionEmbedding(
            config.d_model, config.max_seq_len
        )
        
        # 嵌入层后的 LayerNorm 和 Dropout
        self.embedding_layer_norm = nn.LayerNorm(config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout_rate)
        
        # N 个 Decoder Layer
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])
        
        # 最终 LayerNorm
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        
        # LM Head
        self.lm_head = LMHead(config.d_model, config.vocab_size)
        
        # 权重绑定
        if config.tie_weights:
            self.lm_head.decoder.weight = self.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)，1 表示有效位置，0 表示 padding
        
        Returns:
            {
                'hidden_states': (batch, seq_len, d_model),
                'logits': (batch, seq_len, vocab_size)
            }
        """
        # Token Embedding
        token_emb = self.token_embedding(input_ids)
        
        # 位置嵌入
        embeddings = self.position_embedding(token_emb)
        
        # LayerNorm + Dropout
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # 转换 attention_mask 为 padding_mask
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (1 - attention_mask).float()
        
        # 通过 Decoder 层
        hidden_states = embeddings
        for decoder_layer in self.decoder_layers:
            hidden_states = decoder_layer(hidden_states, padding_mask=padding_mask)
        
        # 最终 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # LM logits
        logits = self.lm_head(hidden_states)
        
        return {
            'hidden_states': hidden_states,
            'logits': logits
        }
    
    def get_lm_head(self) -> nn.Module:
        """返回语言模型头"""
        return self.lm_head
