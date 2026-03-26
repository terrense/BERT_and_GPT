"""
Transformer 层实现

包含：
- EncoderLayer: Transformer Encoder 层
- DecoderLayer: Transformer Decoder 层（GPT 风格）
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Transformer Encoder 层
    
    结构:
    1. Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed Forward Network + Residual + LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1
    ):
        """
        初始化 Encoder 层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络中间层维度
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        # Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        # Feed Forward Network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len)，padding 位置为 1
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-Attention + Residual + LayerNorm
        attn_output = self.self_attention(x, x, x, mask=padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    Transformer Decoder 层（GPT 风格，仅自注意力）
    
    结构:
    1. Masked Multi-Head Self-Attention + Residual + LayerNorm
    2. Feed Forward Network + Residual + LayerNorm
    
    自动应用 causal mask 确保自回归特性。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_rate: float = 0.1
    ):
        """
        初始化 Decoder 层
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            d_ff: 前馈网络中间层维度
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        # Masked Multi-Head Self-Attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        # Feed Forward Network
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout_rate)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        自动生成并应用 causal mask。
        
        Args:
            x: (batch, seq_len, d_model)
            padding_mask: (batch, seq_len)，padding 位置为 1
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # 创建 causal mask
        causal_mask = MultiHeadAttention.create_causal_mask(seq_len, device=x.device)
        
        # 合并 causal mask 和 padding mask
        if padding_mask is not None:
            # padding_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(2)
            # causal_mask: (seq_len, seq_len) -> (1, 1, seq_len, seq_len)
            causal_mask_expanded = causal_mask.unsqueeze(0).unsqueeze(0)
            # 合并：任一为 True 则掩码
            combined_mask = causal_mask_expanded | padding_mask_expanded.bool()
        else:
            combined_mask = causal_mask
        
        # Masked Self-Attention + Residual + LayerNorm
        attn_output = self.self_attention(x, x, x, mask=combined_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed Forward + Residual + LayerNorm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x
