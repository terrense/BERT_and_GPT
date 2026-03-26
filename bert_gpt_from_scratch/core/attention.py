"""
Multi-Head Attention 实现

包含：
- Scaled Dot-Product Attention
- Multi-Head Attention
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 Scaled Dot-Product Attention
    
    公式: softmax(QK^T / sqrt(d_k))V
    
    Args:
        query: (batch, heads, seq_len, d_k)
        key: (batch, heads, seq_len, d_k)
        value: (batch, heads, seq_len, d_v)
        mask: (batch, 1, 1, seq_len) 或 (batch, 1, seq_len, seq_len)
              被掩码位置为 True 或 1
        dropout: 可选的 Dropout 层
    
    Returns:
        output: (batch, heads, seq_len, d_v)
        attention_weights: (batch, heads, seq_len, seq_len)
    """
    d_k = query.size(-1)
    
    # 计算注意力分数: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 应用掩码（将被掩码位置设为负无穷）
    if mask is not None:
        scores = scores.masked_fill(mask.bool(), float('-inf'))
    
    # Softmax 归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 应用 Dropout
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 计算输出: attention_weights @ V
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    将输入通过线性投影分为多个注意力头并行计算后拼接。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1
    ):
        """
        初始化 Multi-Head Attention
        
        Args:
            d_model: 模型维度
            num_heads: 注意力头数量
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q、K、V 线性投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query: (batch, seq_len, d_model)
            key: (batch, seq_len, d_model)
            value: (batch, seq_len, d_model)
            mask: (batch, seq_len) 或 (batch, seq_len, seq_len)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # 线性投影并分头
        # (batch, seq_len, d_model) -> (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 处理 mask 形状
        if mask is not None:
            if mask.dim() == 2:
                # (batch, seq_len) -> (batch, 1, 1, seq_len)
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
                mask = mask.unsqueeze(1)
        
        # 计算注意力
        attn_output, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        
        # 拼接多头输出
        # (batch, num_heads, seq_len, d_k) -> (batch, seq_len, num_heads, d_k) -> (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出投影
        output = self.w_o(attn_output)
        
        return output
    
    @staticmethod
    def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
        """
        创建因果掩码（上三角矩阵）
        
        Args:
            seq_len: 序列长度
            device: 设备
        
        Returns:
            mask: (seq_len, seq_len)，上三角为 True
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
