"""
位置编码实现

包含：
- SinusoidalPositionEncoding: 正弦余弦位置编码（不可学习）
- LearnablePositionEmbedding: 可学习位置嵌入
"""

import math

import torch
import torch.nn as nn


class SinusoidalPositionEncoding(nn.Module):
    """
    正弦余弦位置编码（不可学习）
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        初始化正弦余弦位置编码
        
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 预计算位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为 buffer（不参与梯度计算）
        # (1, max_seq_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置编码与输入相加
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x + position_encoding: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class LearnablePositionEmbedding(nn.Module):
    """
    可学习位置嵌入
    
    使用 nn.Embedding 实现可学习的位置向量。
    """
    
    def __init__(self, d_model: int, max_seq_len: int):
        """
        初始化可学习位置嵌入
        
        Args:
            d_model: 模型维度
            max_seq_len: 最大序列长度
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # 可学习的位置嵌入矩阵
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将位置嵌入与输入相加
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            x + position_embedding: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return x + self.position_embedding(positions)
