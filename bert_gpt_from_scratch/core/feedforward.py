"""
前馈神经网络实现

结构: Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    前馈神经网络
    
    包含两层线性变换，中间使用 GELU 激活函数和 Dropout。
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout_rate: float = 0.1
    ):
        """
        初始化前馈网络
        
        Args:
            d_model: 模型维度（输入和输出维度）
            d_ff: 中间层维度
            dropout_rate: Dropout 比率
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 两层线性变换
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # GELU 激活函数
        self.activation = nn.GELU()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Linear -> GELU -> Dropout -> Linear
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        
        return x
