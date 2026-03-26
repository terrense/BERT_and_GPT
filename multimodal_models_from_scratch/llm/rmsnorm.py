"""
Root Mean Square Layer Normalization (RMSNorm) 实现

RMSNorm 是 LLaMA 和 Qwen 等现代大语言模型中使用的归一化方法，
相比 LayerNorm 更加简单高效，不需要计算均值和减去均值的操作。

公式: x * weight / sqrt(mean(x^2) + eps)

需求: 7.1, 10.1
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    
    与 LayerNorm 不同，RMSNorm 不进行中心化（不减去均值），
    只进行缩放操作，计算效率更高。
    
    Args:
        d_model: 输入维度
        eps: 数值稳定性参数，防止除零
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # 可学习的缩放参数，初始化为 1
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入进行 RMSNorm 归一化
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            normalized: 归一化后的张量，形状为 (batch, seq_len, d_model)
        """
        # 计算 x^2 的均值（沿最后一个维度）
        # mean(x^2) 形状: (batch, seq_len, 1)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        
        # 计算 RMS: sqrt(mean(x^2) + eps)
        rms = torch.sqrt(variance + self.eps)
        
        # 归一化: x / rms
        x_normalized = x / rms
        
        # 应用可学习的缩放参数
        return self.weight * x_normalized
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, eps={self.eps}'
