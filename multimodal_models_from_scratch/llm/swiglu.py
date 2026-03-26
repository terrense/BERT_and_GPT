"""
SwiGLU (Swish-Gated Linear Unit) 激活函数实现

SwiGLU 是 LLaMA 和 Qwen 等现代大语言模型中使用的前馈网络激活函数，
相比传统的 GELU 或 ReLU，SwiGLU 在大语言模型中表现更好。

结构: Linear(d_model, d_ff) * Swish(Linear(d_model, d_ff)) -> Linear(d_ff, d_model)
其中 Swish(x) = x * sigmoid(x)

需求: 7.3, 10.3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """
    SwiGLU 激活函数的前馈网络
    
    SwiGLU 使用门控机制，将输入通过两个并行的线性变换，
    一个作为门控信号（通过 Swish 激活），另一个作为值，
    两者相乘后再通过输出投影层。
    
    公式: output = W_down(gate * swish(gate_proj))
    其中:
    - gate = W_gate(x)
    - gate_proj = W_up(x)  
    - swish(x) = x * sigmoid(x)
    
    Args:
        d_model: 输入/输出维度
        d_ff: 中间层维度（隐藏层维度）
        dropout_rate: Dropout 比率
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # 门控投影层: d_model -> d_ff
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        
        # 上投影层: d_model -> d_ff
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        
        # 下投影层: d_ff -> d_model
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU 前向传播
        
        Args:
            x: 输入张量，形状为 (batch, seq_len, d_model)
        
        Returns:
            output: 输出张量，形状为 (batch, seq_len, d_model)
        """
        # 计算门控信号并应用 Swish 激活
        # swish(x) = x * sigmoid(x) = silu(x)
        gate = F.silu(self.w_gate(x))
        
        # 计算上投影
        up = self.w_up(x)
        
        # 门控乘法
        hidden = gate * up
        
        # 应用 dropout
        hidden = self.dropout(hidden)
        
        # 下投影
        output = self.w_down(hidden)
        
        return output
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, d_ff={self.d_ff}'
