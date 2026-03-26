"""
Gated Cross Attention 模块

Gated Cross Attention 是 Flamingo 模型中的关键组件，用于在 LLM 的 Decoder Layer 中
插入交叉注意力层，使文本 token 能够关注视觉 token。

结构:
- Cross-Attention: 文本 token 关注视觉 token
- 门控参数: tanh(alpha)，初始化为 0

输出: hidden_states + tanh(alpha) * cross_attention_output

参考: Flamingo 论文 https://arxiv.org/abs/2204.14198

需求: 8.5, 8.6
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedCrossAttentionLayer(nn.Module):
    """
    门控交叉注意力层 (Flamingo)
    
    在 LLM 的 Decoder Layer 中插入的交叉注意力层，使文本 token 能够关注视觉 token。
    使用可学习的门控参数 tanh(alpha) 控制视觉信息的融合程度，初始化为 0。
    
    结构:
    - LayerNorm -> Cross-Attention
    - 门控参数: tanh(alpha)，初始化为 0
    
    输出: hidden_states + tanh(alpha) * cross_attention_output
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        dropout_rate: Dropout 比率
    
    Examples:
        >>> layer = GatedCrossAttentionLayer(d_model=768, num_heads=8)
        >>> hidden_states = torch.randn(2, 128, 768)  # (batch, seq_len, d_model)
        >>> visual_features = torch.randn(2, 64, 768)  # (batch, num_visual_tokens, d_model)
        >>> output = layer(hidden_states, visual_features)  # (2, 128, 768)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        # 门控参数 alpha，初始化为 0，使得 tanh(alpha) = 0
        # 这样在训练初期，门控交叉注意力不会影响 LLM 的输出
        self.alpha = nn.Parameter(torch.zeros(1))
        
        # Cross-Attention 组件
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_dropout = nn.Dropout(dropout_rate)
        
        # 残差连接的 Dropout
        self.residual_dropout = nn.Dropout(dropout_rate)
    
    def _cross_attention(
        self,
        hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        交叉注意力：文本 token 关注视觉 token
        
        Args:
            hidden_states: (batch, seq_len, d_model)
            visual_features: (batch, num_visual_tokens, d_model)
            visual_attention_mask: (batch, num_visual_tokens)，1 表示被掩码的位置
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_visual_tokens = visual_features.size(1)
        
        # 线性投影
        q = self.cross_attn_q(hidden_states)
        k = self.cross_attn_k(visual_features)
        v = self.cross_attn_v(visual_features)
        
        # 分头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_visual_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (batch, num_heads, seq_len, num_visual_tokens)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if visual_attention_mask is not None:
            # (batch, num_visual_tokens) -> (batch, 1, 1, num_visual_tokens)
            visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(visual_attention_mask.bool(), float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.cross_attn_dropout(attn_weights)
        
        # 计算输出: (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头: (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 输出投影
        output = self.cross_attn_out(attn_output)
        
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: 文本隐藏状态，形状为 (batch, seq_len, d_model)
            visual_features: 视觉特征，形状为 (batch, num_visual_tokens, d_model)
            visual_attention_mask: 视觉特征掩码，形状为 (batch, num_visual_tokens)
                                   1 表示被掩码的位置，0 表示有效位置
        
        Returns:
            output: 融合视觉信息后的隐藏状态，形状为 (batch, seq_len, d_model)
        """
        # Pre-Norm: LayerNorm -> Cross-Attention
        normed_hidden_states = self.cross_attn_norm(hidden_states)
        cross_attn_output = self._cross_attention(
            normed_hidden_states, 
            visual_features, 
            visual_attention_mask
        )
        cross_attn_output = self.residual_dropout(cross_attn_output)
        
        # 门控残差连接: hidden_states + tanh(alpha) * cross_attention_output
        gate = torch.tanh(self.alpha)
        output = hidden_states + gate * cross_attn_output
        
        return output
    
    def get_gate_value(self) -> torch.Tensor:
        """
        获取当前的门控值 tanh(alpha)
        
        Returns:
            gate: 门控值，范围在 [-1, 1]
        """
        return torch.tanh(self.alpha)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"dropout_rate={self.dropout_rate}"
        )
