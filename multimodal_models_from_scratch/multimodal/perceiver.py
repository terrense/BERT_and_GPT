"""
Perceiver Resampler 模块

Perceiver Resampler 是 Flamingo 模型中的关键组件，用于将可变长度的视觉特征
压缩为固定数量的 latent 向量。

结构:
- 可学习的 latent 向量 (num_latents, d_model)
- N 个 PerceiverResamplerLayer，每层包含:
  1. 交叉注意力层：latent 向量关注视觉特征
  2. 自注意力层：latent 向量之间的交互
  3. 前馈网络

参考: Flamingo 论文 https://arxiv.org/abs/2204.14198

需求: 8.2, 8.3
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResamplerLayer(nn.Module):
    """
    Perceiver Resampler 层
    
    结构 (Pre-Norm):
    1. LayerNorm -> Cross-Attention -> Residual
    2. LayerNorm -> Self-Attention -> Residual
    3. LayerNorm -> FFN -> Residual
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络中间层维度
        dropout_rate: Dropout 比率
    
    Examples:
        >>> layer = PerceiverResamplerLayer(d_model=768, num_heads=8, d_ff=3072)
        >>> latents = torch.randn(2, 64, 768)  # (batch, num_latents, d_model)
        >>> visual_features = torch.randn(2, 196, 768)  # (batch, num_patches, d_model)
        >>> output = layer(latents, visual_features)  # (2, 64, 768)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 8,
        d_ff: int = 3072,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        # Cross-Attention 组件 (latent 关注 visual features)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_dropout = nn.Dropout(dropout_rate)

        # Self-Attention 组件 (latent 之间的交互)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_dropout = nn.Dropout(dropout_rate)
        
        # FFN 组件
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_linear2 = nn.Linear(d_ff, d_model)
        self.ffn_activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
        # 残差连接的 Dropout
        self.residual_dropout = nn.Dropout(dropout_rate)
    
    def _cross_attention(
        self,
        latents: torch.Tensor,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        交叉注意力：latent 向量关注视觉特征
        
        Args:
            latents: (batch, num_latents, d_model)
            visual_features: (batch, num_patches, d_model)
            visual_attention_mask: (batch, num_patches)
        
        Returns:
            output: (batch, num_latents, d_model)
        """
        batch_size, num_latents, _ = latents.shape
        num_patches = visual_features.size(1)
        
        # 线性投影
        q = self.cross_attn_q(latents)
        k = self.cross_attn_k(visual_features)
        v = self.cross_attn_v(visual_features)
        
        # 分头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (batch, num_heads, num_latents, num_patches)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if visual_attention_mask is not None:
            # (batch, num_patches) -> (batch, 1, 1, num_patches)
            visual_attention_mask = visual_attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(visual_attention_mask.bool(), float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.cross_attn_dropout(attn_weights)
        
        # 计算输出: (batch, num_heads, num_latents, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头: (batch, num_latents, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_latents, self.d_model)
        
        # 输出投影
        output = self.cross_attn_out(attn_output)
        
        return output

    def _self_attention(
        self,
        latents: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        自注意力：latent 向量之间的交互
        
        Args:
            latents: (batch, num_latents, d_model)
            attention_mask: (batch, num_latents) 或 (batch, num_latents, num_latents)
        
        Returns:
            output: (batch, num_latents, d_model)
        """
        batch_size, num_latents, _ = latents.shape
        
        # 线性投影
        q = self.self_attn_q(latents)
        k = self.self_attn_k(latents)
        v = self.self_attn_v(latents)
        
        # 分头
        q = q.view(batch_size, num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_latents, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (batch, num_heads, num_latents, num_latents)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (batch, num_latents) -> (batch, 1, 1, num_latents)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (batch, num_latents, num_latents) -> (batch, 1, num_latents, num_latents)
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask.bool(), float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.self_attn_dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_latents, self.d_model)
        
        # 输出投影
        output = self.self_attn_out(attn_output)
        
        return output
    
    def _feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈网络
        
        Args:
            x: (batch, seq_len, d_model)
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        x = self.ffn_linear1(x)
        x = self.ffn_activation(x)
        x = self.ffn_dropout(x)
        x = self.ffn_linear2(x)
        return x

    def forward(
        self,
        latents: torch.Tensor,
        visual_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            latents: latent 向量，形状为 (batch, num_latents, d_model)
            visual_features: 视觉特征，形状为 (batch, num_patches, d_model)
            attention_mask: 自注意力掩码，形状为 (batch, num_latents)
            visual_attention_mask: 交叉注意力掩码，形状为 (batch, num_patches)
        
        Returns:
            latents: 更新后的 latent 向量，形状为 (batch, num_latents, d_model)
        """
        # Pre-Norm: LayerNorm -> Attention -> Residual
        
        # Cross-Attention (latent 关注 visual features)
        residual = latents
        latents = self.cross_attn_norm(latents)
        latents = self._cross_attention(latents, visual_features, visual_attention_mask)
        latents = self.residual_dropout(latents)
        latents = residual + latents
        
        # Self-Attention (latent 之间的交互)
        residual = latents
        latents = self.self_attn_norm(latents)
        latents = self._self_attention(latents, attention_mask)
        latents = self.residual_dropout(latents)
        latents = residual + latents
        
        # FFN
        residual = latents
        latents = self.ffn_norm(latents)
        latents = self._feed_forward(latents)
        latents = self.residual_dropout(latents)
        latents = residual + latents
        
        return latents
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"d_ff={self.d_ff}, "
            f"dropout_rate={self.dropout_rate}"
        )


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler (Flamingo)
    
    将可变长度的视觉特征压缩为固定数量的 latent 向量。
    使用可学习的 latent 向量作为查询，通过交叉注意力聚合视觉特征。
    
    Args:
        d_model: 模型维度
        num_latents: latent 向量数量（默认 64）
        num_heads: 注意力头数量
        num_layers: PerceiverResamplerLayer 层数
        d_ff: 前馈网络中间层维度（默认为 4 * d_model）
        dropout_rate: Dropout 比率
    
    Examples:
        >>> resampler = PerceiverResampler(d_model=768, num_latents=64, num_heads=8, num_layers=6)
        >>> visual_features = torch.randn(2, 196, 768)  # (batch, num_patches, d_model)
        >>> output = resampler(visual_features)  # (2, 64, 768)
    """
    
    def __init__(
        self,
        d_model: int,
        num_latents: int = 64,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: Optional[int] = None,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.dropout_rate = dropout_rate
        
        # 可学习的 latent 向量: (1, num_latents, d_model)
        self.latents = nn.Parameter(
            torch.zeros(1, num_latents, d_model)
        )
        
        # Perceiver Resampler 层
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=self.d_ff,
                dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ])
        
        # 最终的 LayerNorm
        self.final_norm = nn.LayerNorm(d_model)
        
        # 初始化 latent 向量
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # 使用截断正态分布初始化 latent 向量
        nn.init.trunc_normal_(self.latents, std=0.02)
    
    def forward(
        self,
        visual_features: torch.Tensor,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        将可变长度的视觉特征压缩为固定数量的 latent 向量。
        
        Args:
            visual_features: 视觉特征，形状为 (batch, num_patches, d_model)
            visual_attention_mask: 视觉特征掩码，形状为 (batch, num_patches)
        
        Returns:
            latents: 压缩后的 latent 向量，形状为 (batch, num_latents, d_model)
        """
        batch_size = visual_features.size(0)
        
        # 扩展 latent 向量到 batch 维度: (1, num_latents, d_model) -> (batch, num_latents, d_model)
        latents = self.latents.expand(batch_size, -1, -1)
        
        # 通过所有 Perceiver Resampler 层
        for layer in self.layers:
            latents = layer(
                latents=latents,
                visual_features=visual_features,
                visual_attention_mask=visual_attention_mask
            )
        
        # 应用最终的 LayerNorm
        latents = self.final_norm(latents)
        
        return latents
    
    def get_latents(self) -> torch.Tensor:
        """
        获取可学习的 latent 向量
        
        Returns:
            latents: (1, num_latents, d_model)
        """
        return self.latents
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"d_model={self.d_model}, "
            f"num_latents={self.num_latents}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"d_ff={self.d_ff}, "
            f"dropout_rate={self.dropout_rate}"
        )
