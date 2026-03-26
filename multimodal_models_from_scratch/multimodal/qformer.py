"""
Q-Former 模块

Q-Former (Querying Transformer) 是 BLIP-2 中的关键组件，用于桥接视觉编码器和 LLM。
它使用可学习的查询 token 通过交叉注意力从图像特征中提取视觉信息。

结构:
- 可学习的查询向量 (num_query_tokens, d_model)
- N 个 QFormerLayer，每层包含:
  1. 自注意力层：查询向量之间的交互
  2. 交叉注意力层：查询向量关注视觉特征
  3. 前馈网络

参考: BLIP-2 论文 https://arxiv.org/abs/2301.12597

需求: 6.2, 6.3, 6.4, 6.5
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class QFormerLayer(nn.Module):
    """
    Q-Former 层
    
    结构 (Pre-Norm):
    1. LayerNorm -> Self-Attention -> Residual
    2. LayerNorm -> Cross-Attention -> Residual
    3. LayerNorm -> FFN -> Residual
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        d_ff: 前馈网络中间层维度
        dropout_rate: Dropout 比率
        use_pre_norm: 是否使用 Pre-Norm 架构（默认 True）
    
    Examples:
        >>> layer = QFormerLayer(d_model=768, num_heads=12, d_ff=3072)
        >>> query_embeds = torch.randn(2, 32, 768)  # (batch, num_queries, d_model)
        >>> encoder_hidden_states = torch.randn(2, 196, 768)  # (batch, num_patches, d_model)
        >>> output = layer(query_embeds, encoder_hidden_states)  # (2, 32, 768)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        dropout_rate: float = 0.1,
        use_pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        self.use_pre_norm = use_pre_norm
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
        # Self-Attention 组件
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_dropout = nn.Dropout(dropout_rate)
        
        # Cross-Attention 组件
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_dropout = nn.Dropout(dropout_rate)
        
        # FFN 组件
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_linear2 = nn.Linear(d_ff, d_model)
        self.ffn_activation = nn.GELU()
        self.ffn_dropout = nn.Dropout(dropout_rate)
        
        # 残差连接的 Dropout
        self.residual_dropout = nn.Dropout(dropout_rate)
    
    def _self_attention(
        self,
        query_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        自注意力：查询向量之间的交互
        
        Args:
            query_embeds: (batch, num_queries, d_model)
            attention_mask: (batch, num_queries) 或 (batch, num_queries, num_queries)
        
        Returns:
            output: (batch, num_queries, d_model)
        """
        batch_size, num_queries, _ = query_embeds.shape
        
        # 线性投影
        q = self.self_attn_q(query_embeds)
        k = self.self_attn_k(query_embeds)
        v = self.self_attn_v(query_embeds)
        
        # 分头: (batch, num_queries, d_model) -> (batch, num_heads, num_queries, head_dim)
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (batch, num_heads, num_queries, num_queries)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # (batch, num_queries) -> (batch, 1, 1, num_queries)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            elif attention_mask.dim() == 3:
                # (batch, num_queries, num_queries) -> (batch, 1, num_queries, num_queries)
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores.masked_fill(attention_mask.bool(), float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.self_attn_dropout(attn_weights)
        
        # 计算输出: (batch, num_heads, num_queries, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头: (batch, num_queries, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.d_model)
        
        # 输出投影
        output = self.self_attn_out(attn_output)
        
        return output
    
    def _cross_attention(
        self,
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        交叉注意力：查询向量关注视觉特征
        
        Args:
            query_embeds: (batch, num_queries, d_model)
            encoder_hidden_states: (batch, num_patches, d_model)
            encoder_attention_mask: (batch, num_patches)
        
        Returns:
            output: (batch, num_queries, d_model)
        """
        batch_size, num_queries, _ = query_embeds.shape
        num_patches = encoder_hidden_states.size(1)
        
        # 线性投影
        q = self.cross_attn_q(query_embeds)
        k = self.cross_attn_k(encoder_hidden_states)
        v = self.cross_attn_v(encoder_hidden_states)
        
        # 分头
        q = q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (batch, num_heads, num_queries, num_patches)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码
        if encoder_attention_mask is not None:
            # (batch, num_patches) -> (batch, 1, 1, num_patches)
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(encoder_attention_mask.bool(), float('-inf'))
        
        # Softmax 和 Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.cross_attn_dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.d_model)
        
        # 输出投影
        output = self.cross_attn_out(attn_output)
        
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
        query_embeds: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            query_embeds: 查询向量，形状为 (batch, num_queries, d_model)
            encoder_hidden_states: 视觉特征，形状为 (batch, num_patches, d_model)
            attention_mask: 自注意力掩码，形状为 (batch, num_queries) 或 (batch, num_queries, num_queries)
            encoder_attention_mask: 交叉注意力掩码，形状为 (batch, num_patches)
        
        Returns:
            query_output: 更新后的查询向量，形状为 (batch, num_queries, d_model)
        """
        if self.use_pre_norm:
            # Pre-Norm: LayerNorm -> Attention -> Residual
            
            # Self-Attention
            residual = query_embeds
            query_embeds = self.self_attn_norm(query_embeds)
            query_embeds = self._self_attention(query_embeds, attention_mask)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = residual + query_embeds
            
            # Cross-Attention
            residual = query_embeds
            query_embeds = self.cross_attn_norm(query_embeds)
            query_embeds = self._cross_attention(query_embeds, encoder_hidden_states, encoder_attention_mask)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = residual + query_embeds
            
            # FFN
            residual = query_embeds
            query_embeds = self.ffn_norm(query_embeds)
            query_embeds = self._feed_forward(query_embeds)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = residual + query_embeds
        else:
            # Post-Norm: Attention -> Residual -> LayerNorm
            
            # Self-Attention
            residual = query_embeds
            query_embeds = self._self_attention(query_embeds, attention_mask)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = self.self_attn_norm(residual + query_embeds)
            
            # Cross-Attention
            residual = query_embeds
            query_embeds = self._cross_attention(query_embeds, encoder_hidden_states, encoder_attention_mask)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = self.cross_attn_norm(residual + query_embeds)
            
            # FFN
            residual = query_embeds
            query_embeds = self._feed_forward(query_embeds)
            query_embeds = self.residual_dropout(query_embeds)
            query_embeds = self.ffn_norm(residual + query_embeds)
        
        return query_embeds
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"d_ff={self.d_ff}, "
            f"dropout_rate={self.dropout_rate}, "
            f"use_pre_norm={self.use_pre_norm}"
        )


class QFormer(nn.Module):
    """
    Querying Transformer (Q-Former)
    
    BLIP-2 中的核心组件，使用可学习的查询向量从视觉特征中提取固定数量的视觉 token。
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        num_layers: QFormerLayer 层数
        d_ff: 前馈网络中间层维度
        num_query_tokens: 查询 token 数量（默认 32）
        dropout_rate: Dropout 比率
        use_pre_norm: 是否使用 Pre-Norm 架构
    
    Examples:
        >>> qformer = QFormer(d_model=768, num_heads=12, num_layers=6, num_query_tokens=32)
        >>> encoder_hidden_states = torch.randn(2, 196, 768)  # (batch, num_patches, d_model)
        >>> output = qformer(encoder_hidden_states)  # (2, 32, 768)
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        d_ff: int = 3072,
        num_query_tokens: int = 32,
        dropout_rate: float = 0.1,
        use_pre_norm: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.num_query_tokens = num_query_tokens
        self.dropout_rate = dropout_rate
        self.use_pre_norm = use_pre_norm
        
        # 可学习的查询向量: (1, num_query_tokens, d_model)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_tokens, d_model)
        )
        
        # QFormer 层
        self.layers = nn.ModuleList([
            QFormerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout_rate=dropout_rate,
                use_pre_norm=use_pre_norm
            )
            for _ in range(num_layers)
        ])
        
        # 最终的 LayerNorm（仅在 Pre-Norm 架构中使用）
        self.final_norm = nn.LayerNorm(d_model) if use_pre_norm else None
        
        # 初始化查询向量
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        # 使用截断正态分布初始化查询向量
        nn.init.trunc_normal_(self.query_tokens, std=0.02)
    
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        query_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            encoder_hidden_states: 视觉特征，形状为 (batch, num_patches, d_model)
            encoder_attention_mask: 视觉特征掩码，形状为 (batch, num_patches)
            query_embeds: 可选的外部查询向量，形状为 (batch, num_queries, d_model)
                         如果提供，将使用此向量而非内部的 query_tokens
        
        Returns:
            query_output: 查询输出，形状为 (batch, num_query_tokens, d_model)
        """
        batch_size = encoder_hidden_states.size(0)
        
        # 获取查询向量
        if query_embeds is None:
            # 扩展查询向量到 batch 维度: (1, num_query_tokens, d_model) -> (batch, num_query_tokens, d_model)
            query_embeds = self.query_tokens.expand(batch_size, -1, -1)
        
        # 通过所有 QFormer 层
        for layer in self.layers:
            query_embeds = layer(
                query_embeds=query_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask
            )
        
        # 应用最终的 LayerNorm（Pre-Norm 架构）
        if self.final_norm is not None:
            query_embeds = self.final_norm(query_embeds)
        
        return query_embeds
    
    def get_query_tokens(self) -> torch.Tensor:
        """
        获取可学习的查询向量
        
        Returns:
            query_tokens: (1, num_query_tokens, d_model)
        """
        return self.query_tokens
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"d_model={self.d_model}, "
            f"num_heads={self.num_heads}, "
            f"num_layers={self.num_layers}, "
            f"d_ff={self.d_ff}, "
            f"num_query_tokens={self.num_query_tokens}, "
            f"dropout_rate={self.dropout_rate}, "
            f"use_pre_norm={self.use_pre_norm}"
        )
