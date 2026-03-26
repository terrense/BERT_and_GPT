"""
Grouped Query Attention (GQA) 实现

GQA 是 LLaMA 2、Qwen 等现代大语言模型中使用的注意力机制。
通过减少 Key/Value 头的数量来降低 KV Cache 的内存占用，
同时保持模型性能。

核心思想：
- Query 头数 > Key/Value 头数
- 多个 Query 头共享同一组 Key/Value 头
- num_groups = num_heads // num_kv_heads

需求: 7.5, 10.4
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention)
    
    允许 num_kv_heads < num_heads，多个 Query 头共享同一组 Key/Value 头。
    这种设计可以显著减少 KV Cache 的内存占用，同时保持模型性能。
    
    Args:
        d_model: 模型维度
        num_heads: Query 头数
        num_kv_heads: Key/Value 头数 (num_heads 必须能被 num_kv_heads 整除)
        dropout_rate: Dropout 比率，默认 0.0
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0
    ):
        super().__init__()
        
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads
        
        assert self.head_dim * num_heads == d_model, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Query 投影: d_model -> num_heads * head_dim
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        
        # Key/Value 投影: d_model -> num_kv_heads * head_dim
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 Key/Value 头重复以匹配 Query 头数
        
        Args:
            x: (batch, num_kv_heads, seq_len, head_dim)
        
        Returns:
            repeated: (batch, num_heads, seq_len, head_dim)
        """
        if self.num_groups == 1:
            return x
        
        batch, num_kv_heads, seq_len, head_dim = x.shape
        # 扩展维度: (batch, num_kv_heads, 1, seq_len, head_dim)
        x = x.unsqueeze(2)
        # 重复: (batch, num_kv_heads, num_groups, seq_len, head_dim)
        x = x.expand(batch, num_kv_heads, self.num_groups, seq_len, head_dim)
        # 重塑: (batch, num_heads, seq_len, head_dim)
        x = x.reshape(batch, self.num_heads, seq_len, head_dim)
        return x
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入张量，形状为 (batch, seq_len, d_model)
            attention_mask: 注意力掩码，形状为 (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len)
                           值为 0 表示允许注意力，值为 -inf 表示屏蔽
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
                                cos/sin 形状: (1, 1, seq_len, head_dim)
            past_key_value: KV Cache，元组 (past_key, past_value)
                           past_key/past_value 形状: (batch, num_kv_heads, past_seq_len, head_dim)
            use_cache: 是否返回更新后的 KV Cache
        
        Returns:
            output: 输出张量，形状为 (batch, seq_len, d_model)
            past_key_value: 如果 use_cache=True，返回更新后的 KV Cache
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算 Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        # query: (batch, seq_len, num_heads, head_dim) -> (batch, num_heads, seq_len, head_dim)
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # key/value: (batch, seq_len, num_kv_heads, head_dim) -> (batch, num_kv_heads, seq_len, head_dim)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE 位置编码
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query = self._apply_rotary_emb(query, cos, sin)
            key = self._apply_rotary_emb(key, cos, sin)
        
        # 处理 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            # 拼接历史 KV
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # 保存当前 KV 用于缓存
        current_key_value = (key, value) if use_cache else None
        
        # 将 KV 头重复以匹配 Query 头数
        key_expanded = self._repeat_kv(key)
        value_expanded = self._repeat_kv(value)
        
        # 计算注意力分数
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, kv_seq_len)
        # -> (batch, num_heads, seq_len, kv_seq_len)
        attn_weights = torch.matmul(query, key_expanded.transpose(-2, -1)) * self.scale
        
        # 应用注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax 归一化
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        # (batch, num_heads, seq_len, kv_seq_len) @ (batch, num_heads, kv_seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, value_expanded)
        
        # 重塑回原始格式
        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # (batch, seq_len, num_heads * head_dim)
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output, current_key_value
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将张量的后半部分取负并与前半部分交换
        
        用于实现旋转变换: [x1, x2] -> [-x2, x1]
        
        Args:
            x: 输入张量，形状为 (..., head_dim)
        
        Returns:
            rotated: 旋转后的张量
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        对输入张量应用旋转位置编码
        
        旋转公式:
        x_rotated = x * cos + rotate_half(x) * sin
        
        Args:
            x: 输入张量，形状为 (batch, num_heads, seq_len, head_dim)
            cos: cos 缓存，形状为 (1, 1, seq_len, head_dim)
            sin: sin 缓存，形状为 (1, 1, seq_len, head_dim)
        
        Returns:
            rotated: 旋转后的张量，形状与输入相同
        """
        return x * cos + self._rotate_half(x) * sin
    
    def extra_repr(self) -> str:
        return (
            f'd_model={self.d_model}, num_heads={self.num_heads}, '
            f'num_kv_heads={self.num_kv_heads}, num_groups={self.num_groups}, '
            f'head_dim={self.head_dim}'
        )
