"""
Rotary Position Embedding (RoPE) 实现

RoPE 是 LLaMA、Qwen 等现代大语言模型中使用的位置编码方法。
通过旋转变换将位置信息编码到 Query 和 Key 向量中，
具有相对位置编码的特性，且支持外推到更长的序列。

核心思想：
- 将 Q/K 向量的每两个相邻维度视为一个复数
- 根据位置乘以不同频率的旋转矩阵
- 旋转角度 θ_i = position / (theta^(2i/d))

需求: 7.2, 10.2
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RotaryPositionEmbedding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding)
    
    将位置信息通过旋转变换编码到 Query 和 Key 向量中。
    支持 NTK-aware 插值用于扩展上下文长度（Qwen 使用）。
    
    Args:
        d_model: 模型维度（head_dim，每个注意力头的维度）
        max_seq_len: 最大序列长度，用于预计算缓存
        theta: 基础频率，默认 10000.0
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 4096,
        theta: float = 10000.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # 预计算频率
        # inv_freq[i] = 1 / (theta^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        # 注册为 buffer，不参与梯度计算，但会随模型保存/加载
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin 缓存
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int) -> None:
        """
        预计算 cos 和 sin 缓存
        
        Args:
            seq_len: 序列长度
        """
        # 位置索引: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        
        # 计算角度: positions @ inv_freq
        # freqs 形状: (seq_len, d_model/2)
        freqs = torch.outer(positions, self.inv_freq)
        
        # 将频率复制一份，形成完整的 d_model 维度
        # emb 形状: (seq_len, d_model)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # 预计算 cos 和 sin
        # 形状: (1, 1, seq_len, d_model) 用于广播
        cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def _extend_cache(self, seq_len: int) -> None:
        """
        扩展缓存以支持更长的序列
        
        Args:
            seq_len: 新的序列长度
        """
        if seq_len > self.cos_cached.shape[2]:
            self._build_cache(seq_len)
    
    def apply_ntk_scaling(self, seq_len: int, alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用 NTK-aware 插值扩展上下文长度（用于 Qwen）
        
        NTK-aware 插值通过调整基础频率 theta 来扩展位置编码的有效范围，
        而不是简单地线性插值位置索引。
        
        公式: theta_new = theta * alpha^(d/(d-2))
        
        Args:
            seq_len: 目标序列长度
            alpha: 缩放因子，通常为 seq_len / max_seq_len
        
        Returns:
            cos: 缩放后的 cos 缓存
            sin: 缩放后的 sin 缓存
        """
        if alpha == 1.0:
            self._extend_cache(seq_len)
            return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]
        
        # NTK-aware 缩放: 调整基础频率
        # theta_new = theta * alpha^(d/(d-2))
        d = self.d_model
        theta_scaled = self.theta * (alpha ** (d / (d - 2)))
        
        # 重新计算 inv_freq
        inv_freq_scaled = 1.0 / (theta_scaled ** (torch.arange(0, d, 2, device=self.inv_freq.device).float() / d))
        
        # 计算新的 cos 和 sin
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(positions, inv_freq_scaled)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        
        return cos, sin
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将张量的后半部分取负并与前半部分交换
        
        用于实现旋转变换: [x1, x2] -> [-x2, x1]
        
        Args:
            x: 输入张量，形状为 (..., d_model)
        
        Returns:
            rotated: 旋转后的张量
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """
        对输入张量应用旋转位置编码
        
        旋转公式:
        x_rotated = x * cos + rotate_half(x) * sin
        
        其中 rotate_half 将 [x1, x2, x3, x4, ...] 变换为 [-x2, x1, -x4, x3, ...]
        
        Args:
            x: 输入张量，形状为 (batch, num_heads, seq_len, head_dim)
            cos: cos 缓存，形状为 (1, 1, seq_len, head_dim)
            sin: sin 缓存，形状为 (1, 1, seq_len, head_dim)
        
        Returns:
            rotated: 旋转后的张量，形状与输入相同
        """
        return x * cos + self._rotate_half(x) * sin
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Query 和 Key 应用旋转位置编码
        
        Args:
            q: Query 张量，形状为 (batch, num_heads, seq_len, head_dim)
            k: Key 张量，形状为 (batch, num_heads, seq_len, head_dim)
            position_ids: 可选的位置索引，形状为 (batch, seq_len)
                         如果为 None，则使用 [0, 1, 2, ..., seq_len-1]
        
        Returns:
            rotated_q: 旋转后的 Query，形状与输入相同
            rotated_k: 旋转后的 Key，形状与输入相同
        """
        seq_len = q.shape[2]
        
        if position_ids is None:
            # 使用默认位置索引
            # 确保缓存足够长
            self._extend_cache(seq_len)
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        else:
            # 使用自定义位置索引（用于 KV Cache 场景）
            # position_ids 形状: (batch, seq_len)
            max_pos = position_ids.max().item() + 1
            self._extend_cache(int(max_pos))
            
            # 从缓存中选取对应位置的 cos 和 sin
            # cos_cached 形状: (1, 1, max_seq_len, d_model)
            # 需要根据 position_ids 索引，得到 (batch, seq_len, d_model)
            batch_size = position_ids.shape[0]
            
            # 使用 reshape 而不是 view，因为 position_ids 可能不是连续的
            # cos_cached[0, 0] 形状: (max_seq_len, d_model)
            pos_flat = position_ids.reshape(-1)  # (batch*seq_len,)
            cos_flat = self.cos_cached[0, 0, pos_flat, :]  # (batch*seq_len, d_model)
            sin_flat = self.sin_cached[0, 0, pos_flat, :]  # (batch*seq_len, d_model)
            
            # 重塑为 (batch, 1, seq_len, d_model) 以便广播
            cos = cos_flat.reshape(batch_size, seq_len, -1).unsqueeze(1)
            sin = sin_flat.reshape(batch_size, seq_len, -1).unsqueeze(1)
        
        # 应用旋转编码
        rotated_q = self.apply_rotary_emb(q, cos, sin)
        rotated_k = self.apply_rotary_emb(k, cos, sin)
        
        return rotated_q, rotated_k
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, max_seq_len={self.max_seq_len}, theta={self.theta}'
