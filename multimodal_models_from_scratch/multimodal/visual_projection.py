"""
视觉投影层模块

将视觉特征映射到 LLM 的嵌入空间。支持两种投影模式：
1. linear: 单层线性投影
2. mlp: 两层 MLP (Linear -> GELU -> Linear)，用于 LLaVA 等模型

需求: 4.3, 6.6, 9.2
"""

import torch
import torch.nn as nn
from typing import Literal


class VisualProjection(nn.Module):
    """
    视觉投影层
    
    将视觉特征映射到 LLM 的嵌入空间。
    
    Args:
        vision_dim: 视觉特征维度（来自视觉编码器的输出维度）
        llm_dim: LLM 嵌入维度（目标输出维度）
        projection_type: 投影类型
            - 'linear': 单层线性投影
            - 'mlp': 两层 MLP (Linear -> GELU -> Linear)
        mlp_depth: MLP 层数（仅在 projection_type='mlp' 时使用，默认为 2）
    
    Examples:
        >>> # 线性投影
        >>> proj = VisualProjection(vision_dim=768, llm_dim=4096, projection_type='linear')
        >>> visual_features = torch.randn(2, 196, 768)
        >>> output = proj(visual_features)  # (2, 196, 4096)
        
        >>> # MLP 投影（LLaVA 风格）
        >>> proj = VisualProjection(vision_dim=768, llm_dim=4096, projection_type='mlp')
        >>> visual_features = torch.randn(2, 196, 768)
        >>> output = proj(visual_features)  # (2, 196, 4096)
    """
    
    def __init__(
        self,
        vision_dim: int,
        llm_dim: int,
        projection_type: Literal['linear', 'mlp'] = 'mlp',
        mlp_depth: int = 2
    ):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        self.projection_type = projection_type
        self.mlp_depth = mlp_depth
        
        if projection_type == 'linear':
            self.projection = nn.Linear(vision_dim, llm_dim)
        elif projection_type == 'mlp':
            self.projection = self._build_mlp(vision_dim, llm_dim, mlp_depth)
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}. Expected 'linear' or 'mlp'.")
    
    def _build_mlp(self, input_dim: int, output_dim: int, depth: int) -> nn.Sequential:
        """
        构建 MLP 投影层
        
        对于 depth=2（默认）：
            Linear(input_dim, output_dim) -> GELU -> Linear(output_dim, output_dim)
        
        对于 depth>2：
            Linear(input_dim, output_dim) -> GELU -> [Linear(output_dim, output_dim) -> GELU] * (depth-2) -> Linear(output_dim, output_dim)
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            depth: MLP 层数（至少为 2）
        
        Returns:
            nn.Sequential: MLP 模块
        """
        if depth < 2:
            raise ValueError(f"mlp_depth must be at least 2, got {depth}")
        
        layers = []
        
        # 第一层：input_dim -> output_dim
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.GELU())
        
        # 中间层（如果 depth > 2）
        for _ in range(depth - 2):
            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.GELU())
        
        # 最后一层：output_dim -> output_dim（无激活函数）
        layers.append(nn.Linear(output_dim, output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            visual_features: 视觉特征张量，形状为 (batch, num_tokens, vision_dim)
        
        Returns:
            projected: 投影后的特征张量，形状为 (batch, num_tokens, llm_dim)
        """
        return self.projection(visual_features)
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"vision_dim={self.vision_dim}, "
            f"llm_dim={self.llm_dim}, "
            f"projection_type='{self.projection_type}'"
            + (f", mlp_depth={self.mlp_depth}" if self.projection_type == 'mlp' else "")
        )
