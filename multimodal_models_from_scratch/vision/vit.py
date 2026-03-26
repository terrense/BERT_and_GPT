"""
Vision Transformer (ViT) 模型

实现完整的 ViT 模型，用于图像分类和多模态任务的视觉编码。
复用 bert-gpt-from-scratch 的 Encoder Layer 组件。

需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.1, 17.2, 17.3
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from bert_gpt_from_scratch.core.layers import EncoderLayer
from ..config import VisionConfig
from .patch_embedding import PatchEmbedding


class ViTModel(nn.Module):
    """Vision Transformer 模型
    
    将图像分割为 patch，通过 Transformer Encoder 编码，
    用于图像分类和多模态任务的视觉特征提取。
    
    组件:
    - Patch Embedding: 将图像分割为 patch 并映射到嵌入空间
    - N x Encoder Layer: 复用 bert-gpt-from-scratch 的 Encoder Layer
    - Layer Normalization: 在最后一层 Encoder 输出后应用
    - Classification Head: 将 [CLS] token 映射到类别 logits（可选）
    
    Args:
        config: VisionConfig 配置对象，包含模型超参数
    
    Attributes:
        patch_embedding: Patch Embedding 模块
        encoder_layers: N 个 Encoder Layer
        norm: 最终的 Layer Normalization
        classifier: 分类头（如果 num_classes > 0）
    """
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        
        self.config = config
        
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            d_model=config.d_model
        )
        
        # Dropout after embedding
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # N x Encoder Layer (复用 bert-gpt-from-scratch)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                dropout_rate=config.dropout_rate
            )
            for _ in range(config.num_layers)
        ])
        
        # Layer Normalization after the last encoder layer
        self.norm = nn.LayerNorm(config.d_model)
        
        # Classification Head (optional)
        # 如果 num_classes > 0，则创建分类头
        self.classifier = None
        if config.num_classes > 0:
            self.classifier = nn.Linear(config.d_model, config.num_classes)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            pixel_values: 输入图像张量，形状为 (batch, 3, H, W)
            output_hidden_states: 是否输出所有层的隐藏状态
        
        Returns:
            Dict 包含:
            - 'last_hidden_state': (batch, num_patches + 1, d_model) 最后一层的隐藏状态
            - 'pooler_output': (batch, d_model) [CLS] token 的隐藏状态
            - 'logits': (batch, num_classes) 分类 logits（如果有分类头）
            - 'hidden_states': tuple 所有层的隐藏状态（如果 output_hidden_states=True）
        """
        # 1. Patch Embedding
        # (batch, 3, H, W) -> (batch, num_patches + 1, d_model)
        hidden_states = self.patch_embedding(pixel_values)
        hidden_states = self.dropout(hidden_states)
        
        # 收集所有层的隐藏状态（如果需要）
        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
        if output_hidden_states:
            all_hidden_states = (hidden_states,)
        
        # 2. N x Encoder Layer
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        
        # 3. Layer Normalization
        hidden_states = self.norm(hidden_states)
        
        # 4. 提取 [CLS] token 作为 pooler_output
        # [CLS] token 在序列的第一个位置
        pooler_output = hidden_states[:, 0]
        
        # 5. 构建输出字典
        output = {
            'last_hidden_state': hidden_states,
            'pooler_output': pooler_output,
        }
        
        # 6. 分类 logits（如果有分类头）
        if self.classifier is not None:
            logits = self.classifier(pooler_output)
            output['logits'] = logits
        
        # 7. 所有层的隐藏状态（如果需要）
        if output_hidden_states:
            output['hidden_states'] = all_hidden_states
        
        return output
    
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """获取图像特征（不含 [CLS] token）
        
        用于多模态任务，返回所有 patch 的特征表示。
        
        Args:
            pixel_values: 输入图像张量，形状为 (batch, 3, H, W)
        
        Returns:
            features: (batch, num_patches, d_model) 不含 [CLS] token 的特征
        """
        # 获取完整的隐藏状态
        output = self.forward(pixel_values, output_hidden_states=False)
        last_hidden_state = output['last_hidden_state']
        
        # 移除 [CLS] token（第一个位置）
        # (batch, num_patches + 1, d_model) -> (batch, num_patches, d_model)
        features = last_hidden_state[:, 1:]
        
        return features
