"""
LLaVA (Large Language and Vision Assistant) 模型实现

LLaVA 是一种视觉指令微调模型，通过将视觉特征直接注入到 LLM 的输入序列中，
使大语言模型具备视觉理解能力。

核心架构：
1. Vision Encoder (ViT): 将图像编码为视觉特征
2. Visual Projection (MLP): 将视觉特征映射到 LLM 的嵌入空间
3. LLM (LLaMA): 语言模型基座，处理融合后的多模态序列

关键特性：
- 支持 <image> 特殊 token 标记图像插入位置
- 支持两阶段训练：第一阶段仅训练 Visual Projection，第二阶段全参数微调
- 支持多轮对话格式
- 仅对 response 部分计算损失（指令微调时）

需求: 9.1, 9.2, 9.3, 9.4, 9.5, 9.8
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple

from multimodal_models_from_scratch.config import LLaVAConfig
from multimodal_models_from_scratch.vision.vit import ViTModel
from multimodal_models_from_scratch.multimodal.visual_projection import VisualProjection
from multimodal_models_from_scratch.llm.llama import LLaMAModel


# 默认的 <image> token ID
DEFAULT_IMAGE_TOKEN_ID = -200


class LLaVAModel(nn.Module):
    """
    LLaVA 模型
    
    将视觉特征直接注入到 LLM 的输入序列中，实现视觉-语言多模态理解。
    
    组件:
    - Vision Encoder (ViT): 可选冻结或微调
    - Visual Projection (MLP): 将视觉特征映射到 LLM 嵌入空间
    - LLM (LLaMA): 语言模型基座
    
    Args:
        config: LLaVAConfig 配置对象
    
    Attributes:
        vision_encoder: ViT 视觉编码器
        visual_projection: 视觉投影层
        llm: LLaMA 语言模型
    """
    
    def __init__(self, config: LLaVAConfig):
        super().__init__()
        
        self.config = config
        
        # Vision Encoder (ViT)
        self.vision_encoder = ViTModel(config.vision_config)
        
        # Visual Projection (MLP)
        self.visual_projection = VisualProjection(
            vision_dim=config.vision_config.d_model,
            llm_dim=config.llm_config.d_model,
            projection_type=config.projection_type
        )
        
        # LLM (LLaMA)
        self.llm = LLaMAModel(config.llm_config)
        
        # 冻结参数（根据配置）
        if config.freeze_vision:
            self._freeze_vision_encoder()
        
        if config.freeze_llm:
            self._freeze_llm()
    
    def _freeze_vision_encoder(self):
        """冻结视觉编码器参数"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def _freeze_llm(self):
        """冻结 LLM 参数"""
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_vision_encoder(self):
        """解冻视觉编码器参数（用于第二阶段训练）"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = True
    
    def unfreeze_llm(self):
        """解冻 LLM 参数（用于第二阶段训练）"""
        for param in self.llm.parameters():
            param.requires_grad = True
    
    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取视觉特征并投影到 LLM 嵌入空间
        
        Args:
            pixel_values: 输入图像，形状为 (batch, 3, H, W)
        
        Returns:
            visual_tokens: 投影后的视觉 token，形状为 (batch, num_patches, llm_dim)
        """
        # 获取 ViT 输出（不含 [CLS] token）
        # (batch, num_patches, vision_dim)
        visual_features = self.vision_encoder.get_image_features(pixel_values)
        
        # 投影到 LLM 嵌入空间
        # (batch, num_patches, llm_dim)
        visual_tokens = self.visual_projection(visual_features)
        
        return visual_tokens
    
    def _merge_visual_tokens(
        self,
        input_ids: torch.Tensor,
        visual_tokens: torch.Tensor,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        将视觉 token 插入到 <image> 位置
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            visual_tokens: 视觉 token，形状为 (batch, num_visual_tokens, llm_dim)
            image_token_index: <image> token 的 ID
        
        Returns:
            merged_embeds: 融合后的嵌入，形状为 (batch, new_seq_len, llm_dim)
            merged_attention_mask: 融合后的注意力掩码，形状为 (batch, new_seq_len)
            position_ids: 位置 ID，形状为 (batch, new_seq_len)
        """
        batch_size, seq_len = input_ids.shape
        num_visual_tokens = visual_tokens.shape[1]
        device = input_ids.device
        
        # 获取文本嵌入
        text_embeds = self.llm.embed_tokens(input_ids)  # (batch, seq_len, llm_dim)
        
        # 找到每个样本中 <image> token 的位置
        # 假设每个样本只有一个 <image> token
        image_positions = (input_ids == image_token_index).nonzero(as_tuple=False)
        
        # 如果没有 <image> token，直接返回文本嵌入
        if image_positions.shape[0] == 0:
            attention_mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.long)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            return text_embeds, attention_mask, position_ids
        
        # 计算新序列长度：原序列长度 - 1（移除 <image> token）+ num_visual_tokens
        new_seq_len = seq_len - 1 + num_visual_tokens
        
        # 创建输出张量
        merged_embeds = torch.zeros(
            batch_size, new_seq_len, text_embeds.shape[-1],
            device=device, dtype=text_embeds.dtype
        )
        merged_attention_mask = torch.ones(
            batch_size, new_seq_len,
            device=device, dtype=torch.long
        )
        
        # 为每个样本处理
        for i in range(batch_size):
            # 找到当前样本的 <imag