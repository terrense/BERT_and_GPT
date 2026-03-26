"""
Patch Embedding 模块

将图像分割为固定大小的 patch 并映射到嵌入空间。
实现 ViT 的核心组件：patch 分割、线性投影、[CLS] token 和位置嵌入。

需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """图像块嵌入模块
    
    将输入图像分割为固定大小的 patch，并通过卷积层映射到 d_model 维的嵌入向量。
    添加可学习的 [CLS] token 和位置嵌入。
    
    Args:
        image_size: 输入图像尺寸（假设为正方形）
        patch_size: patch 大小（默认 16x16）
        in_channels: 输入通道数（默认 3，RGB 图像）
        d_model: 嵌入维度
    
    Attributes:
        num_patches: patch 数量，等于 (image_size // patch_size) ** 2
        projection: 使用 Conv2d 实现的 patch 分割和线性投影
        cls_token: 可学习的 [CLS] token，形状为 (1, 1, d_model)
        position_embedding: 可学习的位置嵌入，形状为 (1, num_patches + 1, d_model)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 768
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        
        # 计算 patch 数量
        # num_patches = (H / patch_size) * (W / patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        
        # 使用 Conv2d 实现 patch 分割和线性投影
        # kernel_size=patch_size, stride=patch_size 实现不重叠的 patch 分割
        # 输出通道数为 d_model，实现线性投影
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 可学习的 [CLS] token，形状为 (1, 1, d_model)
        # 用于聚合整个图像的信息，用于分类任务
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # 可学习的位置嵌入，形状为 (1, num_patches + 1, d_model)
        # +1 是因为包含 [CLS] token
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重
        
        - [CLS] token 使用截断正态分布初始化
        - 位置嵌入使用截断正态分布初始化
        """
        # 使用截断正态分布初始化，标准差为 0.02
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: 输入图像张量，形状为 (batch, in_channels, H, W)
        
        Returns:
            embeddings: 嵌入张量，形状为 (batch, num_patches + 1, d_model)
                       包含 [CLS] token 在序列开头
        """
        batch_size = x.shape[0]
        
        # 1. Patch 分割和线性投影
        # 输入: (batch, in_channels, H, W)
        # 输出: (batch, d_model, H/patch_size, W/patch_size)
        x = self.projection(x)
        
        # 2. 展平空间维度并转置
        # (batch, d_model, H/patch_size, W/patch_size) -> (batch, d_model, num_patches)
        x = x.flatten(2)
        # (batch, d_model, num_patches) -> (batch, num_patches, d_model)
        x = x.transpose(1, 2)
        
        # 3. 扩展 [CLS] token 到 batch 维度
        # (1, 1, d_model) -> (batch, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 4. 将 [CLS] token 拼接到序列开头
        # (batch, num_patches, d_model) -> (batch, num_patches + 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 5. 添加位置嵌入
        # 位置嵌入会自动广播到 batch 维度
        x = x + self.position_embedding
        
        return x
