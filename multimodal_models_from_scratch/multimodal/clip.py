"""
CLIP 模型（Contrastive Language-Image Pre-training）

实现图文对比学习模型，学习图像和文本的对齐表示。
- Vision Encoder: 基于 ViT 模型
- Text Encoder: 复用 bert-gpt-from-scratch 的 Transformer Encoder
- Visual Projection 和 Text Projection: 将特征投影到共享嵌入空间
- L2 归一化: 对投影后的特征进行归一化
- 可学习的温度参数: 用于对比学习

需求: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 17.4
"""

from typing import Dict, Optional, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_gpt_from_scratch.config import TransformerConfig
from bert_gpt_from_scratch.core.layers import EncoderLayer
from bert_gpt_from_scratch.core.position import LearnablePositionEmbedding

from ..config import CLIPConfig, VisionConfig
from ..vision.vit import ViTModel


class TextEncoder(nn.Module):
    """
    CLIP 文本编码器
    
    复用 bert-gpt-from-scratch 的 Transformer Encoder 组件。
    使用 [EOS] token（序列最后一个 token）的隐藏状态作为文本表示。
    
    Args:
        config: TransformerConfig 配置对象
    
    Attributes:
        token_embedding: Token 嵌入层
        position_embedding: 可学习的位置嵌入
        encoder_layers: N 个 Encoder Layer
        norm: 最终的 Layer Normalization
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position Embedding (复用 bert-gpt-from-scratch 的 LearnablePositionEmbedding)
        self.position_embedding = LearnablePositionEmbedding(
            config.d_model, config.max_seq_len
        )
        
        # Embedding Dropout
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
        
        # Layer Normalization
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码，1 表示有效位置，0 表示 padding
        
        Returns:
            text_features: (batch, d_model) 文本特征（使用 [EOS] token 的隐藏状态）
        """
        batch_size, seq_len = input_ids.shape
        
        # Token Embedding
        hidden_states = self.token_embedding(input_ids)
        
        # Position Embedding
        hidden_states = self.position_embedding(hidden_states)
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 转换 attention_mask 为 padding_mask
        # attention_mask: 1=有效, 0=padding -> padding_mask: 0=有效, 1=padding
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (1 - attention_mask).float()
        
        # 通过 Encoder 层
        for encoder_layer in self.encoder_layers:
            hidden_states = encoder_layer(hidden_states, padding_mask=padding_mask)
        
        # Layer Normalization
        hidden_states = self.norm(hidden_states)
        
        # 获取 [EOS] token 的隐藏状态（序列中最后一个有效 token）
        # 如果有 attention_mask，找到每个序列的最后一个有效位置
        if attention_mask is not None:
            # 计算每个序列的有效长度
            seq_lengths = attention_mask.sum(dim=1).long() - 1  # (batch,)
            # 使用 gather 获取每个序列的 [EOS] token
            text_features = hidden_states[
                torch.arange(batch_size, device=hidden_states.device),
                seq_lengths
            ]
        else:
            # 没有 attention_mask，使用最后一个位置
            text_features = hidden_states[:, -1]
        
        return text_features


class CLIPModel(nn.Module):
    """
    CLIP 模型（Contrastive Language-Image Pre-training）
    
    组件:
    - Vision Encoder (ViT): 编码图像
    - Text Encoder (Transformer Encoder): 编码文本，复用 bert-gpt-from-scratch
    - Visual Projection: 将视觉特征投影到共享嵌入空间
    - Text Projection: 将文本特征投影到共享嵌入空间
    - 可学习的温度参数: 用于对比学习
    
    Args:
        config: CLIPConfig 配置对象
    
    Attributes:
        vision_encoder: ViT 视觉编码器
        text_encoder: Transformer 文本编码器
        visual_projection: 视觉投影层
        text_projection: 文本投影层
        logit_scale: 可学习的温度参数（log scale）
    
    Examples:
        >>> config = CLIPConfig()
        >>> model = CLIPModel(config)
        >>> pixel_values = torch.randn(2, 3, 224, 224)
        >>> input_ids = torch.randint(0, 30000, (2, 77))
        >>> output = model(pixel_values, input_ids)
        >>> print(output['image_embeds'].shape)  # (2, 512)
        >>> print(output['text_embeds'].shape)   # (2, 512)
        >>> print(output['logits_per_image'].shape)  # (2, 2)
    """
    
    def __init__(self, config: CLIPConfig):
        super().__init__()
        
        self.config = config
        
        # Vision Encoder (ViT)
        # 创建一个不带分类头的 VisionConfig
        vision_config = VisionConfig(
            image_size=config.vision_config.image_size,
            patch_size=config.vision_config.patch_size,
            in_channels=config.vision_config.in_channels,
            d_model=config.vision_config.d_model,
            num_heads=config.vision_config.num_heads,
            num_layers=config.vision_config.num_layers,
            d_ff=config.vision_config.d_ff,
            dropout_rate=config.vision_config.dropout_rate,
            num_classes=0  # 不需要分类头
        )
        self.vision_encoder = ViTModel(vision_config)
        
        # Text Encoder (复用 bert-gpt-from-scratch 的 Transformer Encoder)
        self.text_encoder = TextEncoder(config.text_config)
        
        # Visual Projection: vision_d_model -> projection_dim
        self.visual_projection = nn.Linear(
            config.vision_config.d_model,
            config.projection_dim,
            bias=False
        )
        
        # Text Projection: text_d_model -> projection_dim
        self.text_projection = nn.Linear(
            config.text_config.d_model,
            config.projection_dim,
            bias=False
        )
        
        # 可学习的温度参数（使用 log scale，初始化为 log(1/0.07) ≈ 2.66）
        # temperature = exp(logit_scale)
        self.logit_scale = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.0 / config.temperature)).item()])
        )
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
        
        Returns:
            image_embeds: (batch, projection_dim) L2 归一化后的图像嵌入
        """
        # 通过 Vision Encoder 获取 [CLS] token 的隐藏状态
        vision_output = self.vision_encoder(pixel_values)
        image_features = vision_output['pooler_output']  # (batch, vision_d_model)
        
        # 投影到共享嵌入空间
        image_embeds = self.visual_projection(image_features)  # (batch, projection_dim)
        
        # L2 归一化
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        return image_embeds
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        编码文本
        
        Args:
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
        
        Returns:
            text_embeds: (batch, projection_dim) L2 归一化后的文本嵌入
        """
        # 通过 Text Encoder 获取文本特征
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, text_d_model)
        
        # 投影到共享嵌入空间
        text_embeds = self.text_projection(text_features)  # (batch, projection_dim)
        
        # L2 归一化
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        return text_embeds
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        计算图像和文本的嵌入，以及它们之间的相似度矩阵。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
        
        Returns:
            Dict 包含:
            - 'image_embeds': (batch, projection_dim) L2 归一化后的图像嵌入
            - 'text_embeds': (batch, projection_dim) L2 归一化后的文本嵌入
            - 'logits_per_image': (batch, batch) 图像到文本的相似度矩阵
            - 'logits_per_text': (batch, batch) 文本到图像的相似度矩阵
            - 'temperature': scalar 当前温度值
        """
        # 编码图像和文本
        image_embeds = self.encode_image(pixel_values)  # (batch, projection_dim)
        text_embeds = self.encode_text(input_ids, attention_mask)  # (batch, projection_dim)
        
        # 计算温度（从 log scale 转换）
        # 限制 logit_scale 的范围以防止数值不稳定
        logit_scale = torch.clamp(self.logit_scale, max=100.0)
        temperature = torch.exp(logit_scale)
        
        # 计算相似度矩阵
        # logits_per_image[i, j] = similarity(image_i, text_j) * temperature
        logits_per_image = temperature * (image_embeds @ text_embeds.T)  # (batch, batch)
        logits_per_text = logits_per_image.T  # (batch, batch)
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'temperature': temperature.squeeze()
        }
    
    def get_temperature(self) -> torch.Tensor:
        """
        获取当前温度值
        
        Returns:
            temperature: scalar 当前温度值
        """
        return torch.exp(self.logit_scale).squeeze()
    
    def zero_shot_classify(
        self,
        pixel_values: torch.Tensor,
        text_labels: List[str],
        tokenizer: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        零样本图像分类
        
        给定图像和文本标签列表，返回最匹配的标签。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            text_labels: 文本标签列表，例如 ["a photo of a cat", "a photo of a dog"]
            tokenizer: 分词器，需要支持 __call__ 方法，返回包含 'input_ids' 和可选 'attention_mask' 的字典
        
        Returns:
            predicted_labels: (batch,) 预测的标签索引
            probabilities: (batch, num_labels) 每个标签的概率分布
        
        Examples:
            >>> config = CLIPConfig()
            >>> model = CLIPModel(config)
            >>> pixel_values = torch.randn(2, 3, 224, 224)
            >>> text_labels = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
            >>> # 假设 tokenizer 是一个简单的分词器
            >>> predicted_labels, probabilities = model.zero_shot_classify(pixel_values, text_labels, tokenizer)
            >>> print(predicted_labels.shape)  # (2,)
            >>> print(probabilities.shape)     # (2, 3)
        """
        # 编码图像
        image_embeds = self.encode_image(pixel_values)  # (batch, projection_dim)
        
        # 对文本标签进行分词
        # tokenizer 应该返回包含 'input_ids' 的字典
        tokenized = tokenizer(text_labels)
        input_ids = tokenized['input_ids']  # (num_labels, seq_len)
        attention_mask = tokenized.get('attention_mask', None)
        
        # 确保 input_ids 是 tensor
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        
        # 将 input_ids 移动到与 pixel_values 相同的设备
        input_ids = input_ids.to(pixel_values.device)
        
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(pixel_values.device)
        
        # 编码文本标签
        text_embeds = self.encode_text(input_ids, attention_mask)  # (num_labels, projection_dim)
        
        # 计算图像与每个文本标签的相似度
        # similarity[i, j] = image_i · text_j
        similarity = image_embeds @ text_embeds.T  # (batch, num_labels)
        
        # 应用温度缩放
        logit_scale = torch.clamp(self.logit_scale, max=100.0)
        temperature = torch.exp(logit_scale)
        logits = temperature * similarity  # (batch, num_labels)
        
        # 计算概率分布（softmax）
        probabilities = F.softmax(logits, dim=-1)  # (batch, num_labels)
        
        # 获取预测的标签索引
        predicted_labels = torch.argmax(probabilities, dim=-1)  # (batch,)
        
        return predicted_labels, probabilities


def contrastive_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor
) -> torch.Tensor:
    """
    计算 CLIP 对比损失（InfoNCE Loss）
    
    双向对比损失：(image->text + text->image) / 2
    
    Args:
        logits_per_image: (batch, batch) 图像到文本的相似度矩阵
        logits_per_text: (batch, batch) 文本到图像的相似度矩阵
    
    Returns:
        loss: scalar 对比损失
    """
    batch_size = logits_per_image.shape[0]
    
    # 创建标签：对角线上的元素是正样本
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    # 图像到文本的损失
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    
    # 文本到图像的损失
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    # 双向平均
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss
