"""
BLIP 模型（Bootstrapping Language-Image Pre-training）

实现图文理解与生成模型，支持三个预训练任务：
- ITC (Image-Text Contrastive): 图文对比学习
- ITM (Image-Text Matching): 图文匹配判断
- ITG (Image-grounded Text Generation): 基于图像的文本生成

组件:
- Vision Encoder: 基于 ViT 模型
- Text Encoder: 带 Cross-Attention 的 Transformer Encoder
- Text Decoder: 带 Cross-Attention 的 Transformer Decoder
- ITC Head: 对比学习投影头
- ITM Head: 图文匹配分类头

需求: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_gpt_from_scratch.config import TransformerConfig
from bert_gpt_from_scratch.core.attention import MultiHeadAttention
from bert_gpt_from_scratch.core.feedforward import FeedForwardNetwork
from bert_gpt_from_scratch.core.position import LearnablePositionEmbedding

from ..config import BLIPConfig, VisionConfig
from ..vision.vit import ViTModel


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层
    
    用于在 Text Encoder/Decoder 中融合视觉特征。
    
    结构:
    - Cross-Attention: 文本 token 关注视觉 token
    - Layer Normalization + Residual
    
    Args:
        d_model: 模型维度
        num_heads: 注意力头数量
        dropout_rate: Dropout 比率
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Cross-Attention
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        
        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            hidden_states: (batch, seq_len, d_model) 文本隐藏状态
            encoder_hidden_states: (batch, num_patches, d_model) 视觉特征
            encoder_attention_mask: (batch, num_patches) 视觉特征掩码
        
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Cross-Attention: Q 来自文本，K/V 来自视觉
        cross_attn_output = self.cross_attention(
            query=hidden_states,
            key=encoder_hidden_states,
            value=encoder_hidden_states,
            mask=encoder_attention_mask
        )
        
        # Residual + LayerNorm
        output = self.norm(hidden_states + self.dropout(cross_attn_output))
        
        return output


class TextEncoderWithCrossAttention(nn.Module):
    """
    带 Cross-Attention 的文本编码器
    
    用于 BLIP 的 ITC 和 ITM 任务。
    每个 Encoder Layer 后添加 Cross-Attention 层融合视觉特征。
    
    Args:
        config: TransformerConfig 配置对象
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position Embedding
        self.position_embedding = LearnablePositionEmbedding(
            config.d_model, config.max_seq_len
        )
        
        # Embedding Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Encoder Layers with Cross-Attention
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attention': MultiHeadAttention(
                    config.d_model, config.num_heads, config.dropout_rate
                ),
                'self_attn_norm': nn.LayerNorm(config.d_model),
                'cross_attention': CrossAttentionLayer(
                    config.d_model, config.num_heads, config.dropout_rate
                ),
                'ffn': FeedForwardNetwork(
                    config.d_model, config.d_ff, config.dropout_rate
                ),
                'ffn_norm': nn.LayerNorm(config.d_model)
            }))
        
        # Final Layer Normalization
        self.norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码，1=有效，0=padding
            encoder_hidden_states: (batch, num_patches, d_model) 视觉特征
            encoder_attention_mask: (batch, num_patches) 视觉特征掩码
        
        Returns:
            hidden_states: (batch, seq_len, d_model)
        """
        # Token Embedding
        hidden_states = self.token_embedding(input_ids)
        
        # Position Embedding
        hidden_states = self.position_embedding(hidden_states)
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 转换 attention_mask 为 padding_mask
        padding_mask = None
        if attention_mask is not None:
            padding_mask = (1 - attention_mask).float()
        
        # 转换 encoder_attention_mask
        encoder_padding_mask = None
        if encoder_attention_mask is not None:
            encoder_padding_mask = (1 - encoder_attention_mask).float()
        
        # 通过 Encoder 层
        for layer in self.layers:
            # Self-Attention + Residual + LayerNorm
            self_attn_output = layer['self_attention'](
                hidden_states, hidden_states, hidden_states, mask=padding_mask
            )
            hidden_states = layer['self_attn_norm'](
                hidden_states + self.dropout(self_attn_output)
            )
            
            # Cross-Attention (如果有视觉特征)
            if encoder_hidden_states is not None:
                hidden_states = layer['cross_attention'](
                    hidden_states, encoder_hidden_states, encoder_padding_mask
                )
            
            # FFN + Residual + LayerNorm
            ffn_output = layer['ffn'](hidden_states)
            hidden_states = layer['ffn_norm'](
                hidden_states + self.dropout(ffn_output)
            )
        
        # Final Layer Normalization
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class TextDecoderWithCrossAttention(nn.Module):
    """
    带 Cross-Attention 的文本解码器
    
    用于 BLIP 的 ITG 任务（基于图像的文本生成）。
    使用 causal mask 实现自回归生成。
    
    Args:
        config: TransformerConfig 配置对象
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Token Embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Position Embedding
        self.position_embedding = LearnablePositionEmbedding(
            config.d_model, config.max_seq_len
        )
        
        # Embedding Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Decoder Layers with Cross-Attention
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attention': MultiHeadAttention(
                    config.d_model, config.num_heads, config.dropout_rate
                ),
                'self_attn_norm': nn.LayerNorm(config.d_model),
                'cross_attention': CrossAttentionLayer(
                    config.d_model, config.num_heads, config.dropout_rate
                ),
                'ffn': FeedForwardNetwork(
                    config.d_model, config.d_ff, config.dropout_rate
                ),
                'ffn_norm': nn.LayerNorm(config.d_model)
            }))
        
        # Final Layer Normalization
        self.norm = nn.LayerNorm(config.d_model)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码，1=有效，0=padding
            encoder_hidden_states: (batch, num_patches, d_model) 视觉特征
            encoder_attention_mask: (batch, num_patches) 视觉特征掩码
            labels: (batch, seq_len) 目标 token IDs，用于计算损失
        
        Returns:
            Dict 包含:
            - 'logits': (batch, seq_len, vocab_size)
            - 'loss': scalar (如果提供 labels)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token Embedding
        hidden_states = self.token_embedding(input_ids)
        
        # Position Embedding
        hidden_states = self.position_embedding(hidden_states)
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # 创建 causal mask (seq_len, seq_len)
        # MultiHeadAttention 期望 mask 形状为 (batch, seq_len) 或 (batch, seq_len, seq_len)
        causal_mask = MultiHeadAttention.create_causal_mask(seq_len, device=device)
        # 扩展为 (batch, seq_len, seq_len)
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 合并 causal mask 和 padding mask
        if attention_mask is not None:
            # padding_mask: (batch, seq_len) -> (batch, 1, seq_len)
            padding_mask = (1 - attention_mask).bool().unsqueeze(1)
            # 合并: 任一为 True 则掩码
            combined_mask = causal_mask | padding_mask
        else:
            combined_mask = causal_mask
        
        # 转换 encoder_attention_mask
        encoder_padding_mask = None
        if encoder_attention_mask is not None:
            encoder_padding_mask = (1 - encoder_attention_mask).float()
        
        # 通过 Decoder 层
        for layer in self.layers:
            # Masked Self-Attention + Residual + LayerNorm
            self_attn_output = layer['self_attention'](
                hidden_states, hidden_states, hidden_states, mask=combined_mask
            )
            hidden_states = layer['self_attn_norm'](
                hidden_states + self.dropout(self_attn_output)
            )
            
            # Cross-Attention (如果有视觉特征)
            if encoder_hidden_states is not None:
                hidden_states = layer['cross_attention'](
                    hidden_states, encoder_hidden_states, encoder_padding_mask
                )
            
            # FFN + Residual + LayerNorm
            ffn_output = layer['ffn'](hidden_states)
            hidden_states = layer['ffn_norm'](
                hidden_states + self.dropout(ffn_output)
            )
        
        # Final Layer Normalization
        hidden_states = self.norm(hidden_states)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        output = {'logits': logits}
        
        # 计算损失（如果提供 labels）
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            # 计算交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['loss'] = loss
        
        return output



class BLIPModel(nn.Module):
    """
    BLIP 模型（Bootstrapping Language-Image Pre-training）
    
    组件:
    - Vision Encoder (ViT): 编码图像
    - Text Encoder (带 Cross-Attention): 用于 ITC 和 ITM 任务
    - Text Decoder (带 Cross-Attention): 用于 ITG 任务
    - ITC Head: 对比学习投影头
    - ITM Head: 图文匹配分类头
    
    支持三个预训练任务:
    - ITC (Image-Text Contrastive): 图文对比学习
    - ITM (Image-Text Matching): 图文匹配判断
    - ITG (Image-grounded Text Generation): 基于图像的文本生成
    
    Args:
        config: BLIPConfig 配置对象
    
    Examples:
        >>> config = BLIPConfig()
        >>> model = BLIPModel(config)
        >>> pixel_values = torch.randn(2, 3, 224, 224)
        >>> input_ids = torch.randint(0, 30000, (2, 32))
        >>> # ITC 前向传播
        >>> itc_output = model.forward_itc(pixel_values, input_ids)
        >>> print(itc_output['image_embeds'].shape)  # (2, 256)
        >>> # ITM 前向传播
        >>> itm_logits = model.forward_itm(pixel_values, input_ids)
        >>> print(itm_logits.shape)  # (2, 2)
        >>> # ITG 前向传播
        >>> labels = torch.randint(0, 30000, (2, 32))
        >>> lm_loss = model.forward_itg(pixel_values, input_ids, labels)
    """
    
    def __init__(self, config: BLIPConfig):
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
        
        # Text Encoder (带 Cross-Attention，用于 ITC 和 ITM)
        self.text_encoder = TextEncoderWithCrossAttention(config.text_config)
        
        # Text Decoder (带 Cross-Attention，用于 ITG)
        self.text_decoder = TextDecoderWithCrossAttention(config.text_config)
        
        # ITC Head: 投影到共享嵌入空间
        # Visual Projection
        self.visual_projection = nn.Linear(
            config.vision_config.d_model,
            config.projection_dim,
            bias=False
        )
        # Text Projection
        self.text_projection = nn.Linear(
            config.text_config.d_model,
            config.projection_dim,
            bias=False
        )
        
        # 可学习的温度参数（用于 ITC）
        self.logit_scale = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.0 / 0.07)).item()])
        )
        
        # ITM Head: 二分类（匹配/不匹配）
        self.itm_head = nn.Linear(config.text_config.d_model, 2)
    
    def encode_image(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        编码图像
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
        
        Returns:
            Dict 包含:
            - 'image_features': (batch, num_patches, d_model) 所有 patch 的特征
            - 'pooler_output': (batch, d_model) [CLS] token 的特征
        """
        vision_output = self.vision_encoder(pixel_values)
        
        return {
            'image_features': vision_output['last_hidden_state'][:, 1:],  # 不含 [CLS]
            'pooler_output': vision_output['pooler_output']
        }
    
    def forward_itc(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Image-Text Contrastive 前向传播
        
        计算图像和文本的对比学习嵌入和相似度矩阵。
        
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
        """
        batch_size = pixel_values.shape[0]
        
        # 编码图像
        image_output = self.encode_image(pixel_values)
        image_features = image_output['pooler_output']  # (batch, vision_d_model)
        
        # 编码文本（不使用 Cross-Attention，仅用于 ITC）
        text_hidden_states = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=None  # ITC 不使用视觉特征
        )
        
        # 获取 [CLS] token 的隐藏状态（第一个位置）
        text_features = text_hidden_states[:, 0]  # (batch, text_d_model)
        
        # 投影到共享嵌入空间
        image_embeds = self.visual_projection(image_features)  # (batch, projection_dim)
        text_embeds = self.text_projection(text_features)  # (batch, projection_dim)
        
        # L2 归一化
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # 计算温度
        logit_scale = torch.clamp(self.logit_scale, max=100.0)
        temperature = torch.exp(logit_scale)
        
        # 计算相似度矩阵
        logits_per_image = temperature * (image_embeds @ text_embeds.T)
        logits_per_text = logits_per_image.T
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
    
    def forward_itm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Image-Text Matching 前向传播
        
        判断图文是否匹配。使用 Cross-Attention 融合视觉特征。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
        
        Returns:
            itm_logits: (batch, 2) 匹配/不匹配的 logits
        """
        # 编码图像
        image_output = self.encode_image(pixel_values)
        image_features = image_output['image_features']  # (batch, num_patches, d_model)
        
        # 编码文本（使用 Cross-Attention 融合视觉特征）
        text_hidden_states = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_features
        )
        
        # 获取 [CLS] token 的隐藏状态
        cls_hidden_state = text_hidden_states[:, 0]  # (batch, d_model)
        
        # ITM 分类
        itm_logits = self.itm_head(cls_hidden_state)  # (batch, 2)
        
        return itm_logits
    
    def forward_itg(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Image-grounded Text Generation 前向传播
        
        基于图像生成文本，计算语言模型损失。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            labels: (batch, seq_len) 目标 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
        
        Returns:
            lm_loss: scalar 语言模型损失
        """
        # 编码图像
        image_output = self.encode_image(pixel_values)
        image_features = image_output['image_features']  # (batch, num_patches, d_model)
        
        # 解码文本（使用 Cross-Attention 融合视觉特征）
        decoder_output = self.text_decoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_features,
            labels=labels
        )
        
        return decoder_output['loss']
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        完整前向传播（计算所有三个任务的输出）
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
            labels: (batch, seq_len) 目标 token IDs（用于 ITG）
            return_loss: 是否返回损失
        
        Returns:
            Dict 包含:
            - 'itc_output': ITC 任务输出
            - 'itm_logits': ITM 任务 logits
            - 'itg_loss': ITG 任务损失（如果提供 labels）
        """
        output = {}
        
        # ITC
        itc_output = self.forward_itc(pixel_values, input_ids, attention_mask)
        output['itc_output'] = itc_output
        
        # ITM
        itm_logits = self.forward_itm(pixel_values, input_ids, attention_mask)
        output['itm_logits'] = itm_logits
        
        # ITG
        if labels is not None:
            itg_loss = self.forward_itg(pixel_values, input_ids, labels, attention_mask)
            output['itg_loss'] = itg_loss
        
        return output
    
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        生成图像描述
        
        使用贪婪解码生成图像的文本描述。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            max_length: 最大生成长度
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
            pad_token_id: 填充 token ID
        
        Returns:
            generated_ids: (batch, generated_len) 生成的 token IDs
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 编码图像
        image_output = self.encode_image(pixel_values)
        image_features = image_output['image_features']
        
        # 初始化输入（BOS token）
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        # 自回归生成
        for _ in range(max_length - 1):
            # 解码
            decoder_output = self.text_decoder(
                input_ids,
                encoder_hidden_states=image_features
            )
            
            # 获取下一个 token
            next_token_logits = decoder_output['logits'][:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # 检查是否所有序列都生成了 EOS
            if (next_token_id == eos_token_id).all():
                break
        
        return input_ids
    
    def visual_question_answering(
        self,
        pixel_values: torch.Tensor,
        question_ids: torch.Tensor,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        视觉问答
        
        给定图像和问题，生成答案。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            question_ids: (batch, question_len) 问题 token IDs
            max_length: 最大生成长度
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
        
        Returns:
            answer_ids: (batch, answer_len) 答案 token IDs
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 编码图像
        image_output = self.encode_image(pixel_values)
        image_features = image_output['image_features']
        
        # 将问题作为前缀，然后生成答案
        # 在问题后添加 BOS token 作为答案的开始
        input_ids = torch.cat([
            question_ids,
            torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        ], dim=1)
        
        question_len = question_ids.shape[1]
        
        # 自回归生成答案
        for _ in range(max_length):
            # 解码
            decoder_output = self.text_decoder(
                input_ids,
                encoder_hidden_states=image_features
            )
            
            # 获取下一个 token
            next_token_logits = decoder_output['logits'][:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # 拼接
            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # 检查是否所有序列都生成了 EOS
            if (next_token_id == eos_token_id).all():
                break
        
        # 返回答案部分（不含问题）
        answer_ids = input_ids[:, question_len + 1:]
        
        return answer_ids


def itc_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor
) -> torch.Tensor:
    """
    计算 ITC 损失（InfoNCE Loss）
    
    双向对比损失：(image->text + text->image) / 2
    
    Args:
        logits_per_image: (batch, batch) 图像到文本的相似度矩阵
        logits_per_text: (batch, batch) 文本到图像的相似度矩阵
    
    Returns:
        loss: scalar ITC 损失
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)
    
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i2t + loss_t2i) / 2


def itm_loss(
    itm_logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    计算 ITM 损失
    
    二分类交叉熵损失。
    
    Args:
        itm_logits: (batch, 2) ITM logits
        labels: (batch,) 标签，1=匹配，0=不匹配
    
    Returns:
        loss: scalar ITM 损失
    """
    return F.cross_entropy(itm_logits, labels)
