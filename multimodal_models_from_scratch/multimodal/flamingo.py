"""
Flamingo 模型实现

Flamingo 是 DeepMind 提出的视觉-语言交错输入多模态模型，支持多图像输入。

核心组件:
- 冻结的 Vision Encoder (ViT)
- Perceiver Resampler: 将可变长度视觉特征压缩为固定数量的 latent 向量
- 冻结的 LLM (LLaMA)，每隔 N 层插入 Gated Cross Attention
- Gated Cross Attention: 使文本 token 能够关注视觉 token

参考: Flamingo 论文 https://arxiv.org/abs/2204.14198

需求: 8.1, 8.4, 8.5, 8.6, 8.7, 8.8
"""

from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_models_from_scratch.config import FlamingoConfig, VisionConfig, LLaMAConfig
from multimodal_models_from_scratch.vision.vit import ViTModel
from multimodal_models_from_scratch.llm.llama import LLaMAModel, LLaMADecoderLayer
from multimodal_models_from_scratch.llm.rmsnorm import RMSNorm
from multimodal_models_from_scratch.llm.swiglu import SwiGLU
from multimodal_models_from_scratch.llm.gqa import GroupedQueryAttention
from multimodal_models_from_scratch.multimodal.perceiver import PerceiverResampler
from multimodal_models_from_scratch.multimodal.gated_cross_attention import GatedCrossAttentionLayer


class FlamingoDecoderLayer(nn.Module):
    """
    Flamingo Decoder 层
    
    包装 LLaMADecoderLayer，可选地在自注意力之前插入 Gated Cross Attention。
    
    结构:
    1. (可选) Gated Cross Attention: 文本 token 关注视觉 token
    2. LLaMA Decoder Layer: 自注意力 + FFN
    
    Args:
        config: LLaMA 模型配置
        layer_idx: 层索引
        has_cross_attention: 是否包含门控交叉注意力层
    
    Examples:
        >>> config = LLaMAConfig(d_model=768, num_heads=12, num_kv_heads=4, num_layers=12, d_ff=3072)
        >>> layer = FlamingoDecoderLayer(config, layer_idx=0, has_cross_attention=True)
        >>> hidden_states = torch.randn(2, 128, 768)
        >>> visual_features = torch.randn(2, 64, 768)
        >>> output, _ = layer(hidden_states, visual_features=visual_features)
    """
    
    def __init__(
        self,
        config: LLaMAConfig,
        layer_idx: int,
        has_cross_attention: bool = False
    ):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.has_cross_attention = has_cross_attention
        self.d_model = config.d_model
        
        # 门控交叉注意力层（可选）
        self.gated_cross_attn = None
        if has_cross_attention:
            self.gated_cross_attn = GatedCrossAttentionLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                dropout_rate=config.dropout_rate
            )
        
        # LLaMA Decoder Layer
        self.llama_layer = LLaMADecoderLayer(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        visual_features: Optional[torch.Tensor] = None,
        visual_attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入张量，形状为 (batch, seq_len, d_model)
            attention_mask: 注意力掩码
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: KV Cache
            use_cache: 是否返回更新后的 KV Cache
            visual_features: 视觉特征，形状为 (batch, num_visual_tokens, d_model)
            visual_attention_mask: 视觉特征掩码
        
        Returns:
            hidden_states: 输出张量
            past_key_value: 如果 use_cache=True，返回更新后的 KV Cache
        """
        # 1. 门控交叉注意力（如果有）
        if self.gated_cross_attn is not None and visual_features is not None:
            hidden_states = self.gated_cross_attn(
                hidden_states=hidden_states,
                visual_features=visual_features,
                visual_attention_mask=visual_attention_mask
            )
        
        # 2. LLaMA Decoder Layer
        hidden_states, present_key_value = self.llama_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        return hidden_states, present_key_value
    
    def extra_repr(self) -> str:
        return (
            f'layer_idx={self.layer_idx}, '
            f'has_cross_attention={self.has_cross_attention}, '
            f'd_model={self.d_model}'
        )


class FlamingoModel(nn.Module):
    """
    Flamingo 模型
    
    支持视觉-语言交错输入的多模态模型，可处理多图像输入。
    
    组件:
    - 冻结的 Vision Encoder (ViT)
    - Perceiver Resampler: 将视觉特征压缩为固定数量的 latent 向量
    - 修改后的 LLM: 每隔 N 层插入 Gated Cross Attention
    
    Args:
        config: Flamingo 模型配置
    
    Examples:
        >>> config = FlamingoConfig()
        >>> model = FlamingoModel(config)
        >>> input_ids = torch.randint(0, 32000, (2, 128))
        >>> images = torch.randn(2, 3, 3, 224, 224)  # 2 samples, 3 images each
        >>> output = model(input_ids, images)
    """
    
    def __init__(self, config: FlamingoConfig):
        super().__init__()
        
        self.config = config
        self.vision_config = config.vision_config
        self.llm_config = config.llm_config
        
        # Vision Encoder (ViT)
        self.vision_encoder = ViTModel(config.vision_config)
        
        # Perceiver Resampler
        self.perceiver_resampler = PerceiverResampler(
            d_model=config.vision_config.d_model,
            num_latents=config.perceiver_num_latents,
            num_heads=config.vision_config.num_heads,
            num_layers=config.perceiver_depth,
            dropout_rate=config.vision_config.dropout_rate
        )
        
        # 视觉特征投影层（如果视觉维度与 LLM 维度不同）
        self.visual_projection = None
        if config.vision_config.d_model != config.llm_config.d_model:
            self.visual_projection = nn.Linear(
                config.vision_config.d_model,
                config.llm_config.d_model
            )
        
        # 构建修改后的 LLM
        self._build_llm_with_cross_attention()
        
        # 冻结参数
        if config.freeze_vision:
            self.freeze_vision_encoder()
        if config.freeze_llm:
            self.freeze_llm()
    
    def _build_llm_with_cross_attention(self):
        """
        构建带有门控交叉注意力的 LLM
        
        每隔 cross_attn_every_n_layers 层插入一个门控交叉注意力层
        """
        config = self.llm_config
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # RoPE 位置编码
        from multimodal_models_from_scratch.llm.rope import RotaryPositionEmbedding
        head_dim = config.d_model // config.num_heads
        self.rotary_emb = RotaryPositionEmbedding(
            d_model=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        # 构建 Decoder Layers，每隔 N 层插入交叉注意力
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            # 每隔 cross_attn_every_n_layers 层插入交叉注意力
            # 例如 cross_attn_every_n_layers=4 时，在第 3, 7, 11, ... 层插入
            has_cross_attention = (i + 1) % self.config.cross_attn_every_n_layers == 0
            self.layers.append(
                FlamingoDecoderLayer(
                    config=config,
                    layer_idx=i,
                    has_cross_attention=has_cross_attention
                )
            )
        
        # Final RMSNorm
        self.norm = RMSNorm(config.d_model)
        
        # LM Head
        self.lm_head = nn.Linear(config.vocab_size, config.d_model, bias=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        if not self.llm_config.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        if self.visual_projection is not None:
            nn.init.normal_(self.visual_projection.weight, mean=0.0, std=0.02)
    
    def freeze_vision_encoder(self):
        """冻结 Vision Encoder 参数"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def freeze_llm(self):
        """
        冻结 LLM 参数（门控交叉注意力层除外）
        
        Flamingo 的训练策略是冻结预训练的 LLM，只训练:
        - Perceiver Resampler
        - Gated Cross Attention 层
        - Visual Projection（如果有）
        """
        # 冻结 Token Embedding
        for param in self.embed_tokens.parameters():
            param.requires_grad = False
        
        # 冻结 LM Head
        for param in self.lm_head.parameters():
            param.requires_grad = False
        
        # 冻结 Final Norm
        for param in self.norm.parameters():
            param.requires_grad = False
        
        # 冻结 Decoder Layers 中的 LLaMA 部分，但保留 Gated Cross Attention
        for layer in self.layers:
            # 冻结 LLaMA Decoder Layer
            for param in layer.llama_layer.parameters():
                param.requires_grad = False
            
            # 保持 Gated Cross Attention 可训练
            if layer.gated_cross_attn is not None:
                for param in layer.gated_cross_attn.parameters():
                    param.requires_grad = True
    
    def unfreeze_all(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def encode_images(
        self,
        images: torch.Tensor
    ) -> torch.Tensor:
        """
        编码图像
        
        处理多图像输入，通过 ViT 和 Perceiver Resampler 获取视觉特征。
        
        Args:
            images: 图像张量
                - 单图像: (batch, 3, H, W)
                - 多图像: (batch, num_images, 3, H, W)
        
        Returns:
            visual_features: 视觉特征
                - 单图像: (batch, num_latents, d_model)
                - 多图像: (batch, num_images * num_latents, d_model)
        """
        # 处理输入维度
        if images.dim() == 4:
            # 单图像: (batch, 3, H, W)
            batch_size = images.size(0)
            num_images = 1
            images_flat = images
        elif images.dim() == 5:
            # 多图像: (batch, num_images, 3, H, W)
            batch_size, num_images, c, h, w = images.shape
            images_flat = images.view(batch_size * num_images, c, h, w)
        else:
            raise ValueError(f"Expected 4D or 5D image tensor, got {images.dim()}D")
        
        # 通过 Vision Encoder 获取图像特征
        # (batch * num_images, 3, H, W) -> (batch * num_images, num_patches, d_model)
        image_features = self.vision_encoder.get_image_features(images_flat)
        
        # 通过 Perceiver Resampler 压缩特征
        # (batch * num_images, num_patches, d_model) -> (batch * num_images, num_latents, d_model)
        visual_features = self.perceiver_resampler(image_features)
        
        # 投影到 LLM 维度（如果需要）
        if self.visual_projection is not None:
            visual_features = self.visual_projection(visual_features)
        
        # 重塑为 (batch, num_images * num_latents, d_model)
        num_latents = visual_features.size(1)
        d_model = visual_features.size(2)
        visual_features = visual_features.view(batch_size, num_images * num_latents, d_model)
        
        return visual_features
    
    def _create_causal_mask(
        self,
        seq_len: int,
        past_key_values_length: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """创建因果注意力掩码"""
        total_len = seq_len + past_key_values_length
        mask = torch.ones(seq_len, total_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=past_key_values_length)
        causal_mask = torch.zeros(seq_len, total_len, device=device, dtype=dtype)
        causal_mask.masked_fill_(~mask, float('-inf'))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        return causal_mask
    
    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        past_key_values_length: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """准备注意力掩码"""
        total_len = seq_len + past_key_values_length
        causal_mask = self._create_causal_mask(
            seq_len, past_key_values_length, device, dtype
        )
        
        if attention_mask is None:
            return causal_mask.expand(batch_size, -1, -1, -1)
        
        mask_len = attention_mask.shape[1]
        if mask_len < total_len:
            past_mask = torch.ones(batch_size, total_len - mask_len, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([past_mask, attention_mask], dim=1)
        elif mask_len > total_len:
            attention_mask = attention_mask[:, -total_len:]
        
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
        inverted_mask = 1.0 - expanded_mask
        expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), float('-inf'))
        combined_mask = torch.minimum(causal_mask, expanded_mask)
        
        return combined_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            images: 图像张量
                - 单图像: (batch, 3, H, W)
                - 多图像: (batch, num_images, 3, H, W)
                - None: 纯文本输入
            attention_mask: 注意力掩码，形状为 (batch, seq_len)
            position_ids: 位置 ID
            past_key_values: KV Cache
            use_cache: 是否返回更新后的 KV Cache
            labels: 标签，用于计算语言模型损失
            output_hidden_states: 是否返回所有层的隐藏状态
        
        Returns:
            {
                'logits': (batch, seq_len, vocab_size),
                'loss': scalar if labels provided,
                'hidden_states': (batch, seq_len, d_model),
                'past_key_values': List[Tuple] if use_cache,
                'all_hidden_states': Tuple if output_hidden_states
            }
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 编码图像（如果有）
        visual_features = None
        if images is not None:
            visual_features = self.encode_images(images)
        
        # 计算 KV Cache 长度
        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        # 生成位置 ID
        if position_ids is None:
            position_ids = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Token Embedding
        hidden_states = self.embed_tokens(input_ids)
        dtype = hidden_states.dtype
        
        # 准备注意力掩码
        combined_mask = self._prepare_attention_mask(
            attention_mask, batch_size, seq_len,
            past_key_values_length, device, dtype
        )
        
        # 计算 RoPE 位置编码
        max_pos = position_ids.max().item() + 1
        self.rotary_emb._extend_cache(int(max_pos))
        
        cos = self.rotary_emb.cos_cached[:, :, :int(max_pos), :]
        sin = self.rotary_emb.sin_cached[:, :, :int(max_pos), :]
        
        pos_flat = position_ids.reshape(-1)
        cos_selected = cos[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        sin_selected = sin[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        
        position_embeddings = (
            cos_selected.unsqueeze(1),
            sin_selected.unsqueeze(1)
        )
        
        # 存储所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        
        # 存储新的 KV Cache
        present_key_values = [] if use_cache else None
        
        # 通过所有 Decoder 层
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 获取当前层的 past_key_value
            past_key_value = None
            if past_key_values is not None and i < len(past_key_values):
                past_key_value = past_key_values[i]
            
            # 前向传播
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_mask=combined_mask,
                position_embeddings=position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                visual_features=visual_features
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 移位标签以计算下一个 token 预测损失
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 构建输出字典
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if use_cache:
            output['past_key_values'] = present_key_values
        
        if output_hidden_states:
            output['all_hidden_states'] = all_hidden_states
        
        return output
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成所需的输入
        
        Args:
            input_ids: 输入 token ID
            past_key_values: KV Cache
            attention_mask: 注意力掩码
            images: 图像张量（仅在第一次调用时使用）
            **kwargs: 其他参数
        
        Returns:
            准备好的输入字典
        """
        # 如果有 KV Cache，只需要最后一个 token
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, -1:]
            position_ids = torch.tensor(
                [[past_length]],
                device=input_ids.device
            ).expand(input_ids.shape[0], -1)
            # 在后续生成步骤中不需要再传入图像
            images = None
        else:
            past_length = 0
            position_ids = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': True,
            'images': images,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        多图像条件生成
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            images: 图像张量
                - 单图像: (batch, 3, H, W)
                - 多图像: (batch, num_images, 3, H, W)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            eos_token_id: 结束 token ID
            pad_token_id: padding token ID
        
        Returns:
            generated_ids: 生成的 token ID
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化
        generated_ids = input_ids
        past_key_values = None
        attention_mask = torch.ones_like(input_ids)
        
        # 记录每个序列是否已结束
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # 第一次前向传播时传入图像
        current_images = images
        
        for _ in range(max_new_tokens):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                images=current_images,
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs)
            
            # 获取最后一个 token 的 logits
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 更新 KV Cache
            past_key_values = outputs.get('past_key_values')
            
            # 后续步骤不再需要图像
            current_images = None
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪解码
            if do_sample:
                filtered_logits = next_token_logits.clone()
                
                # Top-K 采样
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    top_k_values, _ = torch.topk(filtered_logits, k)
                    threshold = top_k_values[..., -1, None]
                    indices_to_remove = filtered_logits < threshold
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Top-P 采样
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # 计算概率分布
                max_logits = filtered_logits.max(dim=-1, keepdim=True)[0]
                safe_logits = torch.where(
                    filtered_logits == float('-inf'),
                    torch.full_like(filtered_logits, -1e10),
                    filtered_logits - max_logits
                )
                probs = torch.softmax(safe_logits, dim=-1)
                
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # 对于已结束的序列，使用 pad_token_id
            if pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            # 拼接生成的 token
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新 attention_mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)
            
            # 检查是否遇到 EOS
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
                if unfinished_sequences.max() == 0:
                    break
        
        return generated_ids
    
    def get_num_trainable_params(self) -> int:
        """获取可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_total_params(self) -> int:
        """获取总参数数量"""
        return sum(p.numel() for p in self.parameters())
    
    def extra_repr(self) -> str:
        return (
            f'perceiver_num_latents={self.config.perceiver_num_latents}, '
            f'cross_attn_every_n_layers={self.config.cross_attn_every_n_layers}, '
            f'freeze_vision={self.config.freeze_vision}, '
            f'freeze_llm={self.config.freeze_llm}'
        )
