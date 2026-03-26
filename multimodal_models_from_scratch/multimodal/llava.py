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
        
        # 将 image_token_index 替换为 0（或其他有效 token）以便进行嵌入
        # 这些位置的嵌入会被视觉 token 替换，所以具体值不重要
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == image_token_index] = 0
        
        # 获取文本嵌入
        text_embeds = self.llm.embed_tokens(safe_input_ids)  # (batch, seq_len, llm_dim)
        
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
            # 找到当前样本的 <image> 位置
            sample_image_positions = image_positions[image_positions[:, 0] == i]
            
            if sample_image_positions.shape[0] == 0:
                # 没有 <image> token，直接复制文本嵌入
                merged_embeds[i, :seq_len] = text_embeds[i]
                continue
            
            # 取第一个 <image> 位置
            img_pos = sample_image_positions[0, 1].item()
            
            # 复制 <image> 之前的文本嵌入
            if img_pos > 0:
                merged_embeds[i, :img_pos] = text_embeds[i, :img_pos]
            
            # 插入视觉 token
            merged_embeds[i, img_pos:img_pos + num_visual_tokens] = visual_tokens[i]
            
            # 复制 <image> 之后的文本嵌入
            remaining_len = seq_len - img_pos - 1
            if remaining_len > 0:
                merged_embeds[i, img_pos + num_visual_tokens:img_pos + num_visual_tokens + remaining_len] = \
                    text_embeds[i, img_pos + 1:seq_len]
        
        # 生成位置 ID
        position_ids = torch.arange(new_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        return merged_embeds, merged_attention_mask, position_ids
    
    def _create_labels_mask(
        self,
        labels: torch.Tensor,
        input_ids: torch.Tensor,
        num_visual_tokens: int,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID
    ) -> torch.Tensor:
        """
        创建调整后的 labels，将视觉 token 位置的 label 设为 -100（不计算损失）
        
        Args:
            labels: 原始 labels，形状为 (batch, seq_len)
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            num_visual_tokens: 视觉 token 数量
            image_token_index: <image> token 的 ID
        
        Returns:
            new_labels: 调整后的 labels，形状为 (batch, new_seq_len)
        """
        batch_size, seq_len = labels.shape
        device = labels.device
        
        # 找到 <image> token 位置
        image_positions = (input_ids == image_token_index).nonzero(as_tuple=False)
        
        # 如果没有 <image> token，直接返回原始 labels
        if image_positions.shape[0] == 0:
            return labels
        
        # 计算新序列长度
        new_seq_len = seq_len - 1 + num_visual_tokens
        
        # 创建新的 labels，初始化为 -100（忽略）
        new_labels = torch.full(
            (batch_size, new_seq_len),
            fill_value=-100,
            device=device,
            dtype=labels.dtype
        )
        
        # 为每个样本处理
        for i in range(batch_size):
            sample_image_positions = image_positions[image_positions[:, 0] == i]
            
            if sample_image_positions.shape[0] == 0:
                # 没有 <image> token，直接复制 labels
                new_labels[i, :seq_len] = labels[i]
                continue
            
            img_pos = sample_image_positions[0, 1].item()
            
            # 复制 <image> 之前的 labels
            if img_pos > 0:
                new_labels[i, :img_pos] = labels[i, :img_pos]
            
            # 视觉 token 位置保持 -100（不计算损失）
            
            # 复制 <image> 之后的 labels
            remaining_len = seq_len - img_pos - 1
            if remaining_len > 0:
                new_labels[i, img_pos + num_visual_tokens:img_pos + num_visual_tokens + remaining_len] = \
                    labels[i, img_pos + 1:seq_len]
        
        return new_labels

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播
        
        将视觉 token 插入到 <image> 位置，然后通过 LLM 处理。
        
        Args:
            pixel_values: 输入图像，形状为 (batch, 3, H, W)，可选
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            attention_mask: 注意力掩码，形状为 (batch, seq_len)
            labels: 训练标签，形状为 (batch, seq_len)，-100 表示忽略
            image_token_index: <image> token 的 ID
            past_key_values: KV Cache
            use_cache: 是否返回 KV Cache
            output_hidden_states: 是否返回所有隐藏状态
        
        Returns:
            {
                'logits': (batch, new_seq_len, vocab_size),
                'loss': scalar if labels provided,
                'hidden_states': (batch, new_seq_len, d_model),
                'past_key_values': List[Tuple] if use_cache
            }
        """
        # 如果有图像输入，获取视觉特征
        visual_tokens = None
        if pixel_values is not None:
            visual_tokens = self.get_vision_features(pixel_values)
            num_visual_tokens = visual_tokens.shape[1]
        else:
            num_visual_tokens = 0
        
        # 如果有视觉 token 且有 <image> token，进行融合
        if visual_tokens is not None and (input_ids == image_token_index).any():
            # 融合视觉和文本嵌入
            inputs_embeds, merged_attention_mask, position_ids = self._merge_visual_tokens(
                input_ids=input_ids,
                visual_tokens=visual_tokens,
                image_token_index=image_token_index
            )
            
            # 如果提供了 attention_mask，需要相应调整
            if attention_mask is not None:
                # 扩展 attention_mask 以适应新的序列长度
                batch_size = input_ids.shape[0]
                new_seq_len = inputs_embeds.shape[1]
                
                # 创建新的 attention_mask
                new_attention_mask = torch.ones(
                    batch_size, new_seq_len,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype
                )
                
                # 根据原始 attention_mask 调整
                image_positions = (input_ids == image_token_index).nonzero(as_tuple=False)
                for i in range(batch_size):
                    sample_positions = image_positions[image_positions[:, 0] == i]
                    if sample_positions.shape[0] > 0:
                        img_pos = sample_positions[0, 1].item()
                        # 复制 <image> 之前的 mask
                        if img_pos > 0:
                            new_attention_mask[i, :img_pos] = attention_mask[i, :img_pos]
                        # 视觉 token 位置设为 1（有效）
                        new_attention_mask[i, img_pos:img_pos + num_visual_tokens] = 1
                        # 复制 <image> 之后的 mask
                        remaining_len = attention_mask.shape[1] - img_pos - 1
                        if remaining_len > 0:
                            new_attention_mask[i, img_pos + num_visual_tokens:img_pos + num_visual_tokens + remaining_len] = \
                                attention_mask[i, img_pos + 1:]
                    else:
                        new_attention_mask[i, :attention_mask.shape[1]] = attention_mask[i]
                
                merged_attention_mask = new_attention_mask
            
            # 调整 labels
            if labels is not None:
                labels = self._create_labels_mask(
                    labels=labels,
                    input_ids=input_ids,
                    num_visual_tokens=num_visual_tokens,
                    image_token_index=image_token_index
                )
            
            # 通过 LLM 处理（使用 inputs_embeds 而不是 input_ids）
            # 需要直接调用 LLM 的内部组件
            hidden_states = inputs_embeds
            
            # 准备注意力掩码
            batch_size, seq_len = hidden_states.shape[:2]
            dtype = hidden_states.dtype
            device = hidden_states.device
            
            # 计算 KV Cache 长度
            past_key_values_length = 0
            if past_key_values is not None and len(past_key_values) > 0:
                past_key_values_length = past_key_values[0][0].shape[2]
            
            # 创建因果掩码
            combined_mask = self.llm._prepare_attention_mask(
                merged_attention_mask, batch_size, seq_len,
                past_key_values_length, device, dtype
            )
            
            # 计算 RoPE 位置编码
            max_pos = seq_len + past_key_values_length
            self.llm.rotary_emb._extend_cache(max_pos)
            
            cos = self.llm.rotary_emb.cos_cached[:, :, :max_pos, :]
            sin = self.llm.rotary_emb.sin_cached[:, :, :max_pos, :]
            
            # 根据 position_ids 选择对应的位置编码
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
            for i, layer in enumerate(self.llm.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                past_key_value = None
                if past_key_values is not None and i < len(past_key_values):
                    past_key_value = past_key_values[i]
                
                hidden_states, present_key_value = layer(
                    hidden_states=hidden_states,
                    attention_mask=combined_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                
                if use_cache:
                    present_key_values.append(present_key_value)
            
            # 最终归一化
            hidden_states = self.llm.norm(hidden_states)
            
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # LM Head
            logits = self.llm.lm_head(hidden_states)
            
        else:
            # 没有图像或没有 <image> token，直接使用 LLM
            # 如果 input_ids 中包含 image_token_index，需要替换为有效 token
            safe_input_ids = input_ids.clone()
            if (safe_input_ids == image_token_index).any():
                safe_input_ids[safe_input_ids == image_token_index] = 0
            
            outputs = self.llm(
                input_ids=safe_input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states
            )
            
            logits = outputs['logits']
            hidden_states = outputs['hidden_states']
            present_key_values = outputs.get('past_key_values')
            all_hidden_states = outputs.get('all_hidden_states')
        
        # 计算损失
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # 构建输出
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
        if loss is not None:
            output['loss'] = loss
        
        if use_cache:
            output['past_key_values'] = present_key_values
        
        if output_hidden_states and all_hidden_states is not None:
            output['all_hidden_states'] = all_hidden_states
        
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID,
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成所需的输入
        
        在自回归生成过程中，如果有 KV Cache，只需要处理最后一个 token。
        第一次调用时处理完整输入（包括图像），后续调用只处理新生成的 token。
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            pixel_values: 输入图像，形状为 (batch, 3, H, W)
            past_key_values: KV Cache
            attention_mask: 注意力掩码
            image_token_index: <image> token 的 ID
            **kwargs: 其他参数
        
        Returns:
            准备好的输入字典
        """
        # 如果有 KV Cache，说明不是第一次调用
        if past_key_values is not None and len(past_key_values) > 0:
            # 只需要最后一个 token
            input_ids = input_ids[:, -1:]
            # 图像已经在第一次调用时处理过了
            pixel_values = None
        
        return {
            'input_ids': input_ids,
            'pixel_values': pixel_values,
            'attention_mask': attention_mask,
            'past_key_values': past_key_values,
            'use_cache': True,
            'image_token_index': image_token_index,
        }
    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID,
    ) -> torch.Tensor:
        """
        视觉对话生成
        
        Args:
            pixel_values: 输入图像，形状为 (batch, 3, H, W)
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            eos_token_id: 结束 token ID
            pad_token_id: padding token ID
            image_token_index: <image> token 的 ID
        
        Returns:
            generated_ids: 生成的 token ID
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化
        generated_ids = input_ids.clone()
        past_key_values = None
        attention_mask = torch.ones_like(input_ids)
        
        # 记录每个序列是否已结束
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        # 第一次调用时处理图像
        current_pixel_values = pixel_values
        
        for step in range(max_new_tokens):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids,
                pixel_values=current_pixel_values,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                image_token_index=image_token_index,
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs)
            
            # 获取最后一个 token 的 logits
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 更新 KV Cache
            past_key_values = outputs.get('past_key_values')
            
            # 第一次调用后，不再需要图像
            current_pixel_values = None
            
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
    
    def prepare_conversation_input(
        self,
        conversations: List[Dict[str, str]],
        tokenizer: Any,
        image_token: str = "<image>",
        system_prompt: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        准备多轮对话输入
        
        将对话格式转换为模型输入，并创建 labels（仅对 assistant 回复计算损失）。
        
        Args:
            conversations: 对话列表，格式为 [{'role': 'user/assistant', 'content': str}, ...]
            tokenizer: 分词器
            image_token: 图像占位符 token
            system_prompt: 系统提示（可选）
        
        Returns:
            input_ids: 输入 token ID
            labels: 训练标签（user 部分为 -100）
        """
        # 构建完整的对话文本
        full_text = ""
        label_mask = []  # 记录哪些部分需要计算损失
        
        if system_prompt:
            full_text += f"System: {system_prompt}\n\n"
            # 系统提示不计算损失
            system_tokens = tokenizer.encode(f"System: {system_prompt}\n\n")
            label_mask.extend([False] * len(system_tokens))
        
        for turn in conversations:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                turn_text = f"User: {content}\n"
                full_text += turn_text
                # User 部分不计算损失
                turn_tokens = tokenizer.encode(turn_text)
                label_mask.extend([False] * len(turn_tokens))
            elif role == 'assistant':
                turn_text = f"Assistant: {content}\n"
                full_text += turn_text
                # Assistant 部分计算损失
                turn_tokens = tokenizer.encode(turn_text)
                label_mask.extend([True] * len(turn_tokens))
        
        # 编码完整文本
        input_ids = tokenizer.encode(full_text)
        input_ids = torch.tensor([input_ids], dtype=torch.long)
        
        # 创建 labels
        labels = input_ids.clone()
        
        # 将不需要计算损失的位置设为 -100
        # 注意：由于分词可能导致 token 数量不完全匹配，这里使用简化处理
        # 实际应用中需要更精确的对齐
        for i, should_compute_loss in enumerate(label_mask):
            if i < labels.shape[1] and not should_compute_loss:
                labels[0, i] = -100
        
        return input_ids, labels
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"vision_dim={self.config.vision_config.d_model}, "
            f"llm_dim={self.config.llm_config.d_model}, "
            f"projection_type='{self.config.projection_type}', "
            f"freeze_vision={self.config.freeze_vision}, "
            f"freeze_llm={self.config.freeze_llm}"
        )
