"""
BLIP-2 模型（Bootstrapping Language-Image Pre-training 2）

BLIP-2 使用 Q-Former 高效地桥接冻结的视觉编码器和大语言模型。

组件:
- 冻结的 Vision Encoder (ViT): 提取图像特征
- Q-Former: 使用可学习的查询向量从图像特征中提取固定数量的视觉 token
- Visual Projection: 将 Q-Former 输出投影到 LLM 的嵌入空间
- 冻结的 LLM (LLaMA): 进行视觉语言生成

两阶段训练:
- 第一阶段: 训练 Q-Former (ITC、ITM、ITG 任务)
- 第二阶段: 训练 Visual Projection，连接冻结的 LLM

需求: 6.1, 6.2, 6.6, 6.7, 6.8
"""

from typing import Dict, Optional, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from bert_gpt_from_scratch.config import TransformerConfig

from ..config import BLIP2Config, VisionConfig, LLaMAConfig
from ..vision.vit import ViTModel
from ..llm.llama import LLaMAModel
from .qformer import QFormer
from .visual_projection import VisualProjection


class BLIP2Model(nn.Module):
    """
    BLIP-2 模型
    
    组件:
    - 冻结的 Vision Encoder (ViT): 提取图像特征
    - Q-Former: 使用可学习的查询向量从图像特征中提取固定数量的视觉 token
    - Visual Projection: 将 Q-Former 输出投影到 LLM 的嵌入空间
    - 冻结的 LLM (LLaMA): 进行视觉语言生成
    
    Args:
        config: BLIP2Config 配置对象
    
    Examples:
        >>> config = BLIP2Config()
        >>> model = BLIP2Model(config)
        >>> pixel_values = torch.randn(2, 3, 224, 224)
        >>> input_ids = torch.randint(0, 32000, (2, 32))
        >>> # 第一阶段训练
        >>> stage1_output = model.forward_stage1(pixel_values, input_ids)
        >>> # 第二阶段训练
        >>> labels = torch.randint(0, 32000, (2, 32))
        >>> stage2_loss = model.forward_stage2(pixel_values, input_ids, labels)
    """
    
    def __init__(self, config: BLIP2Config):
        super().__init__()
        
        self.config = config
        
        # 1. 冻结的 Vision Encoder (ViT)
        # 创建不带分类头的 VisionConfig
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
        
        # 冻结 Vision Encoder
        self._freeze_vision_encoder()
        
        # 2. Q-Former
        self.qformer = QFormer(
            d_model=config.qformer_config.d_model,
            num_heads=config.qformer_config.num_heads,
            num_layers=config.qformer_config.num_layers,
            d_ff=config.qformer_config.d_ff,
            num_query_tokens=config.num_query_tokens,
            dropout_rate=config.qformer_config.dropout_rate
        )
        
        # 3. Visual Projection (将 Q-Former 输出投影到 LLM 嵌入空间)
        self.visual_projection = VisualProjection(
            vision_dim=config.qformer_config.d_model,
            llm_dim=config.llm_config.d_model,
            projection_type='linear'
        )
        
        # 4. 冻结的 LLM (LLaMA)
        self.llm = LLaMAModel(config.llm_config)
        
        # 冻结 LLM
        self._freeze_llm()
        
        # 第一阶段训练组件
        # ITC: 图文对比学习投影头
        self.vision_proj_itc = nn.Linear(
            config.qformer_config.d_model,
            config.projection_dim,
            bias=False
        )
        self.text_proj_itc = nn.Linear(
            config.qformer_config.d_model,
            config.projection_dim,
            bias=False
        )
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(
            torch.tensor([torch.log(torch.tensor(1.0 / 0.07)).item()])
        )
        
        # ITM: 图文匹配分类头
        self.itm_head = nn.Linear(config.qformer_config.d_model, 2)
        
        # ITG: 文本生成解码器（用于第一阶段）
        # 使用简单的 LM Head 进行文本生成
        self.itg_head = nn.Linear(
            config.qformer_config.d_model,
            config.qformer_config.vocab_size,
            bias=False
        )
        
        # 文本嵌入层（用于第一阶段的文本编码）
        self.text_embedding = nn.Embedding(
            config.qformer_config.vocab_size,
            config.qformer_config.d_model
        )
    
    def _freeze_vision_encoder(self):
        """冻结 Vision Encoder 的所有参数"""
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
    
    def _freeze_llm(self):
        """冻结 LLM 的所有参数"""
        for param in self.llm.parameters():
            param.requires_grad = False
    
    def unfreeze_llm(self):
        """解冻 LLM 的所有参数（用于全参数微调）"""
        for param in self.llm.parameters():
            param.requires_grad = True
    
    def get_trainable_params_stage1(self) -> List[nn.Parameter]:
        """获取第一阶段可训练的参数"""
        params = []
        # Q-Former
        params.extend(self.qformer.parameters())
        # ITC 投影头
        params.extend(self.vision_proj_itc.parameters())
        params.extend(self.text_proj_itc.parameters())
        params.append(self.logit_scale)
        # ITM 头
        params.extend(self.itm_head.parameters())
        # ITG 头
        params.extend(self.itg_head.parameters())
        # 文本嵌入
        params.extend(self.text_embedding.parameters())
        return params
    
    def get_trainable_params_stage2(self) -> List[nn.Parameter]:
        """获取第二阶段可训练的参数"""
        params = []
        # Visual Projection
        params.extend(self.visual_projection.parameters())
        return params
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        编码图像
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
        
        Returns:
            image_features: (batch, num_patches, vision_d_model) 图像特征
        """
        with torch.no_grad():
            vision_output = self.vision_encoder(pixel_values)
            # 获取所有 patch 的特征（不含 [CLS] token）
            image_features = vision_output['last_hidden_state'][:, 1:]
        return image_features
    
    def forward_stage1(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        itm_labels: Optional[torch.Tensor] = None,
        itg_labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        第一阶段训练前向传播 (训练 Q-Former)
        
        包含三个任务:
        - ITC (Image-Text Contrastive): 图文对比学习
        - ITM (Image-Text Matching): 图文匹配判断
        - ITG (Image-grounded Text Generation): 基于图像的文本生成
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
            itm_labels: (batch,) ITM 标签，1=匹配，0=不匹配
            itg_labels: (batch, seq_len) ITG 目标 token IDs
        
        Returns:
            Dict 包含:
            - 'itc_loss': scalar ITC 损失
            - 'itm_loss': scalar ITM 损失（如果提供 itm_labels）
            - 'itg_loss': scalar ITG 损失（如果提供 itg_labels）
            - 'image_embeds': (batch, projection_dim) 图像嵌入
            - 'text_embeds': (batch, projection_dim) 文本嵌入
            - 'itm_logits': (batch, 2) ITM logits
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 1. 编码图像
        image_features = self.encode_image(pixel_values)
        
        # 2. Q-Former 提取视觉 token
        query_output = self.qformer(image_features)  # (batch, num_query_tokens, d_model)
        
        # 3. ITC: 图文对比学习
        # 图像嵌入：使用 Q-Former 输出的平均池化
        image_embeds = query_output.mean(dim=1)  # (batch, d_model)
        image_embeds = self.vision_proj_itc(image_embeds)  # (batch, projection_dim)
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        
        # 文本嵌入：使用文本嵌入层
        text_hidden = self.text_embedding(input_ids)  # (batch, seq_len, d_model)
        # 使用 [CLS] token（第一个位置）或平均池化
        if attention_mask is not None:
            # 使用掩码平均池化
            mask_expanded = attention_mask.unsqueeze(-1).float()
            text_embeds = (text_hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            text_embeds = text_hidden[:, 0]  # 使用第一个 token
        text_embeds = self.text_proj_itc(text_embeds)  # (batch, projection_dim)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # 计算相似度矩阵
        logit_scale = torch.clamp(self.logit_scale, max=100.0)
        temperature = torch.exp(logit_scale)
        logits_per_image = temperature * (image_embeds @ text_embeds.T)
        logits_per_text = logits_per_image.T
        
        # ITC 损失
        labels = torch.arange(batch_size, device=device)
        itc_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        
        output = {
            'itc_loss': itc_loss,
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
        }
        
        # 4. ITM: 图文匹配
        # 使用 Q-Former 输出的第一个 token 进行分类
        itm_hidden = query_output[:, 0]  # (batch, d_model)
        itm_logits = self.itm_head(itm_hidden)  # (batch, 2)
        output['itm_logits'] = itm_logits
        
        if itm_labels is not None:
            itm_loss = F.cross_entropy(itm_logits, itm_labels)
            output['itm_loss'] = itm_loss
        
        # 5. ITG: 基于图像的文本生成
        if itg_labels is not None:
            # 使用 Q-Former 输出作为条件，生成文本
            # 简化实现：将 Q-Former 输出与文本嵌入拼接
            text_hidden = self.text_embedding(input_ids)  # (batch, seq_len, d_model)
            
            # 将 query_output 作为前缀
            combined_hidden = torch.cat([query_output, text_hidden], dim=1)
            # (batch, num_query_tokens + seq_len, d_model)
            
            # 只对文本部分计算损失
            text_logits = self.itg_head(combined_hidden[:, self.config.num_query_tokens:])
            # (batch, seq_len, vocab_size)
            
            # 计算语言模型损失
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = itg_labels[:, 1:].contiguous()
            
            itg_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output['itg_loss'] = itg_loss
        
        return output
    
    def forward_stage2(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        第二阶段训练前向传播 (训练 Visual Projection)
        
        将 Q-Former 输出投影到 LLM 嵌入空间，与文本 token 拼接后输入 LLM。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            labels: (batch, seq_len) 目标 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
        
        Returns:
            lm_loss: scalar 语言模型损失
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 1. 编码图像
        image_features = self.encode_image(pixel_values)
        
        # 2. Q-Former 提取视觉 token
        with torch.no_grad():
            query_output = self.qformer(image_features)  # (batch, num_query_tokens, qformer_d_model)
        
        # 3. Visual Projection 投影到 LLM 嵌入空间
        visual_embeds = self.visual_projection(query_output)  # (batch, num_query_tokens, llm_d_model)
        
        # 4. 获取文本嵌入
        with torch.no_grad():
            text_embeds = self.llm.embed_tokens(input_ids)  # (batch, seq_len, llm_d_model)
        
        # 5. 拼接视觉和文本嵌入
        # 视觉 token 在前，文本 token 在后
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        # (batch, num_query_tokens + seq_len, llm_d_model)
        
        # 6. 准备注意力掩码
        num_visual_tokens = self.config.num_query_tokens
        visual_attention_mask = torch.ones(
            batch_size, num_visual_tokens,
            device=device, dtype=torch.long
        )
        
        if attention_mask is not None:
            combined_attention_mask = torch.cat([visual_attention_mask, attention_mask], dim=1)
        else:
            text_attention_mask = torch.ones(
                batch_size, input_ids.shape[1],
                device=device, dtype=torch.long
            )
            combined_attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)
        
        # 7. 准备标签
        # 视觉 token 部分不计算损失（设为 -100）
        visual_labels = torch.full(
            (batch_size, num_visual_tokens),
            -100,
            device=device, dtype=torch.long
        )
        combined_labels = torch.cat([visual_labels, labels], dim=1)
        
        # 8. 通过 LLM 计算损失
        # 由于 LLM 是冻结的，我们需要手动计算前向传播
        # 使用 inputs_embeds 而不是 input_ids
        hidden_states = inputs_embeds
        
        # 计算位置 ID
        seq_len = hidden_states.shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # 准备 RoPE 位置编码
        self.llm.rotary_emb._extend_cache(seq_len)
        cos = self.llm.rotary_emb.cos_cached[:, :, :seq_len, :]
        sin = self.llm.rotary_emb.sin_cached[:, :, :seq_len, :]
        
        pos_flat = position_ids.reshape(-1)
        cos_selected = cos[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        sin_selected = sin[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        position_embeddings = (cos_selected.unsqueeze(1), sin_selected.unsqueeze(1))
        
        # 创建因果掩码
        causal_mask = self.llm._create_causal_mask(
            seq_len, 0, device, hidden_states.dtype
        )
        
        # 合并 padding 掩码
        if combined_attention_mask is not None:
            expanded_mask = combined_attention_mask.unsqueeze(1).unsqueeze(2).to(hidden_states.dtype)
            inverted_mask = 1.0 - expanded_mask
            expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), float('-inf'))
            combined_mask = torch.minimum(causal_mask, expanded_mask)
        else:
            combined_mask = causal_mask.expand(batch_size, -1, -1, -1)
        
        # 通过 LLM 层
        for layer in self.llm.layers:
            hidden_states, _ = layer(
                hidden_states=hidden_states,
                attention_mask=combined_mask,
                position_embeddings=position_embeddings,
                use_cache=False
            )
        
        # 最终归一化
        hidden_states = self.llm.norm(hidden_states)
        
        # LM Head
        logits = self.llm.lm_head(hidden_states)
        
        # 计算损失
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = combined_labels[:, 1:].contiguous()
        
        lm_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return lm_loss
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        stage: int = 2
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            input_ids: (batch, seq_len) 输入 token IDs
            attention_mask: (batch, seq_len) 注意力掩码
            labels: (batch, seq_len) 目标 token IDs
            stage: 训练阶段 (1 或 2)
        
        Returns:
            Dict 包含损失和其他输出
        """
        if stage == 1:
            return self.forward_stage1(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                itg_labels=labels
            )
        else:
            loss = self.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            return {'loss': loss}

    
    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor,
        prompt_ids: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        视觉语言生成
        
        使用图像和可选的文本提示生成文本。
        
        Args:
            pixel_values: (batch, 3, H, W) 输入图像
            prompt_ids: (batch, prompt_len) 可选的文本提示 token IDs
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
            pad_token_id: 填充 token ID
        
        Returns:
            generated_ids: (batch, generated_len) 生成的 token IDs
        """
        batch_size = pixel_values.shape[0]
        device = pixel_values.device
        
        # 1. 编码图像
        image_features = self.encode_image(pixel_values)
        
        # 2. Q-Former 提取视觉 token
        query_output = self.qformer(image_features)
        
        # 3. Visual Projection
        visual_embeds = self.visual_projection(query_output)
        num_visual_tokens = visual_embeds.shape[1]
        
        # 4. 初始化输入
        if prompt_ids is not None:
            input_ids = prompt_ids
        else:
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=device
            )
        
        # 5. 获取初始文本嵌入
        text_embeds = self.llm.embed_tokens(input_ids)
        
        # 6. 拼接视觉和文本嵌入
        inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)
        
        # 7. 准备注意力掩码
        visual_attention_mask = torch.ones(batch_size, num_visual_tokens, device=device, dtype=torch.long)
        text_attention_mask = torch.ones(batch_size, input_ids.shape[1], device=device, dtype=torch.long)
        attention_mask = torch.cat([visual_attention_mask, text_attention_mask], dim=1)
        
        # 8. 自回归生成
        generated_ids = input_ids
        past_key_values = None
        
        for step in range(max_length - input_ids.shape[1]):
            seq_len = inputs_embeds.shape[1]
            
            # 计算位置 ID
            if past_key_values is None:
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
                past_len = 0
            else:
                past_len = past_key_values[0][0].shape[2]
                position_ids = torch.tensor([[past_len]], device=device).expand(batch_size, -1)
            
            # 准备 RoPE
            max_pos = position_ids.max().item() + 1
            self.llm.rotary_emb._extend_cache(int(max_pos))
            cos = self.llm.rotary_emb.cos_cached[:, :, :int(max_pos), :]
            sin = self.llm.rotary_emb.sin_cached[:, :, :int(max_pos), :]
            
            pos_flat = position_ids.reshape(-1)
            cos_selected = cos[0, 0, pos_flat, :].reshape(batch_size, position_ids.shape[1], -1)
            sin_selected = sin[0, 0, pos_flat, :].reshape(batch_size, position_ids.shape[1], -1)
            position_embeddings = (cos_selected.unsqueeze(1), sin_selected.unsqueeze(1))
            
            # 创建因果掩码
            current_seq_len = inputs_embeds.shape[1]
            total_len = current_seq_len + past_len
            causal_mask = self.llm._create_causal_mask(
                current_seq_len, past_len, device, inputs_embeds.dtype
            )
            
            # 合并 attention mask
            if past_key_values is None:
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(inputs_embeds.dtype)
            else:
                expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(inputs_embeds.dtype)
            
            inverted_mask = 1.0 - expanded_mask
            expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), float('-inf'))
            combined_mask = torch.minimum(causal_mask, expanded_mask)
            
            # 通过 LLM 层
            hidden_states = inputs_embeds
            new_past_key_values = []
            
            for i, layer in enumerate(self.llm.layers):
                past_kv = past_key_values[i] if past_key_values is not None else None
                hidden_states, present_kv = layer(
                    hidden_states=hidden_states,
                    attention_mask=combined_mask,
                    position_embeddings=position_embeddings,
                    past_key_value=past_kv,
                    use_cache=True
                )
                new_past_key_values.append(present_kv)
            
            past_key_values = new_past_key_values
            
            # 最终归一化
            hidden_states = self.llm.norm(hidden_states)
            
            # 获取最后一个 token 的 logits
            next_token_logits = self.llm.lm_head(hidden_states[:, -1, :])
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪解码
            if do_sample:
                filtered_logits = next_token_logits.clone()
                
                # Top-K
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    top_k_values, _ = torch.topk(filtered_logits, k)
                    threshold = top_k_values[..., -1, None]
                    indices_to_remove = filtered_logits < threshold
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Top-P
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # 采样
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
            
            # 更新生成的 IDs
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # 更新 inputs_embeds（只需要新 token）
            inputs_embeds = self.llm.embed_tokens(next_tokens.unsqueeze(-1))
            
            # 更新 attention_mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)
            
            # 检查 EOS
            if (next_tokens == eos_token_id).all():
                break
        
        return generated_ids
    
    def extra_repr(self) -> str:
        """返回模块的额外表示信息"""
        return (
            f"num_query_tokens={self.config.num_query_tokens}, "
            f"projection_dim={self.config.projection_dim}"
        )
