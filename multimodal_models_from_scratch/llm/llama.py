"""
LLaMA (Large Language Model Meta AI) 模型实现

LLaMA 是 Meta 开源的大语言模型，采用了多项现代架构改进：
- Pre-Norm 架构：在注意力和 FFN 之前应用归一化
- RMSNorm：替代 LayerNorm，计算效率更高
- RoPE：旋转位置编码，支持外推到更长序列
- SwiGLU：门控激活函数，性能优于 GELU
- GQA：分组查询注意力，减少 KV Cache 内存占用

需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 17.5
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from multimodal_models_from_scratch.config import LLaMAConfig
from multimodal_models_from_scratch.llm.rmsnorm import RMSNorm
from multimodal_models_from_scratch.llm.rope import RotaryPositionEmbedding
from multimodal_models_from_scratch.llm.swiglu import SwiGLU
from multimodal_models_from_scratch.llm.gqa import GroupedQueryAttention


class LLaMADecoderLayer(nn.Module):
    """
    LLaMA Decoder 层
    
    采用 Pre-Norm 架构：
    1. RMSNorm -> GQA + Residual
    2. RMSNorm -> SwiGLU + Residual
    
    Args:
        config: LLaMA 模型配置
        layer_idx: 层索引（用于调试和日志）
    """
    
    def __init__(self, config: LLaMAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        
        # Pre-Norm: 注意力前的 RMSNorm
        self.input_layernorm = RMSNorm(config.d_model)
        
        # Grouped Query Attention
        self.self_attn = GroupedQueryAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_rate=config.dropout_rate
        )
        
        # Pre-Norm: FFN 前的 RMSNorm
        self.post_attention_layernorm = RMSNorm(config.d_model)
        
        # SwiGLU FFN
        self.mlp = SwiGLU(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout_rate=config.dropout_rate
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: 输入张量，形状为 (batch, seq_len, d_model)
            attention_mask: 注意力掩码，形状为 (batch, 1, seq_len, seq_len) 或 (batch, 1, 1, seq_len)
                           值为 0 表示允许注意力，值为 -inf 表示屏蔽
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: KV Cache，元组 (past_key, past_value)
            use_cache: 是否返回更新后的 KV Cache
        
        Returns:
            hidden_states: 输出张量，形状为 (batch, seq_len, d_model)
            past_key_value: 如果 use_cache=True，返回更新后的 KV Cache
        """
        residual = hidden_states
        
        # Pre-Norm + Self Attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        # Pre-Norm + FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value
    
    def extra_repr(self) -> str:
        return f'layer_idx={self.layer_idx}, d_model={self.d_model}'


class LLaMAModel(nn.Module):
    """
    LLaMA 模型
    
    组件:
    - Token Embedding: 将 token ID 映射到嵌入向量
    - RoPE: 旋转位置编码
    - N x LLaMADecoderLayer: 堆叠的 Decoder 层
    - RMSNorm: 最终的归一化层
    - LM Head: 语言模型头（可选权重绑定）
    
    Args:
        config: LLaMA 模型配置
    """
    
    def __init__(self, config: LLaMAConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        
        # Token Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # RoPE 位置编码
        head_dim = config.d_model // config.num_heads
        self.rotary_emb = RotaryPositionEmbedding(
            d_model=head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        # Decoder Layers
        self.layers = nn.ModuleList([
            LLaMADecoderLayer(config, layer_idx=i)
            for i in range(config.num_layers)
        ])
        
        # Final RMSNorm
        self.norm = RMSNorm(config.d_model)
        
        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重绑定
        if config.tie_weights:
            self.lm_head.weight = self.embed_tokens.weight
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 使用较小的标准差初始化嵌入层
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        
        # 如果没有权重绑定，初始化 LM Head
        if not self.config.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """返回输入嵌入层"""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """设置输入嵌入层"""
        self.embed_tokens = value
    
    def _create_causal_mask(
        self,
        seq_len: int,
        past_key_values_length: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建因果注意力掩码
        
        Args:
            seq_len: 当前序列长度
            past_key_values_length: KV Cache 中的历史长度
            device: 设备
            dtype: 数据类型
        
        Returns:
            mask: 因果掩码，形状为 (1, 1, seq_len, total_len)
                  允许注意力的位置为 0，屏蔽的位置为 -inf
        """
        total_len = seq_len + past_key_values_length
        
        # 创建因果掩码：下三角为 True（允许注意力）
        # 对于 KV Cache 场景，需要考虑历史 token
        mask = torch.ones(seq_len, total_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=past_key_values_length)
        
        # 转换为注意力掩码格式：True -> 0, False -> -inf
        causal_mask = torch.zeros(seq_len, total_len, device=device, dtype=dtype)
        causal_mask.masked_fill_(~mask, float('-inf'))
        
        # 扩展维度: (1, 1, seq_len, total_len)
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
        """
        准备注意力掩码
        
        Args:
            attention_mask: 用户提供的注意力掩码，形状为 (batch, total_seq_len)
                           1 表示有效 token，0 表示 padding
                           total_seq_len 可能包含历史 token
            batch_size: 批次大小
            seq_len: 当前输入序列长度
            past_key_values_length: KV Cache 中的历史长度
            device: 设备
            dtype: 数据类型
        
        Returns:
            combined_mask: 组合后的掩码，形状为 (batch, 1, seq_len, total_len)
        """
        total_len = seq_len + past_key_values_length
        
        # 创建因果掩码
        causal_mask = self._create_causal_mask(
            seq_len, past_key_values_length, device, dtype
        )
        
        if attention_mask is None:
            # 如果没有提供 attention_mask，只使用因果掩码
            return causal_mask.expand(batch_size, -1, -1, -1)
        
        # 处理用户提供的 attention_mask
        # attention_mask 可能是 (batch, current_seq_len) 或 (batch, total_seq_len)
        mask_len = attention_mask.shape[1]
        
        if mask_len < total_len:
            # 如果 attention_mask 只包含当前 token，在前面添加历史 token 的掩码
            past_mask = torch.ones(batch_size, total_len - mask_len, device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([past_mask, attention_mask], dim=1)
        elif mask_len > total_len:
            # 如果 attention_mask 比需要的长，截取最后 total_len 个
            attention_mask = attention_mask[:, -total_len:]
        
        # 转换为注意力掩码格式
        # (batch, total_len) -> (batch, 1, 1, total_len)
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        expanded_mask = expanded_mask.to(dtype)
        
        # 将 0 转换为 -inf，1 保持为 0
        # 使用 masked_fill 避免 0 * -inf = nan 的问题
        inverted_mask = 1.0 - expanded_mask
        expanded_mask = inverted_mask.masked_fill(inverted_mask.bool(), float('-inf'))
        
        # 组合因果掩码和 padding 掩码
        # 使用 torch.minimum 而不是加法，避免 -inf + -inf = nan
        combined_mask = torch.minimum(causal_mask, expanded_mask)
        
        return combined_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            attention_mask: 注意力掩码，形状为 (batch, seq_len)
                           1 表示有效 token，0 表示 padding
            position_ids: 位置 ID，形状为 (batch, seq_len)
                         如果为 None，则自动生成
            past_key_values: KV Cache，列表长度为 num_layers
                            每个元素为 (past_key, past_value) 元组
            use_cache: 是否返回更新后的 KV Cache
            output_hidden_states: 是否返回所有层的隐藏状态
        
        Returns:
            {
                'logits': (batch, seq_len, vocab_size),
                'hidden_states': (batch, seq_len, d_model),
                'past_key_values': List[Tuple] if use_cache,
                'all_hidden_states': Tuple if output_hidden_states
            }
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
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
        # 需要根据 position_ids 获取对应的 cos 和 sin
        max_pos = position_ids.max().item() + 1
        self.rotary_emb._extend_cache(int(max_pos))
        
        # 获取 cos 和 sin
        cos = self.rotary_emb.cos_cached[:, :, :int(max_pos), :]
        sin = self.rotary_emb.sin_cached[:, :, :int(max_pos), :]
        
        # 根据 position_ids 选择对应的位置编码
        # position_ids: (batch, seq_len)
        # cos/sin: (1, 1, max_pos, head_dim)
        pos_flat = position_ids.reshape(-1)  # (batch * seq_len,)
        cos_selected = cos[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        sin_selected = sin[0, 0, pos_flat, :].reshape(batch_size, seq_len, -1)
        
        # 调整形状以匹配 GQA 的期望: (batch, 1, seq_len, head_dim)
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
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # 最终归一化
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
        # 构建输出字典
        output = {
            'logits': logits,
            'hidden_states': hidden_states,
        }
        
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
        **kwargs
    ) -> Dict[str, Any]:
        """
        准备生成所需的输入
        
        在自回归生成过程中，如果有 KV Cache，只需要处理最后一个 token。
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            past_key_values: KV Cache
            attention_mask: 注意力掩码
            **kwargs: 其他参数
        
        Returns:
            准备好的输入字典
        """
        # 如果有 KV Cache，只需要最后一个 token
        if past_key_values is not None and len(past_key_values) > 0:
            # past_key_values[0][0] 形状: (batch, num_kv_heads, past_len, head_dim)
            past_length = past_key_values[0][0].shape[2]
            
            # 只保留最后一个 token
            input_ids = input_ids[:, -1:]
            
            # 计算位置 ID
            position_ids = torch.tensor(
                [[past_length]],
                device=input_ids.device
            ).expand(input_ids.shape[0], -1)
        else:
            past_length = 0
            position_ids = None
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': True,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成
        
        Args:
            input_ids: 输入 token ID，形状为 (batch, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，控制生成随机性
            top_k: Top-K 采样的 K 值
            top_p: Top-P (nucleus) 采样的 P 值
            do_sample: 是否使用采样（False 则使用贪婪解码）
            eos_token_id: 结束 token ID
            pad_token_id: padding token ID
        
        Returns:
            generated_ids: 生成的 token ID，形状为 (batch, seq_len + num_generated)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # 初始化
        generated_ids = input_ids
        past_key_values = None
        attention_mask = torch.ones_like(input_ids)
        
        # 记录每个序列是否已结束
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            # 准备输入
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            
            # 前向传播
            outputs = self.forward(**model_inputs)
            
            # 获取最后一个 token 的 logits
            next_token_logits = outputs['logits'][:, -1, :]
            
            # 更新 KV Cache
            past_key_values = outputs.get('past_key_values')
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 采样或贪婪解码
            if do_sample:
                # 复制 logits 以避免修改原始数据
                filtered_logits = next_token_logits.clone()
                
                # Top-K 采样
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    top_k_values, _ = torch.topk(filtered_logits, k)
                    threshold = top_k_values[..., -1, None]
                    indices_to_remove = filtered_logits < threshold
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Top-P (nucleus) 采样
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过 top_p 的 token（但保留至少一个）
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的 token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    # 将排序后的掩码映射回原始索引
                    indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # 计算概率分布
                # 使用 softmax 前先减去最大值以提高数值稳定性
                max_logits = filtered_logits.max(dim=-1, keepdim=True)[0]
                # 将 -inf 替换为一个很小的值以避免 nan
                safe_logits = torch.where(
                    filtered_logits == float('-inf'),
                    torch.full_like(filtered_logits, -1e10),
                    filtered_logits - max_logits
                )
                probs = torch.softmax(safe_logits, dim=-1)
                
                # 从概率分布中采样
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                # 贪婪解码
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
    
    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.vocab_size}, d_model={self.d_model}, '
            f'num_layers={self.num_layers}'
        )
