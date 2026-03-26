"""
Qwen (通义千问) 模型实现

Qwen 是阿里巴巴开源的大语言模型，继承了 LLaMA 的核心架构，
并添加了以下特性：
- NTK-aware RoPE 插值：支持动态扩展上下文长度
- Sliding Window Attention（可选）：高效处理长序列

需求: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9
"""

import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from multimodal_models_from_scratch.config import QwenConfig
from multimodal_models_from_scratch.llm.rmsnorm import RMSNorm
from multimodal_models_from_scratch.llm.rope import RotaryPositionEmbedding
from multimodal_models_from_scratch.llm.swiglu import SwiGLU
from multimodal_models_from_scratch.llm.gqa import GroupedQueryAttention
from multimodal_models_from_scratch.llm.llama import LLaMADecoderLayer, LLaMAModel


class QwenAttention(GroupedQueryAttention):
    """
    Qwen 注意力模块，支持 Sliding Window Attention
    
    继承自 GroupedQueryAttention，添加滑动窗口注意力支持。
    滑动窗口注意力限制每个 token 只能关注其前后一定范围内的 token，
    从而降低长序列的计算复杂度。
    
    Args:
        d_model: 模型维度
        num_heads: Query 头数
        num_kv_heads: Key/Value 头数
        dropout_rate: Dropout 比率
        use_sliding_window: 是否使用滑动窗口注意力
        sliding_window_size: 滑动窗口大小
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout_rate: float = 0.0,
        use_sliding_window: bool = False,
        sliding_window_size: int = 4096
    ):
        super().__init__(d_model, num_heads, num_kv_heads, dropout_rate)
        self.use_sliding_window = use_sliding_window
        self.sliding_window_size = sliding_window_size

    def _create_sliding_window_mask(
        self,
        seq_len: int,
        kv_seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        创建滑动窗口注意力掩码
        
        Args:
            seq_len: 当前序列长度
            kv_seq_len: KV 序列长度（包含历史 token）
            device: 设备
            dtype: 数据类型
        
        Returns:
            mask: 滑动窗口掩码，形状为 (1, 1, seq_len, kv_seq_len)
                  允许注意力的位置为 0，屏蔽的位置为 -inf
        """
        # 创建位置索引
        # query_positions: [kv_seq_len - seq_len, ..., kv_seq_len - 1]
        query_positions = torch.arange(
            kv_seq_len - seq_len, kv_seq_len, device=device
        ).unsqueeze(1)  # (seq_len, 1)
        
        # key_positions: [0, 1, ..., kv_seq_len - 1]
        key_positions = torch.arange(kv_seq_len, device=device).unsqueeze(0)  # (1, kv_seq_len)
        
        # 计算位置差
        position_diff = query_positions - key_positions  # (seq_len, kv_seq_len)
        
        # 滑动窗口条件：
        # 1. 因果性：key_position <= query_position (position_diff >= 0)
        # 2. 窗口限制：query_position - key_position < window_size
        causal_mask = position_diff >= 0
        window_mask = position_diff < self.sliding_window_size
        
        # 组合掩码
        combined_mask = causal_mask & window_mask
        
        # 转换为注意力掩码格式
        mask = torch.zeros(seq_len, kv_seq_len, device=device, dtype=dtype)
        mask.masked_fill_(~combined_mask, float('-inf'))
        
        # 扩展维度: (1, 1, seq_len, kv_seq_len)
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播，支持滑动窗口注意力
        
        Args:
            hidden_states: 输入张量，形状为 (batch, seq_len, d_model)
            attention_mask: 注意力掩码
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: KV Cache
            use_cache: 是否返回更新后的 KV Cache
        
        Returns:
            output: 输出张量
            past_key_value: 更新后的 KV Cache（如果 use_cache=True）
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 计算 Q, K, V
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # 重塑为多头格式
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE 位置编码
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query = self._apply_rotary_emb(query, cos, sin)
            key = self._apply_rotary_emb(key, cos, sin)
        
        # 处理 KV Cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        kv_seq_len = key.shape[2]
        
        # 保存当前 KV 用于缓存
        current_key_value = (key, value) if use_cache else None
        
        # 将 KV 头重复以匹配 Query 头数
        key_expanded = self._repeat_kv(key)
        value_expanded = self._repeat_kv(value)
        
        # 计算注意力分数
        attn_weights = torch.matmul(query, key_expanded.transpose(-2, -1)) * self.scale
        
        # 应用滑动窗口掩码
        if self.use_sliding_window:
            sliding_mask = self._create_sliding_window_mask(
                seq_len, kv_seq_len, hidden_states.device, hidden_states.dtype
            )
            attn_weights = attn_weights + sliding_mask
        
        # 应用用户提供的注意力掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax 归一化
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, value_expanded)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # 输出投影
        output = self.o_proj(attn_output)
        
        return output, current_key_value
    
    def extra_repr(self) -> str:
        base_repr = super().extra_repr()
        return (
            f'{base_repr}, use_sliding_window={self.use_sliding_window}, '
            f'sliding_window_size={self.sliding_window_size}'
        )


class QwenDecoderLayer(nn.Module):
    """
    Qwen Decoder 层
    
    继承 LLaMA 的 Pre-Norm 架构，添加滑动窗口注意力支持。
    
    结构:
    1. RMSNorm -> QwenAttention + Residual
    2. RMSNorm -> SwiGLU + Residual
    
    Args:
        config: Qwen 模型配置
        layer_idx: 层索引
    """
    
    def __init__(self, config: QwenConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = config.d_model
        
        # Pre-Norm: 注意力前的 RMSNorm
        self.input_layernorm = RMSNorm(config.d_model)
        
        # Qwen Attention（支持滑动窗口）
        self.self_attn = QwenAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            dropout_rate=config.dropout_rate,
            use_sliding_window=config.use_sliding_window,
            sliding_window_size=config.sliding_window_size
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
            attention_mask: 注意力掩码
            position_embeddings: RoPE 位置编码 (cos, sin) 元组
            past_key_value: KV Cache
            use_cache: 是否返回更新后的 KV Cache
        
        Returns:
            hidden_states: 输出张量
            past_key_value: 更新后的 KV Cache（如果 use_cache=True）
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


class QwenModel(nn.Module):
    """
    Qwen 模型
    
    继承 LLaMA 的核心架构，添加以下特性：
    - NTK-aware RoPE 插值：动态扩展上下文长度
    - Sliding Window Attention（可选）：高效处理长序列
    
    组件:
    - Token Embedding: 将 token ID 映射到嵌入向量
    - RoPE: 旋转位置编码（支持 NTK-aware 插值）
    - N x QwenDecoderLayer: 堆叠的 Decoder 层
    - RMSNorm: 最终的归一化层
    - LM Head: 语言模型头（可选权重绑定）
    
    Args:
        config: Qwen 模型配置
    """
    
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.d_model = config.d_model
        self.num_layers = config.num_layers
        self.max_seq_len = config.max_seq_len
        
        # NTK-aware 插值配置
        self.rope_scaling = config.rope_scaling
        self.ntk_alpha = self._get_ntk_alpha()
        
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
            QwenDecoderLayer(config, layer_idx=i)
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
    
    def _get_ntk_alpha(self) -> float:
        """
        获取 NTK-aware 插值的 alpha 参数
        
        Returns:
            alpha: NTK 缩放因子
        """
        if self.rope_scaling is None:
            return 1.0
        
        scaling_type = self.rope_scaling.get('type', 'ntk')
        if scaling_type == 'ntk':
            return self.rope_scaling.get('factor', 1.0)
        return 1.0
    
    def _init_weights(self):
        """初始化模型权重"""
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        if not self.config.tie_weights:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
    
    def get_input_embeddings(self) -> nn.Embedding:
        """返回输入嵌入层"""
        return self.embed_tokens
    
    def set_input_embeddings(self, value: nn.Embedding):
        """设置输入嵌入层"""
        self.embed_tokens = value

    def _compute_ntk_rope(
        self,
        seq_len: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算 NTK-aware RoPE 位置编码
        
        当序列长度超过训练时的最大长度时，使用 NTK-aware 插值
        动态调整 RoPE 的基础频率，以支持更长的上下文。
        
        Args:
            seq_len: 当前序列长度
            device: 设备
        
        Returns:
            cos: cos 缓存，形状为 (1, 1, seq_len, head_dim)
            sin: sin 缓存，形状为 (1, 1, seq_len, head_dim)
        """
        # 计算动态 alpha
        if seq_len > self.max_seq_len and self.ntk_alpha > 1.0:
            # 动态 NTK 缩放：根据序列长度自动调整 alpha
            alpha = max(self.ntk_alpha, seq_len / self.max_seq_len)
        elif self.ntk_alpha > 1.0:
            alpha = self.ntk_alpha
        else:
            alpha = 1.0
        
        # 使用 RoPE 的 NTK-aware 插值方法
        cos, sin = self.rotary_emb.apply_ntk_scaling(seq_len, alpha)
        
        return cos.to(device), sin.to(device)
    
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
        """
        total_len = seq_len + past_key_values_length
        
        mask = torch.ones(seq_len, total_len, device=device, dtype=torch.bool)
        mask = torch.tril(mask, diagonal=past_key_values_length)
        
        causal_mask = torch.zeros(seq_len, total_len, device=device, dtype=dtype)
        causal_mask.masked_fill_(~mask, float('-inf'))
        
        return causal_mask.unsqueeze(0).unsqueeze(0)
    
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
            attention_mask: 用户提供的注意力掩码
            batch_size: 批次大小
            seq_len: 当前输入序列长度
            past_key_values_length: KV Cache 中的历史长度
            device: 设备
            dtype: 数据类型
        
        Returns:
            combined_mask: 组合后的掩码
        """
        total_len = seq_len + past_key_values_length
        
        causal_mask = self._create_causal_mask(
            seq_len, past_key_values_length, device, dtype
        )
        
        if attention_mask is None:
            return causal_mask.expand(batch_size, -1, -1, -1)
        
        mask_len = attention_mask.shape[1]
        
        if mask_len < total_len:
            past_mask = torch.ones(
                batch_size, total_len - mask_len, device=device, dtype=attention_mask.dtype
            )
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
            position_ids: 位置 ID，形状为 (batch, seq_len)
            past_key_values: KV Cache
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
        
        # 计算 NTK-aware RoPE 位置编码
        total_seq_len = past_key_values_length + seq_len
        cos, sin = self._compute_ntk_rope(total_seq_len, device)
        
        # 根据 position_ids 选择对应的位置编码
        max_pos = position_ids.max().item() + 1
        if cos.shape[2] < max_pos:
            cos, sin = self._compute_ntk_rope(int(max_pos), device)
        
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
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # LM Head
        logits = self.lm_head(hidden_states)
        
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
        
        Args:
            input_ids: 输入 token ID
            past_key_values: KV Cache
            attention_mask: 注意力掩码
            **kwargs: 其他参数
        
        Returns:
            准备好的输入字典
        """
        if past_key_values is not None and len(past_key_values) > 0:
            past_length = past_key_values[0][0].shape[2]
            input_ids = input_ids[:, -1:]
            position_ids = torch.tensor(
                [[past_length]],
                device=input_ids.device
            ).expand(input_ids.shape[0], -1)
        else:
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
            input_ids: 输入 token ID
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
        
        generated_ids = input_ids
        past_key_values = None
        attention_mask = torch.ones_like(input_ids)
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_new_tokens):
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
            )
            
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs['logits'][:, -1, :]
            past_key_values = outputs.get('past_key_values')
            
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            if do_sample:
                filtered_logits = next_token_logits.clone()
                
                if top_k is not None and top_k > 0:
                    k = min(top_k, filtered_logits.size(-1))
                    top_k_values, _ = torch.topk(filtered_logits, k)
                    threshold = top_k_values[..., -1, None]
                    indices_to_remove = filtered_logits < threshold
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(filtered_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = torch.zeros_like(filtered_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                    filtered_logits = filtered_logits.masked_fill(indices_to_remove, float('-inf'))
                
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
            
            if pad_token_id is not None:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            
            generated_ids = torch.cat([generated_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)
            
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (next_tokens != eos_token_id).long()
                if unfinished_sequences.max() == 0:
                    break
        
        return generated_ids
    
    def extra_repr(self) -> str:
        return (
            f'vocab_size={self.vocab_size}, d_model={self.d_model}, '
            f'num_layers={self.num_layers}, ntk_alpha={self.ntk_alpha}'
        )
