"""
Qwen 模型单元测试

测试 NTK-aware RoPE 插值和 Sliding Window Attention

需求: 10.7
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.config import QwenConfig
from multimodal_models_from_scratch.llm.qwen import (
    QwenAttention,
    QwenDecoderLayer,
    QwenModel
)


class TestQwenAttention:
    """QwenAttention 测试"""
    
    def test_basic_forward(self):
        """测试基本前向传播"""
        attn = QwenAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            use_sliding_window=False
        )
        x = torch.randn(2, 10, 256)
        out, _ = attn(x)
        assert out.shape == (2, 10, 256)
    
    def test_sliding_window_attention(self):
        """测试滑动窗口注意力"""
        attn = QwenAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            use_sliding_window=True,
            sliding_window_size=4
        )
        x = torch.randn(2, 10, 256)
        out, _ = attn(x)
        assert out.shape == (2, 10, 256)
    
    def test_sliding_window_mask_shape(self):
        """测试滑动窗口掩码形状"""
        attn = QwenAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            use_sliding_window=True,
            sliding_window_size=4
        )
        mask = attn._create_sliding_window_mask(
            seq_len=10, kv_seq_len=10,
            device=torch.device('cpu'), dtype=torch.float32
        )
        assert mask.shape == (1, 1, 10, 10)

    def test_sliding_window_mask_values(self):
        """测试滑动窗口掩码值"""
        attn = QwenAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            use_sliding_window=True,
            sliding_window_size=3
        )
        mask = attn._create_sliding_window_mask(
            seq_len=5, kv_seq_len=5,
            device=torch.device('cpu'), dtype=torch.float32
        )
        # 检查因果性：对角线及以下应该有有效值
        # 检查窗口限制：超出窗口的位置应该是 -inf
        mask_2d = mask[0, 0]
        
        # 位置 (0, 0) 应该是 0（允许注意力）
        assert mask_2d[0, 0] == 0
        
        # 位置 (4, 0) 应该是 -inf（超出窗口）
        assert mask_2d[4, 0] == float('-inf')
        
        # 位置 (4, 2) 应该是 0（在窗口内）
        assert mask_2d[4, 2] == 0
    
    def test_kv_cache(self):
        """测试 KV Cache"""
        attn = QwenAttention(
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            use_sliding_window=False
        )
        x = torch.randn(2, 10, 256)
        out, kv = attn(x, use_cache=True)
        
        assert kv is not None
        assert len(kv) == 2  # (key, value)
        assert kv[0].shape == (2, 4, 10, 32)  # (batch, num_kv_heads, seq_len, head_dim)


class TestQwenDecoderLayer:
    """QwenDecoderLayer 测试"""
    
    def test_forward(self):
        """测试前向传播"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            use_sliding_window=True,
            sliding_window_size=128
        )
        layer = QwenDecoderLayer(config, layer_idx=0)
        x = torch.randn(2, 10, 256)
        out, _ = layer(x)
        assert out.shape == (2, 10, 256)
    
    def test_with_kv_cache(self):
        """测试带 KV Cache 的前向传播"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512
        )
        layer = QwenDecoderLayer(config, layer_idx=0)
        x = torch.randn(2, 10, 256)
        out, kv = layer(x, use_cache=True)
        
        assert out.shape == (2, 10, 256)
        assert kv is not None


class TestQwenModel:
    """QwenModel 测试"""
    
    @pytest.fixture
    def config(self):
        """创建测试配置"""
        return QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            use_sliding_window=True,
            sliding_window_size=128
        )
    
    @pytest.fixture
    def model(self, config):
        """创建测试模型"""
        return QwenModel(config)
    
    def test_forward(self, model):
        """测试前向传播"""
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'hidden_states' in outputs
        assert outputs['logits'].shape == (2, 10, 1000)
        assert outputs['hidden_states'].shape == (2, 10, 256)
    
    def test_kv_cache(self, model):
        """测试 KV Cache"""
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids, use_cache=True)
        
        assert 'past_key_values' in outputs
        past_kv = outputs['past_key_values']
        assert len(past_kv) == 2  # num_layers
        assert past_kv[0][0].shape == (2, 4, 10, 32)

    def test_incremental_generation(self, model):
        """测试增量生成（使用 KV Cache）"""
        # 第一步：处理完整序列
        input_ids = torch.randint(0, 1000, (1, 5))
        outputs = model(input_ids, use_cache=True)
        past_kv = outputs['past_key_values']
        
        # 第二步：只处理新 token
        new_token = torch.randint(0, 1000, (1, 1))
        outputs2 = model(new_token, past_key_values=past_kv, use_cache=True)
        
        assert outputs2['logits'].shape == (1, 1, 1000)
        assert outputs2['past_key_values'][0][0].shape[2] == 6  # 5 + 1
    
    def test_generate(self, model):
        """测试文本生成"""
        input_ids = torch.randint(0, 1000, (1, 5))
        generated = model.generate(input_ids, max_new_tokens=10)
        
        assert generated.shape == (1, 15)  # 5 + 10
    
    def test_generate_with_sampling(self, model):
        """测试带采样的文本生成"""
        input_ids = torch.randint(0, 1000, (1, 5))
        generated = model.generate(
            input_ids,
            max_new_tokens=5,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9
        )
        
        assert generated.shape == (1, 10)
    
    def test_output_hidden_states(self, model):
        """测试输出所有隐藏状态"""
        input_ids = torch.randint(0, 1000, (2, 10))
        outputs = model(input_ids, output_hidden_states=True)
        
        assert 'all_hidden_states' in outputs
        # num_layers + 1 (final norm output)
        assert len(outputs['all_hidden_states']) == 3


class TestNTKAwareRoPE:
    """NTK-aware RoPE 插值测试"""
    
    def test_ntk_alpha_default(self):
        """测试默认 NTK alpha"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512
        )
        model = QwenModel(config)
        assert model.ntk_alpha == 1.0
    
    def test_ntk_alpha_with_scaling(self):
        """测试带缩放的 NTK alpha"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            rope_scaling={'type': 'ntk', 'factor': 2.0}
        )
        model = QwenModel(config)
        assert model.ntk_alpha == 2.0
    
    def test_ntk_rope_computation(self):
        """测试 NTK RoPE 计算"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            rope_scaling={'type': 'ntk', 'factor': 2.0}
        )
        model = QwenModel(config)
        
        # 计算 NTK RoPE
        cos, sin = model._compute_ntk_rope(100, torch.device('cpu'))
        
        assert cos.shape == (1, 1, 100, 32)  # head_dim = 256 // 8 = 32
        assert sin.shape == (1, 1, 100, 32)
    
    def test_long_sequence_with_ntk(self):
        """测试长序列的 NTK 插值"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,  # 较短的最大长度
            rope_scaling={'type': 'ntk', 'factor': 2.0}
        )
        model = QwenModel(config)
        
        # 使用超过 max_seq_len 的序列
        input_ids = torch.randint(0, 1000, (1, 100))
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (1, 100, 1000)


class TestSlidingWindowAttention:
    """Sliding Window Attention 测试"""
    
    def test_sliding_window_enabled(self):
        """测试启用滑动窗口"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            use_sliding_window=True,
            sliding_window_size=64
        )
        model = QwenModel(config)
        
        # 检查层是否使用滑动窗口
        assert model.layers[0].self_attn.use_sliding_window is True
        assert model.layers[0].self_attn.sliding_window_size == 64
    
    def test_sliding_window_disabled(self):
        """测试禁用滑动窗口"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            use_sliding_window=False
        )
        model = QwenModel(config)
        
        assert model.layers[0].self_attn.use_sliding_window is False
    
    def test_long_sequence_with_sliding_window(self):
        """测试长序列的滑动窗口注意力"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            use_sliding_window=True,
            sliding_window_size=32
        )
        model = QwenModel(config)
        
        # 使用较长的序列
        input_ids = torch.randint(0, 1000, (1, 100))
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (1, 100, 1000)


class TestWeightTying:
    """权重绑定测试"""
    
    def test_weight_tying_enabled(self):
        """测试启用权重绑定"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            tie_weights=True
        )
        model = QwenModel(config)
        
        # 检查权重是否绑定
        assert model.lm_head.weight is model.embed_tokens.weight
    
    def test_weight_tying_disabled(self):
        """测试禁用权重绑定"""
        config = QwenConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=512,
            tie_weights=False
        )
        model = QwenModel(config)
        
        # 检查权重是否独立
        assert model.lm_head.weight is not model.embed_tokens.weight
