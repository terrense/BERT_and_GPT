"""
LLaMA 模型单元测试

测试 LLaMA 模型的功能正确性，包括：
- 前向传播输出形状（logits, hidden_states）
- KV Cache 生成和增量解码
- prepare_inputs_for_generation 方法
- 不同 LLaMAConfig 配置
- Token Embedding 和 LM Head 权重绑定
- 批处理

需求: 7.8
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.config import LLaMAConfig
from multimodal_models_from_scratch.llm.llama import LLaMAModel, LLaMADecoderLayer


class TestLLaMADecoderLayer:
    """LLaMA Decoder Layer 测试类"""
    
    def test_output_shape_basic(self):
        """测试基本输出形状"""
        batch_size = 2
        seq_len = 16
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128
        )
        
        layer = LLaMADecoderLayer(config, layer_idx=0)
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)
        
        output, past_kv = layer(hidden_states)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
        assert past_kv is None  # use_cache=False by default

    def test_output_shape_with_cache(self):
        """测试带 KV Cache 的输出形状"""
        batch_size = 2
        seq_len = 16
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128
        )
        head_dim = config.d_model // config.num_heads
        
        layer = LLaMADecoderLayer(config, layer_idx=0)
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)
        
        output, past_kv = layer(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
        assert past_kv is not None
        assert len(past_kv) == 2  # (key, value)
        assert past_kv[0].shape == (batch_size, config.num_kv_heads, seq_len, head_dim)
        assert past_kv[1].shape == (batch_size, config.num_kv_heads, seq_len, head_dim)
    
    def test_with_attention_mask(self):
        """测试带注意力掩码的前向传播"""
        batch_size = 2
        seq_len = 16
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128
        )
        
        layer = LLaMADecoderLayer(config, layer_idx=0)
        hidden_states = torch.randn(batch_size, seq_len, config.d_model)
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        output, _ = layer(hidden_states, attention_mask=causal_mask)
        
        assert output.shape == (batch_size, seq_len, config.d_model)
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        config = LLaMAConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128
        )
        
        layer = LLaMADecoderLayer(config, layer_idx=5)
        repr_str = layer.extra_repr()
        
        assert 'layer_idx=5' in repr_str
        assert 'd_model=256' in repr_str


class TestLLaMAModel:
    """LLaMA 模型测试类"""
    
    @pytest.fixture
    def small_config(self):
        """创建小型配置用于测试"""
        return LLaMAConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128,
            dropout_rate=0.0,
            tie_weights=False
        )
    
    @pytest.fixture
    def model(self, small_config):
        """创建模型实例"""
        return LLaMAModel(small_config)
    
    def test_forward_output_shape(self, model, small_config):
        """测试前向传播输出形状"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids)
        
        assert 'logits' in outputs
        assert 'hidden_states' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, small_config.vocab_size)
        assert outputs['hidden_states'].shape == (batch_size, seq_len, small_config.d_model)
    
    def test_forward_with_attention_mask(self, model, small_config):
        """测试带注意力掩码的前向传播"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        # 模拟 padding：第二个样本后半部分为 padding
        attention_mask[1, seq_len//2:] = 0
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs['logits'].shape == (batch_size, seq_len, small_config.vocab_size)
    
    def test_forward_with_position_ids(self, model, small_config):
        """测试带位置 ID 的前向传播"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        outputs = model(input_ids, position_ids=position_ids)
        
        assert outputs['logits'].shape == (batch_size, seq_len, small_config.vocab_size)

    def test_kv_cache_generation(self, model, small_config):
        """测试 KV Cache 生成"""
        batch_size = 2
        seq_len = 16
        head_dim = small_config.d_model // small_config.num_heads
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids, use_cache=True)
        
        assert 'past_key_values' in outputs
        past_key_values = outputs['past_key_values']
        
        # 检查 KV Cache 结构
        assert len(past_key_values) == small_config.num_layers
        for layer_kv in past_key_values:
            assert len(layer_kv) == 2  # (key, value)
            assert layer_kv[0].shape == (batch_size, small_config.num_kv_heads, seq_len, head_dim)
            assert layer_kv[1].shape == (batch_size, small_config.num_kv_heads, seq_len, head_dim)
    
    def test_kv_cache_incremental_decoding(self, model, small_config):
        """测试 KV Cache 增量解码"""
        batch_size = 2
        initial_seq_len = 10
        head_dim = small_config.d_model // small_config.num_heads
        
        # 第一步：处理初始序列
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, initial_seq_len))
        outputs1 = model(input_ids, use_cache=True)
        past_key_values = outputs1['past_key_values']
        
        # 第二步：增量生成一个 token
        new_token = torch.randint(0, small_config.vocab_size, (batch_size, 1))
        outputs2 = model(new_token, past_key_values=past_key_values, use_cache=True)
        
        # 检查输出形状
        assert outputs2['logits'].shape == (batch_size, 1, small_config.vocab_size)
        
        # 检查更新后的 KV Cache
        new_past_key_values = outputs2['past_key_values']
        for layer_kv in new_past_key_values:
            assert layer_kv[0].shape == (batch_size, small_config.num_kv_heads, initial_seq_len + 1, head_dim)
            assert layer_kv[1].shape == (batch_size, small_config.num_kv_heads, initial_seq_len + 1, head_dim)
    
    def test_kv_cache_multiple_steps(self, model, small_config):
        """测试多步 KV Cache 增量解码"""
        batch_size = 2
        initial_seq_len = 5
        num_new_tokens = 3
        head_dim = small_config.d_model // small_config.num_heads
        
        # 初始序列
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, initial_seq_len))
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs['past_key_values']
        
        # 多步增量生成
        for step in range(num_new_tokens):
            new_token = torch.randint(0, small_config.vocab_size, (batch_size, 1))
            outputs = model(new_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs['past_key_values']
        
        # 检查最终 KV Cache 长度
        expected_len = initial_seq_len + num_new_tokens
        for layer_kv in past_key_values:
            assert layer_kv[0].shape[2] == expected_len
            assert layer_kv[1].shape[2] == expected_len

    def test_output_hidden_states(self, model, small_config):
        """测试输出所有隐藏状态"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids, output_hidden_states=True)
        
        assert 'all_hidden_states' in outputs
        all_hidden_states = outputs['all_hidden_states']
        
        # 应该有 num_layers + 1 个隐藏状态（包括最终归一化后的）
        assert len(all_hidden_states) == small_config.num_layers + 1
        for hidden_state in all_hidden_states:
            assert hidden_state.shape == (batch_size, seq_len, small_config.d_model)
    
    def test_prepare_inputs_for_generation_no_cache(self, model, small_config):
        """测试 prepare_inputs_for_generation 无 KV Cache"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len))
        
        prepared = model.prepare_inputs_for_generation(input_ids)
        
        assert 'input_ids' in prepared
        assert 'use_cache' in prepared
        assert prepared['use_cache'] is True
        assert prepared['input_ids'].shape == (batch_size, seq_len)
        assert prepared['position_ids'] is None
    
    def test_prepare_inputs_for_generation_with_cache(self, model, small_config):
        """测试 prepare_inputs_for_generation 有 KV Cache"""
        batch_size = 2
        initial_seq_len = 10
        head_dim = small_config.d_model // small_config.num_heads
        
        # 创建模拟的 past_key_values
        past_key_values = []
        for _ in range(small_config.num_layers):
            key = torch.randn(batch_size, small_config.num_kv_heads, initial_seq_len, head_dim)
            value = torch.randn(batch_size, small_config.num_kv_heads, initial_seq_len, head_dim)
            past_key_values.append((key, value))
        
        # 输入包含历史 + 新 token
        full_seq_len = initial_seq_len + 1
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, full_seq_len))
        
        prepared = model.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values
        )
        
        # 应该只保留最后一个 token
        assert prepared['input_ids'].shape == (batch_size, 1)
        assert prepared['position_ids'] is not None
        assert prepared['position_ids'].shape == (batch_size, 1)
        # 位置 ID 应该是 past_length
        assert (prepared['position_ids'] == initial_seq_len).all()


class TestLLaMAConfigurations:
    """测试不同 LLaMAConfig 配置"""
    
    def test_different_d_model(self):
        """测试不同 d_model 配置"""
        batch_size = 2
        seq_len = 8
        
        for d_model in [128, 256, 512]:
            config = LLaMAConfig(
                vocab_size=500,
                d_model=d_model,
                num_heads=8,
                num_kv_heads=2,
                num_layers=2,
                d_ff=d_model * 2,
                max_seq_len=64
            )
            model = LLaMAModel(config)
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs = model(input_ids)
            
            assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
            assert outputs['hidden_states'].shape == (batch_size, seq_len, d_model)
    
    def test_different_num_layers(self):
        """测试不同层数配置"""
        batch_size = 2
        seq_len = 8
        
        for num_layers in [1, 2, 4]:
            config = LLaMAConfig(
                vocab_size=500,
                d_model=256,
                num_heads=8,
                num_kv_heads=2,
                num_layers=num_layers,
                d_ff=512,
                max_seq_len=64
            )
            model = LLaMAModel(config)
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs = model(input_ids, output_hidden_states=True)
            
            # 隐藏状态数量应该是 num_layers + 1
            assert len(outputs['all_hidden_states']) == num_layers + 1
    
    def test_different_num_kv_heads(self):
        """测试不同 num_kv_heads 配置"""
        batch_size = 2
        seq_len = 8
        num_heads = 8
        d_model = 256
        head_dim = d_model // num_heads
        
        for num_kv_heads in [1, 2, 4, 8]:
            config = LLaMAConfig(
                vocab_size=500,
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                num_layers=2,
                d_ff=512,
                max_seq_len=64
            )
            model = LLaMAModel(config)
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs = model(input_ids, use_cache=True)
            
            # 检查 KV Cache 的 num_kv_heads 维度
            for layer_kv in outputs['past_key_values']:
                assert layer_kv[0].shape[1] == num_kv_heads
                assert layer_kv[1].shape[1] == num_kv_heads
    
    def test_different_vocab_size(self):
        """测试不同词表大小配置"""
        batch_size = 2
        seq_len = 8
        
        for vocab_size in [100, 500, 1000]:
            config = LLaMAConfig(
                vocab_size=vocab_size,
                d_model=256,
                num_heads=8,
                num_kv_heads=2,
                num_layers=2,
                d_ff=512,
                max_seq_len=64
            )
            model = LLaMAModel(config)
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
            outputs = model(input_ids)
            
            assert outputs['logits'].shape == (batch_size, seq_len, vocab_size)


class TestWeightTying:
    """测试权重绑定"""
    
    def test_weight_tying_enabled(self):
        """测试启用权重绑定"""
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            tie_weights=True
        )
        model = LLaMAModel(config)
        
        # 检查权重是否绑定
        assert model.lm_head.weight is model.embed_tokens.weight
    
    def test_weight_tying_disabled(self):
        """测试禁用权重绑定"""
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            tie_weights=False
        )
        model = LLaMAModel(config)
        
        # 检查权重是否独立
        assert model.lm_head.weight is not model.embed_tokens.weight
    
    def test_weight_tying_forward_pass(self):
        """测试权重绑定时的前向传播"""
        batch_size = 2
        seq_len = 8
        
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            tie_weights=True
        )
        model = LLaMAModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_weight_tying_parameter_count(self):
        """测试权重绑定对参数数量的影响"""
        config_tied = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            tie_weights=True
        )
        config_untied = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            tie_weights=False
        )
        
        model_tied = LLaMAModel(config_tied)
        model_untied = LLaMAModel(config_untied)
        
        params_tied = sum(p.numel() for p in model_tied.parameters())
        params_untied = sum(p.numel() for p in model_untied.parameters())
        
        # 绑定权重时参数更少
        embedding_params = config_tied.vocab_size * config_tied.d_model
        assert params_untied - params_tied == embedding_params


class TestBatchProcessing:
    """测试批处理"""
    
    def test_different_batch_sizes(self):
        """测试不同批次大小"""
        seq_len = 8
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        for batch_size in [1, 2, 4, 8]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs = model(input_ids)
            
            assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
            assert outputs['hidden_states'].shape == (batch_size, seq_len, config.d_model)
    
    def test_different_seq_lengths(self):
        """测试不同序列长度"""
        batch_size = 2
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=128
        )
        model = LLaMAModel(config)
        
        for seq_len in [1, 8, 16, 32, 64]:
            input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            outputs = model(input_ids)
            
            assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
            assert outputs['hidden_states'].shape == (batch_size, seq_len, config.d_model)
    
    def test_batch_with_padding_mask(self):
        """测试带 padding 掩码的批处理"""
        batch_size = 4
        max_seq_len = 16
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, max_seq_len))
        
        # 创建不同长度的 padding 掩码
        attention_mask = torch.ones(batch_size, max_seq_len)
        attention_mask[0, 12:] = 0  # 第一个样本长度 12
        attention_mask[1, 8:] = 0   # 第二个样本长度 8
        attention_mask[2, 14:] = 0  # 第三个样本长度 14
        # 第四个样本无 padding
        
        outputs = model(input_ids, attention_mask=attention_mask)
        
        assert outputs['logits'].shape == (batch_size, max_seq_len, config.vocab_size)
    
    def test_single_token_input(self):
        """测试单 token 输入"""
        batch_size = 2
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch_size, 1, config.vocab_size)
        assert outputs['hidden_states'].shape == (batch_size, 1, config.d_model)


class TestLLaMAIntegration:
    """LLaMA 集成测试"""
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        batch_size = 2
        seq_len = 8
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        outputs = model(input_ids)
        
        # 计算损失并反向传播
        loss = outputs['logits'].sum()
        loss.backward()
        
        # 检查梯度是否存在
        assert model.embed_tokens.weight.grad is not None
        assert model.lm_head.weight.grad is not None
        for layer in model.layers:
            assert layer.input_layernorm.weight.grad is not None
            assert layer.self_attn.q_proj.weight.grad is not None
    
    def test_eval_mode(self):
        """测试评估模式"""
        batch_size = 2
        seq_len = 8
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout_rate=0.1
        )
        model = LLaMAModel(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 评估模式下多次前向传播应该得到相同结果
        with torch.no_grad():
            outputs1 = model(input_ids)
            outputs2 = model(input_ids)
        
        assert torch.allclose(outputs1['logits'], outputs2['logits'])
    
    def test_get_input_embeddings(self):
        """测试 get_input_embeddings 方法"""
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        embeddings = model.get_input_embeddings()
        
        assert isinstance(embeddings, nn.Embedding)
        assert embeddings.num_embeddings == config.vocab_size
        assert embeddings.embedding_dim == config.d_model
    
    def test_set_input_embeddings(self):
        """测试 set_input_embeddings 方法"""
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        new_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        model.set_input_embeddings(new_embeddings)
        
        assert model.embed_tokens is new_embeddings
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        config = LLaMAConfig(
            vocab_size=500,
            d_model=256,
            num_heads=8,
            num_kv_heads=2,
            num_layers=2,
            d_ff=512,
            max_seq_len=64
        )
        model = LLaMAModel(config)
        
        repr_str = model.extra_repr()
        
        assert 'vocab_size=500' in repr_str
        assert 'd_model=256' in repr_str
        assert 'num_layers=2' in repr_str


class TestLLaMAGeneration:
    """LLaMA 生成测试"""
    
    @pytest.fixture
    def small_model(self):
        """创建小型模型用于生成测试"""
        config = LLaMAConfig(
            vocab_size=100,
            d_model=128,
            num_heads=4,
            num_kv_heads=2,
            num_layers=2,
            d_ff=256,
            max_seq_len=64
        )
        return LLaMAModel(config)
    
    def test_generate_greedy(self, small_model):
        """测试贪婪解码生成"""
        batch_size = 2
        prompt_len = 5
        max_new_tokens = 10
        
        input_ids = torch.randint(0, small_model.vocab_size, (batch_size, prompt_len))
        
        generated = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] == prompt_len + max_new_tokens
    
    def test_generate_with_sampling(self, small_model):
        """测试采样生成"""
        batch_size = 2
        prompt_len = 5
        max_new_tokens = 10
        
        input_ids = torch.randint(0, small_model.vocab_size, (batch_size, prompt_len))
        
        generated = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] == prompt_len + max_new_tokens
    
    def test_generate_with_top_k(self, small_model):
        """测试 Top-K 采样生成"""
        batch_size = 2
        prompt_len = 5
        max_new_tokens = 10
        
        input_ids = torch.randint(0, small_model.vocab_size, (batch_size, prompt_len))
        
        generated = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=10
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] == prompt_len + max_new_tokens
    
    def test_generate_with_top_p(self, small_model):
        """测试 Top-P (nucleus) 采样生成"""
        batch_size = 2
        prompt_len = 5
        max_new_tokens = 10
        
        input_ids = torch.randint(0, small_model.vocab_size, (batch_size, prompt_len))
        
        generated = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9
        )
        
        assert generated.shape[0] == batch_size
        assert generated.shape[1] == prompt_len + max_new_tokens
    
    def test_generate_with_eos_token(self, small_model):
        """测试带 EOS token 的生成"""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 20
        eos_token_id = 1
        
        input_ids = torch.randint(2, small_model.vocab_size, (batch_size, prompt_len))
        
        generated = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id
        )
        
        # 生成的序列长度应该 <= prompt_len + max_new_tokens
        assert generated.shape[1] <= prompt_len + max_new_tokens
    
    def test_generate_with_temperature(self, small_model):
        """测试不同温度参数的生成"""
        batch_size = 1
        prompt_len = 5
        max_new_tokens = 5
        
        input_ids = torch.randint(0, small_model.vocab_size, (batch_size, prompt_len))
        
        # 低温度应该更确定性
        generated_low_temp = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.1
        )
        
        # 高温度应该更随机
        generated_high_temp = small_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=2.0
        )
        
        assert generated_low_temp.shape == generated_high_temp.shape


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
