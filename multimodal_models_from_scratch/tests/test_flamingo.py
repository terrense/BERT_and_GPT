"""
Flamingo 模型单元测试

测试 Flamingo 模型的各个组件和功能：
- FlamingoDecoderLayer 有无交叉注意力
- FlamingoModel 单图像前向传播
- FlamingoModel 多图像前向传播
- FlamingoModel 纯文本前向传播
- 门控交叉注意力门控参数初始化
- 交叉注意力层位置（每 N 层）
- KV Cache 功能
- 带图像的文本生成
- 冻结/解冻功能

需求: 8.7
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.config import (
    FlamingoConfig,
    VisionConfig,
    LLaMAConfig
)
from multimodal_models_from_scratch.multimodal.flamingo import (
    FlamingoDecoderLayer,
    FlamingoModel
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_vision_config():
    """小型视觉编码器配置（用于快速测试）"""
    return VisionConfig(
        image_size=64,
        patch_size=16,
        in_channels=3,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        dropout_rate=0.1,
        num_classes=0
    )


@pytest.fixture
def small_llm_config():
    """小型 LLM 配置"""
    return LLaMAConfig(
        vocab_size=1000,
        d_model=64,
        num_heads=4,
        num_kv_heads=2,
        num_layers=8,  # 8 层以便测试交叉注意力位置
        d_ff=128,
        max_seq_len=64,
        dropout_rate=0.0,
        rope_theta=10000.0,
        tie_weights=False
    )


@pytest.fixture
def small_flamingo_config(small_vision_config, small_llm_config):
    """小型 Flamingo 配置"""
    return FlamingoConfig(
        vision_config=small_vision_config,
        llm_config=small_llm_config,
        perceiver_num_latents=8,
        perceiver_depth=2,
        cross_attn_every_n_layers=4,  # 每 4 层插入交叉注意力
        freeze_vision=True,
        freeze_llm=True
    )


@pytest.fixture
def flamingo_model(small_flamingo_config):
    """创建 Flamingo 模型实例"""
    model = FlamingoModel(small_flamingo_config)
    model.eval()
    return model


@pytest.fixture
def sample_inputs(small_flamingo_config):
    """创建示例输入"""
    batch_size = 2
    image_size = small_flamingo_config.vision_config.image_size
    seq_len = 16
    vocab_size = small_flamingo_config.llm_config.vocab_size
    
    # 单图像输入
    single_image = torch.randn(batch_size, 3, image_size, image_size)
    
    # 多图像输入 (batch, num_images, 3, H, W)
    num_images = 3
    multi_images = torch.randn(batch_size, num_images, 3, image_size, image_size)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return {
        'single_image': single_image,
        'multi_images': multi_images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# ============================================================================
# 测试 FlamingoDecoderLayer
# ============================================================================

class TestFlamingoDecoderLayer:
    """测试 FlamingoDecoderLayer"""
    
    def test_layer_without_cross_attention(self, small_llm_config):
        """测试不带交叉注意力的 FlamingoDecoderLayer"""
        layer = FlamingoDecoderLayer(
            config=small_llm_config,
            layer_idx=0,
            has_cross_attention=False
        )
        
        batch_size = 2
        seq_len = 16
        d_model = small_llm_config.d_model
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        output, _ = layer(hidden_states)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert layer.gated_cross_attn is None
    
    def test_layer_with_cross_attention(self, small_llm_config):
        """测试带交叉注意力的 FlamingoDecoderLayer"""
        layer = FlamingoDecoderLayer(
            config=small_llm_config,
            layer_idx=0,
            has_cross_attention=True
        )
        
        batch_size = 2
        seq_len = 16
        num_visual_tokens = 8
        d_model = small_llm_config.d_model
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output, _ = layer(hidden_states, visual_features=visual_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert layer.gated_cross_attn is not None
    
    def test_layer_with_cross_attention_no_visual(self, small_llm_config):
        """测试带交叉注意力层但不传入视觉特征"""
        layer = FlamingoDecoderLayer(
            config=small_llm_config,
            layer_idx=0,
            has_cross_attention=True
        )
        
        batch_size = 2
        seq_len = 16
        d_model = small_llm_config.d_model
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # 不传入视觉特征，应该跳过交叉注意力
        output, _ = layer(hidden_states, visual_features=None)
        
        assert output.shape == (batch_size, seq_len, d_model)


# ============================================================================
# 测试 FlamingoModel 初始化
# ============================================================================

class TestFlamingoModelInitialization:
    """测试 Flamingo 模型初始化"""
    
    def test_model_creation(self, small_flamingo_config):
        """测试模型可以正常创建"""
        model = FlamingoModel(small_flamingo_config)
        assert model is not None
    
    def test_model_components_exist(self, flamingo_model):
        """测试模型包含所有必要组件"""
        # 需求 8.1: 冻结的 Vision Encoder
        assert hasattr(flamingo_model, 'vision_encoder')
        
        # 需求 8.2: Perceiver Resampler
        assert hasattr(flamingo_model, 'perceiver_resampler')
        
        # LLM 组件
        assert hasattr(flamingo_model, 'embed_tokens')
        assert hasattr(flamingo_model, 'layers')
        assert hasattr(flamingo_model, 'norm')
        assert hasattr(flamingo_model, 'lm_head')
        assert hasattr(flamingo_model, 'rotary_emb')
    
    def test_config_stored(self, flamingo_model, small_flamingo_config):
        """测试配置被正确存储"""
        assert flamingo_model.config == small_flamingo_config
        assert flamingo_model.config.perceiver_num_latents == 8
        assert flamingo_model.config.cross_attn_every_n_layers == 4


# ============================================================================
# 测试单图像前向传播
# ============================================================================

class TestSingleImageForward:
    """测试单图像前向传播"""
    
    def test_forward_with_single_image(self, flamingo_model, sample_inputs):
        """测试单图像输入的前向传播"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=single_image
            )
        
        batch_size, seq_len = input_ids.shape
        vocab_size = flamingo_model.config.llm_config.vocab_size
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
        assert 'hidden_states' in output
    
    def test_forward_with_single_image_and_labels(self, flamingo_model, sample_inputs):
        """测试单图像输入带标签的前向传播"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=single_image,
                labels=labels
            )
        
        assert 'loss' in output
        assert output['loss'].dim() == 0
        assert output['loss'].item() > 0


# ============================================================================
# 测试多图像前向传播 (需求 8.7)
# ============================================================================

class TestMultiImageForward:
    """测试多图像前向传播"""
    
    def test_forward_with_multiple_images(self, flamingo_model, sample_inputs):
        """需求 8.7: 测试多图像输入的前向传播"""
        input_ids = sample_inputs['input_ids']
        multi_images = sample_inputs['multi_images']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=multi_images
            )
        
        batch_size, seq_len = input_ids.shape
        vocab_size = flamingo_model.config.llm_config.vocab_size
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_encode_multiple_images(self, flamingo_model, sample_inputs):
        """测试多图像编码"""
        multi_images = sample_inputs['multi_images']
        batch_size, num_images = multi_images.shape[:2]
        
        with torch.no_grad():
            visual_features = flamingo_model.encode_images(multi_images)
        
        # 输出形状应该是 (batch, num_images * num_latents, d_model)
        num_latents = flamingo_model.config.perceiver_num_latents
        llm_d_model = flamingo_model.config.llm_config.d_model
        
        expected_visual_tokens = num_images * num_latents
        assert visual_features.shape == (batch_size, expected_visual_tokens, llm_d_model)
    
    def test_encode_single_image(self, flamingo_model, sample_inputs):
        """测试单图像编码"""
        single_image = sample_inputs['single_image']
        batch_size = single_image.shape[0]
        
        with torch.no_grad():
            visual_features = flamingo_model.encode_images(single_image)
        
        # 输出形状应该是 (batch, num_latents, d_model)
        num_latents = flamingo_model.config.perceiver_num_latents
        llm_d_model = flamingo_model.config.llm_config.d_model
        
        assert visual_features.shape == (batch_size, num_latents, llm_d_model)
    
    def test_multi_image_with_labels(self, flamingo_model, sample_inputs):
        """测试多图像输入带标签的前向传播"""
        input_ids = sample_inputs['input_ids']
        multi_images = sample_inputs['multi_images']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=multi_images,
                labels=labels
            )
        
        assert 'loss' in output
        assert output['loss'].dim() == 0


# ============================================================================
# 测试纯文本前向传播
# ============================================================================

class TestPureTextForward:
    """测试纯文本前向传播（无图像）"""
    
    def test_forward_without_images(self, flamingo_model, sample_inputs):
        """测试纯文本输入的前向传播"""
        input_ids = sample_inputs['input_ids']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=None
            )
        
        batch_size, seq_len = input_ids.shape
        vocab_size = flamingo_model.config.llm_config.vocab_size
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_forward_without_images_with_labels(self, flamingo_model, sample_inputs):
        """测试纯文本输入带标签的前向传播"""
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=None,
                labels=labels
            )
        
        assert 'loss' in output
        assert output['loss'].dim() == 0
        assert output['loss'].item() > 0


# ============================================================================
# 测试门控交叉注意力初始化 (需求 8.6)
# ============================================================================

class TestGatedCrossAttentionInitialization:
    """测试门控交叉注意力门控参数初始化"""
    
    def test_gate_initialized_to_zero(self, flamingo_model):
        """需求 8.6: 门控参数 tanh(alpha) 应该初始化为 0"""
        for layer in flamingo_model.layers:
            if layer.gated_cross_attn is not None:
                # alpha 应该初始化为 0
                alpha = layer.gated_cross_attn.alpha
                assert torch.allclose(alpha, torch.zeros_like(alpha)), \
                    "门控参数 alpha 应该初始化为 0"
                
                # tanh(alpha) 也应该是 0
                gate_value = layer.gated_cross_attn.get_gate_value()
                assert torch.allclose(gate_value, torch.zeros_like(gate_value)), \
                    "门控值 tanh(alpha) 应该初始化为 0"
    
    def test_gate_value_range(self, flamingo_model):
        """测试门控值范围在 [-1, 1]"""
        for layer in flamingo_model.layers:
            if layer.gated_cross_attn is not None:
                gate_value = layer.gated_cross_attn.get_gate_value()
                assert gate_value.item() >= -1.0 and gate_value.item() <= 1.0


# ============================================================================
# 测试交叉注意力层位置 (需求 8.5)
# ============================================================================

class TestCrossAttentionLayerPositioning:
    """测试交叉注意力层位置"""
    
    def test_cross_attention_every_n_layers(self, flamingo_model, small_flamingo_config):
        """需求 8.5: 每隔 N 层插入交叉注意力"""
        n = small_flamingo_config.cross_attn_every_n_layers  # 4
        num_layers = small_flamingo_config.llm_config.num_layers  # 8
        
        # 检查每一层是否正确配置了交叉注意力
        for i, layer in enumerate(flamingo_model.layers):
            # 每隔 N 层插入交叉注意力（在第 N-1, 2N-1, 3N-1, ... 层）
            # 即 (i + 1) % n == 0 时有交叉注意力
            should_have_cross_attn = (i + 1) % n == 0
            
            if should_have_cross_attn:
                assert layer.has_cross_attention, \
                    f"Layer {i} should have cross attention"
                assert layer.gated_cross_attn is not None, \
                    f"Layer {i} should have gated_cross_attn module"
            else:
                assert not layer.has_cross_attention, \
                    f"Layer {i} should not have cross attention"
                assert layer.gated_cross_attn is None, \
                    f"Layer {i} should not have gated_cross_attn module"
    
    def test_cross_attention_count(self, flamingo_model, small_flamingo_config):
        """测试交叉注意力层数量"""
        n = small_flamingo_config.cross_attn_every_n_layers  # 4
        num_layers = small_flamingo_config.llm_config.num_layers  # 8
        
        # 预期的交叉注意力层数量
        expected_count = num_layers // n  # 8 // 4 = 2
        
        actual_count = sum(
            1 for layer in flamingo_model.layers 
            if layer.has_cross_attention
        )
        
        assert actual_count == expected_count, \
            f"Expected {expected_count} cross attention layers, got {actual_count}"


# ============================================================================
# 测试 KV Cache 功能
# ============================================================================

class TestKVCache:
    """测试 KV Cache 功能"""
    
    def test_kv_cache_returned(self, flamingo_model, sample_inputs):
        """测试 KV Cache 被正确返回"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=single_image,
                use_cache=True
            )
        
        assert 'past_key_values' in output
        past_key_values = output['past_key_values']
        
        # 应该有与层数相同数量的 KV Cache
        num_layers = flamingo_model.config.llm_config.num_layers
        assert len(past_key_values) == num_layers
    
    def test_kv_cache_shape(self, flamingo_model, sample_inputs):
        """测试 KV Cache 形状"""
        input_ids = sample_inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=None,
                use_cache=True
            )
        
        past_key_values = output['past_key_values']
        
        # 检查第一层的 KV Cache 形状
        k_cache, v_cache = past_key_values[0]
        
        num_kv_heads = flamingo_model.config.llm_config.num_kv_heads
        head_dim = flamingo_model.config.llm_config.d_model // flamingo_model.config.llm_config.num_heads
        
        assert k_cache.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert v_cache.shape == (batch_size, num_kv_heads, seq_len, head_dim)

    
    def test_kv_cache_incremental_generation(self, flamingo_model, sample_inputs):
        """测试 KV Cache 增量生成"""
        input_ids = sample_inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # 第一次前向传播
        with torch.no_grad():
            output1 = flamingo_model(
                input_ids=input_ids,
                images=None,
                use_cache=True
            )
        
        past_key_values = output1['past_key_values']
        
        # 第二次前向传播，只传入一个新 token
        new_token = torch.randint(0, 100, (batch_size, 1))
        
        with torch.no_grad():
            output2 = flamingo_model(
                input_ids=new_token,
                images=None,
                past_key_values=past_key_values,
                use_cache=True
            )
        
        # 输出应该只有一个 token
        assert output2['logits'].shape[1] == 1
        
        # 新的 KV Cache 应该包含 seq_len + 1 个 token
        new_past_key_values = output2['past_key_values']
        k_cache, v_cache = new_past_key_values[0]
        assert k_cache.shape[2] == seq_len + 1


# ============================================================================
# 测试带图像的文本生成
# ============================================================================

class TestTextGenerationWithImages:
    """测试带图像的文本生成"""
    
    def test_generate_with_single_image(self, flamingo_model, sample_inputs):
        """测试单图像条件生成"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5
            )
        
        # 输出应该是 2D 张量
        assert generated_ids.dim() == 2
        # 批次大小应该匹配
        assert generated_ids.shape[0] == batch_size
        # 长度应该大于输入长度
        assert generated_ids.shape[1] > input_ids.shape[1]
    
    def test_generate_with_multiple_images(self, flamingo_model, sample_inputs):
        """测试多图像条件生成"""
        input_ids = sample_inputs['input_ids']
        multi_images = sample_inputs['multi_images']
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=multi_images,
                max_new_tokens=5
            )
        
        assert generated_ids.dim() == 2
        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] > input_ids.shape[1]
    
    def test_generate_greedy(self, flamingo_model, sample_inputs):
        """测试贪婪解码"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        with torch.no_grad():
            generated_ids_1 = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5,
                do_sample=False
            )
            generated_ids_2 = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5,
                do_sample=False
            )
        
        # 贪婪解码应该是确定性的
        assert torch.equal(generated_ids_1, generated_ids_2)

    
    def test_generate_with_sampling(self, flamingo_model, sample_inputs):
        """测试采样生成"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5,
                do_sample=True,
                temperature=1.0
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
        assert generated_ids.shape[1] > input_ids.shape[1]
    
    def test_generate_with_top_k(self, flamingo_model, sample_inputs):
        """测试 Top-K 采样"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5,
                do_sample=True,
                top_k=50
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
    
    def test_generate_with_top_p(self, flamingo_model, sample_inputs):
        """测试 Top-P 采样"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=single_image,
                max_new_tokens=5,
                do_sample=True,
                top_p=0.9
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
    
    def test_generate_without_images(self, flamingo_model, sample_inputs):
        """测试纯文本生成"""
        input_ids = sample_inputs['input_ids']
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            generated_ids = flamingo_model.generate(
                input_ids=input_ids,
                images=None,
                max_new_tokens=5
            )
        
        assert generated_ids.dim() == 2
        assert generated_ids.shape[0] == batch_size


# ============================================================================
# 测试冻结/解冻功能 (需求 8.8)
# ============================================================================

class TestFreezeUnfreeze:
    """测试冻结/解冻功能"""
    
    def test_vision_encoder_frozen(self, flamingo_model):
        """需求 8.8: Vision Encoder 应该被冻结"""
        for param in flamingo_model.vision_encoder.parameters():
            assert not param.requires_grad, "Vision Encoder 参数应该被冻结"
    
    def test_llm_frozen_except_cross_attention(self, flamingo_model):
        """需求 8.8: LLM 应该被冻结（门控交叉注意力除外）"""
        # Token Embedding 应该被冻结
        for param in flamingo_model.embed_tokens.parameters():
            assert not param.requires_grad, "Token Embedding 应该被冻结"
        
        # LM Head 应该被冻结
        for param in flamingo_model.lm_head.parameters():
            assert not param.requires_grad, "LM Head 应该被冻结"
        
        # Final Norm 应该被冻结
        for param in flamingo_model.norm.parameters():
            assert not param.requires_grad, "Final Norm 应该被冻结"
        
        # LLaMA Decoder Layer 应该被冻结
        for layer in flamingo_model.layers:
            for param in layer.llama_layer.parameters():
                assert not param.requires_grad, "LLaMA Decoder Layer 应该被冻结"

    
    def test_gated_cross_attention_trainable(self, flamingo_model):
        """需求 8.8: 门控交叉注意力应该是可训练的"""
        for layer in flamingo_model.layers:
            if layer.gated_cross_attn is not None:
                for param in layer.gated_cross_attn.parameters():
                    assert param.requires_grad, "门控交叉注意力参数应该是可训练的"
    
    def test_perceiver_resampler_trainable(self, flamingo_model):
        """需求 8.8: Perceiver Resampler 应该是可训练的"""
        for param in flamingo_model.perceiver_resampler.parameters():
            assert param.requires_grad, "Perceiver Resampler 参数应该是可训练的"
    
    def test_unfreeze_all(self, flamingo_model):
        """测试解冻所有参数"""
        flamingo_model.unfreeze_all()
        
        # 所有参数应该是可训练的
        for param in flamingo_model.parameters():
            assert param.requires_grad, "解冻后所有参数应该是可训练的"
    
    def test_trainable_params_count(self, flamingo_model):
        """测试可训练参数数量"""
        trainable_params = flamingo_model.get_num_trainable_params()
        total_params = flamingo_model.get_num_total_params()
        
        # 可训练参数应该少于总参数
        assert trainable_params < total_params
        assert trainable_params > 0


# ============================================================================
# 测试梯度流
# ============================================================================

class TestGradientFlow:
    """测试梯度流"""
    
    def test_gradient_flow_with_images(self, flamingo_model, sample_inputs):
        """测试带图像的梯度流"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        labels = sample_inputs['labels']
        
        flamingo_model.train()
        
        output = flamingo_model(
            input_ids=input_ids,
            images=single_image,
            labels=labels
        )
        
        loss = output['loss']
        loss.backward()
        
        # Perceiver Resampler 应该有梯度
        for param in flamingo_model.perceiver_resampler.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Perceiver Resampler 参数应该有梯度"
        
        # 门控交叉注意力应该有梯度
        for layer in flamingo_model.layers:
            if layer.gated_cross_attn is not None:
                for param in layer.gated_cross_attn.parameters():
                    if param.requires_grad:
                        assert param.grad is not None, "门控交叉注意力参数应该有梯度"
        
        # Vision Encoder 不应该有梯度
        for param in flamingo_model.vision_encoder.parameters():
            assert param.grad is None, "Vision Encoder 参数不应该有梯度"
        
        flamingo_model.zero_grad()


# ============================================================================
# 测试输出隐藏状态
# ============================================================================

class TestOutputHiddenStates:
    """测试输出隐藏状态"""
    
    def test_output_hidden_states(self, flamingo_model, sample_inputs):
        """测试输出所有层的隐藏状态"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=single_image,
                output_hidden_states=True
            )
        
        assert 'all_hidden_states' in output
        all_hidden_states = output['all_hidden_states']
        
        # 应该有 num_layers + 1 个隐藏状态（包括输入嵌入）
        num_layers = flamingo_model.config.llm_config.num_layers
        assert len(all_hidden_states) == num_layers + 1
    
    def test_hidden_states_shape(self, flamingo_model, sample_inputs):
        """测试隐藏状态形状"""
        input_ids = sample_inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        d_model = flamingo_model.config.llm_config.d_model
        
        with torch.no_grad():
            output = flamingo_model(
                input_ids=input_ids,
                images=None,
                output_hidden_states=True
            )
        
        all_hidden_states = output['all_hidden_states']
        
        for hidden_state in all_hidden_states:
            assert hidden_state.shape == (batch_size, seq_len, d_model)


# ============================================================================
# 测试不同批次大小
# ============================================================================

class TestDifferentBatchSizes:
    """测试不同批次大小"""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, small_flamingo_config, batch_size):
        """测试不同批次大小的前向传播"""
        model = FlamingoModel(small_flamingo_config)
        model.eval()
        
        image_size = small_flamingo_config.vision_config.image_size
        seq_len = 16
        vocab_size = small_flamingo_config.llm_config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        images = torch.randn(batch_size, 3, image_size, image_size)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, images=images)
        
        assert output['logits'].shape[0] == batch_size
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_generate_different_batch_sizes(self, small_flamingo_config, batch_size):
        """测试不同批次大小的生成"""
        model = FlamingoModel(small_flamingo_config)
        model.eval()
        
        image_size = small_flamingo_config.vision_config.image_size
        seq_len = 8
        vocab_size = small_flamingo_config.llm_config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        images = torch.randn(batch_size, 3, image_size, image_size)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                images=images,
                max_new_tokens=3
            )
        
        assert generated_ids.shape[0] == batch_size


# ============================================================================
# 测试不同图像数量
# ============================================================================

class TestDifferentImageCounts:
    """测试不同图像数量"""
    
    @pytest.mark.parametrize("num_images", [1, 2, 3, 5])
    def test_forward_different_image_counts(self, small_flamingo_config, num_images):
        """测试不同图像数量的前向传播"""
        model = FlamingoModel(small_flamingo_config)
        model.eval()
        
        batch_size = 2
        image_size = small_flamingo_config.vision_config.image_size
        seq_len = 16
        vocab_size = small_flamingo_config.llm_config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        images = torch.randn(batch_size, num_images, 3, image_size, image_size)
        
        with torch.no_grad():
            output = model(input_ids=input_ids, images=images)
        
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    @pytest.mark.parametrize("num_images", [1, 2, 3])
    def test_encode_different_image_counts(self, small_flamingo_config, num_images):
        """测试不同图像数量的编码"""
        model = FlamingoModel(small_flamingo_config)
        model.eval()
        
        batch_size = 2
        image_size = small_flamingo_config.vision_config.image_size
        
        images = torch.randn(batch_size, num_images, 3, image_size, image_size)
        
        with torch.no_grad():
            visual_features = model.encode_images(images)
        
        num_latents = small_flamingo_config.perceiver_num_latents
        llm_d_model = small_flamingo_config.llm_config.d_model
        
        expected_visual_tokens = num_images * num_latents
        assert visual_features.shape == (batch_size, expected_visual_tokens, llm_d_model)


# ============================================================================
# 测试模型表示
# ============================================================================

class TestModelRepresentation:
    """测试模型表示"""
    
    def test_extra_repr(self, flamingo_model):
        """测试 extra_repr 方法"""
        repr_str = flamingo_model.extra_repr()
        
        assert 'perceiver_num_latents' in repr_str
        assert 'cross_attn_every_n_layers' in repr_str
        assert 'freeze_vision' in repr_str
        assert 'freeze_llm' in repr_str
    
    def test_model_str(self, flamingo_model):
        """测试模型字符串表示"""
        model_str = str(flamingo_model)
        
        # 应该包含主要组件
        assert 'vision_encoder' in model_str
        assert 'perceiver_resampler' in model_str
        assert 'layers' in model_str
    
    def test_decoder_layer_extra_repr(self, small_llm_config):
        """测试 FlamingoDecoderLayer 的 extra_repr"""
        layer = FlamingoDecoderLayer(
            config=small_llm_config,
            layer_idx=3,
            has_cross_attention=True
        )
        
        repr_str = layer.extra_repr()
        
        assert 'layer_idx=3' in repr_str
        assert 'has_cross_attention=True' in repr_str


# ============================================================================
# 测试 prepare_inputs_for_generation
# ============================================================================

class TestPrepareInputsForGeneration:
    """测试 prepare_inputs_for_generation"""
    
    def test_prepare_inputs_first_step(self, flamingo_model, sample_inputs):
        """测试第一步生成的输入准备"""
        input_ids = sample_inputs['input_ids']
        single_image = sample_inputs['single_image']
        
        prepared = flamingo_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            images=single_image
        )
        
        assert 'input_ids' in prepared
        assert 'images' in prepared
        assert 'use_cache' in prepared
        assert prepared['use_cache'] is True
        assert prepared['images'] is not None
    
    def test_prepare_inputs_subsequent_step(self, flamingo_model, sample_inputs):
        """测试后续步骤生成的输入准备"""
        input_ids = sample_inputs['input_ids']
        batch_size, seq_len = input_ids.shape
        
        # 模拟 KV Cache
        num_layers = flamingo_model.config.llm_config.num_layers
        num_kv_heads = flamingo_model.config.llm_config.num_kv_heads
        head_dim = flamingo_model.config.llm_config.d_model // flamingo_model.config.llm_config.num_heads
        
        past_key_values = [
            (
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
            )
            for _ in range(num_layers)
        ]
        
        prepared = flamingo_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            images=sample_inputs['single_image']
        )
        
        # 后续步骤只需要最后一个 token
        assert prepared['input_ids'].shape[1] == 1
        # 后续步骤不需要图像
        assert prepared['images'] is None
        # 应该有 position_ids
        assert prepared['position_ids'] is not None
