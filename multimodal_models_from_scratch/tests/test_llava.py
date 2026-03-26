"""
LLaVA 模型单元测试

测试 LLaVA 模型的各个组件和功能：
- LLaVAModel 初始化和组件
- 带图像和 <image> token 的前向传播
- 纯文本前向传播（无图像）
- 视觉 token 插入到 <image> 位置
- 视觉 token 位置的 labels 掩码
- KV Cache 功能
- 带图像的文本生成
- 两阶段训练的冻结/解冻功能
- 多轮对话输入准备
- 不同批次大小和序列长度

需求: 9.8
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.config import (
    LLaVAConfig,
    VisionConfig,
    LLaMAConfig
)
from multimodal_models_from_scratch.multimodal.llava import (
    LLaVAModel,
    DEFAULT_IMAGE_TOKEN_ID
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
        num_layers=4,
        d_ff=128,
        max_seq_len=64,
        dropout_rate=0.0,
        rope_theta=10000.0,
        tie_weights=False
    )


@pytest.fixture
def small_llava_config(small_vision_config, small_llm_config):
    """小型 LLaVA 配置"""
    return LLaVAConfig(
        vision_config=small_vision_config,
        llm_config=small_llm_config,
        projection_type='mlp',
        freeze_vision=True,
        freeze_llm=False
    )


@pytest.fixture
def llava_model(small_llava_config):
    """创建 LLaVA 模型实例"""
    model = LLaVAModel(small_llava_config)
    model.eval()
    return model


@pytest.fixture
def sample_inputs(small_llava_config):
    """创建示例输入"""
    batch_size = 2
    image_size = small_llava_config.vision_config.image_size
    seq_len = 16
    vocab_size = small_llava_config.llm_config.vocab_size
    
    # 图像输入
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    
    # 带 <image> token 的输入
    input_ids_with_image = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 在位置 4 插入 <image> token
    input_ids_with_image[:, 4] = DEFAULT_IMAGE_TOKEN_ID
    
    # 纯文本输入（无 <image> token）
    input_ids_text_only = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return {
        'pixel_values': pixel_values,
        'input_ids_with_image': input_ids_with_image,
        'input_ids_text_only': input_ids_text_only,
        'attention_mask': attention_mask,
        'labels': labels
    }


# ============================================================================
# 测试 LLaVAModel 初始化
# ============================================================================

class TestLLaVAModelInitialization:
    """测试 LLaVA 模型初始化"""
    
    def test_model_creation(self, small_llava_config):
        """测试模型可以正常创建"""
        model = LLaVAModel(small_llava_config)
        assert model is not None
    
    def test_model_components_exist(self, llava_model):
        """测试模型包含所有必要组件"""
        # 需求 9.1: Vision Encoder
        assert hasattr(llava_model, 'vision_encoder')
        
        # 需求 9.2: Visual Projection (MLP)
        assert hasattr(llava_model, 'visual_projection')
        
        # 需求 9.3: LLM (LLaMA)
        assert hasattr(llava_model, 'llm')
    
    def test_config_stored(self, llava_model, small_llava_config):
        """测试配置被正确存储"""
        assert llava_model.config == small_llava_config
        assert llava_model.config.projection_type == 'mlp'
        assert llava_model.config.freeze_vision == True
        assert llava_model.config.freeze_llm == False
    
    def test_linear_projection_type(self, small_vision_config, small_llm_config):
        """测试线性投影类型"""
        config = LLaVAConfig(
            vision_config=small_vision_config,
            llm_config=small_llm_config,
            projection_type='linear',
            freeze_vision=True,
            freeze_llm=False
        )
        model = LLaVAModel(config)
        assert model.config.projection_type == 'linear'


# ============================================================================
# 测试带图像和 <image> token 的前向传播
# ============================================================================

class TestForwardWithImage:
    """测试带图像和 <image> token 的前向传播"""
    
    def test_forward_with_image_and_image_token(self, llava_model, sample_inputs):
        """需求 9.4, 9.5: 测试带图像和 <image> token 的前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        batch_size = input_ids.shape[0]
        vocab_size = llava_model.config.llm_config.vocab_size
        
        assert 'logits' in output
        assert 'hidden_states' in output
        # 输出序列长度应该是原序列长度 - 1 + num_visual_tokens
        # 因为 <image> token 被替换为多个视觉 token
    
    def test_forward_with_image_and_labels(self, llava_model, sample_inputs):
        """测试带图像和标签的前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
        
        assert 'loss' in output
        assert output['loss'].dim() == 0
        assert output['loss'].item() > 0
    
    def test_forward_with_attention_mask(self, llava_model, sample_inputs):
        """测试带注意力掩码的前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        attention_mask = sample_inputs['attention_mask']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        assert 'logits' in output


# ============================================================================
# 测试纯文本前向传播（无图像）
# ============================================================================

class TestPureTextForward:
    """测试纯文本前向传播（无图像）"""
    
    def test_forward_without_image(self, llava_model, sample_inputs):
        """测试纯文本输入的前向传播"""
        input_ids = sample_inputs['input_ids_text_only']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids
            )
        
        batch_size, seq_len = input_ids.shape
        vocab_size = llava_model.config.llm_config.vocab_size
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)
    
    def test_forward_without_image_with_labels(self, llava_model, sample_inputs):
        """测试纯文本输入带标签的前向传播"""
        input_ids = sample_inputs['input_ids_text_only']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                labels=labels
            )
        
        assert 'loss' in output
        assert output['loss'].dim() == 0
        assert output['loss'].item() > 0
    
    def test_forward_with_image_but_no_image_token(self, llava_model, sample_inputs):
        """测试有图像但无 <image> token 的前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_text_only']  # 无 <image> token
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        batch_size, seq_len = input_ids.shape
        vocab_size = llava_model.config.llm_config.vocab_size
        
        # 没有 <image> token，输出形状应该与输入相同
        assert output['logits'].shape == (batch_size, seq_len, vocab_size)


# ============================================================================
# 测试视觉 token 插入到 <image> 位置 (需求 9.4)
# ============================================================================

class TestVisualTokenInsertion:
    """测试视觉 token 插入到 <image> 位置"""
    
    def test_get_vision_features(self, llava_model, sample_inputs):
        """测试获取视觉特征"""
        pixel_values = sample_inputs['pixel_values']
        batch_size = pixel_values.shape[0]
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
        
        # 视觉 token 形状应该是 (batch, num_patches, llm_dim)
        llm_dim = llava_model.config.llm_config.d_model
        assert visual_tokens.shape[0] == batch_size
        assert visual_tokens.shape[2] == llm_dim
    
    def test_visual_token_count(self, llava_model, sample_inputs):
        """测试视觉 token 数量"""
        pixel_values = sample_inputs['pixel_values']
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
        
        # num_patches = (image_size / patch_size) ** 2
        image_size = llava_model.config.vision_config.image_size
        patch_size = llava_model.config.vision_config.patch_size
        expected_num_patches = (image_size // patch_size) ** 2
        
        assert visual_tokens.shape[1] == expected_num_patches
    
    def test_output_sequence_length_with_image(self, llava_model, sample_inputs):
        """测试带图像时的输出序列长度"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
            num_visual_tokens = visual_tokens.shape[1]
            
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        # 新序列长度 = 原序列长度 - 1 (移除 <image>) + num_visual_tokens
        expected_seq_len = seq_len - 1 + num_visual_tokens
        assert output['logits'].shape[1] == expected_seq_len
    
    def test_merge_visual_tokens_internal(self, llava_model, sample_inputs):
        """测试内部视觉 token 合并方法"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
            
            merged_embeds, merged_mask, position_ids = llava_model._merge_visual_tokens(
                input_ids=input_ids,
                visual_tokens=visual_tokens,
                image_token_index=DEFAULT_IMAGE_TOKEN_ID
            )
        
        batch_size, seq_len = input_ids.shape
        num_visual_tokens = visual_tokens.shape[1]
        expected_seq_len = seq_len - 1 + num_visual_tokens
        
        assert merged_embeds.shape[0] == batch_size
        assert merged_embeds.shape[1] == expected_seq_len
        assert merged_mask.shape == (batch_size, expected_seq_len)
        assert position_ids.shape == (batch_size, expected_seq_len)


# ============================================================================
# 测试视觉 token 位置的 labels 掩码 (需求 9.5)
# ============================================================================

class TestLabelsMasking:
    """测试视觉 token 位置的 labels 掩码"""
    
    def test_create_labels_mask(self, llava_model, sample_inputs):
        """需求 9.5: 测试 labels 掩码创建"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        labels = sample_inputs['labels']
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
            num_visual_tokens = visual_tokens.shape[1]
            
            new_labels = llava_model._create_labels_mask(
                labels=labels,
                input_ids=input_ids,
                num_visual_tokens=num_visual_tokens,
                image_token_index=DEFAULT_IMAGE_TOKEN_ID
            )
        
        batch_size, seq_len = labels.shape
        expected_seq_len = seq_len - 1 + num_visual_tokens
        
        assert new_labels.shape == (batch_size, expected_seq_len)
    
    def test_visual_tokens_masked_in_labels(self, llava_model, sample_inputs):
        """测试视觉 token 位置的 labels 被设为 -100"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        labels = sample_inputs['labels']
        
        # 找到 <image> token 位置
        image_pos = (input_ids[0] == DEFAULT_IMAGE_TOKEN_ID).nonzero(as_tuple=True)[0][0].item()
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
            num_visual_tokens = visual_tokens.shape[1]
            
            new_labels = llava_model._create_labels_mask(
                labels=labels,
                input_ids=input_ids,
                num_visual_tokens=num_visual_tokens,
                image_token_index=DEFAULT_IMAGE_TOKEN_ID
            )
        
        # 检查视觉 token 位置的 labels 是否为 -100
        for i in range(num_visual_tokens):
            assert new_labels[0, image_pos + i].item() == -100, \
                f"Visual token position {image_pos + i} should be masked with -100"
    
    def test_labels_without_image_token(self, llava_model, sample_inputs):
        """测试无 <image> token 时 labels 不变"""
        input_ids = sample_inputs['input_ids_text_only']
        labels = sample_inputs['labels']
        
        new_labels = llava_model._create_labels_mask(
            labels=labels,
            input_ids=input_ids,
            num_visual_tokens=16,
            image_token_index=DEFAULT_IMAGE_TOKEN_ID
        )
        
        # 没有 <image> token，labels 应该不变
        assert torch.equal(new_labels, labels)


# ============================================================================
# 测试 KV Cache 功能
# ============================================================================

class TestKVCache:
    """测试 KV Cache 功能"""
    
    def test_kv_cache_returned(self, llava_model, sample_inputs):
        """测试 KV Cache 被正确返回"""
        input_ids = sample_inputs['input_ids_text_only']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                use_cache=True
            )
        
        assert 'past_key_values' in output
        past_key_values = output['past_key_values']
        
        # 应该有与层数相同数量的 KV Cache
        num_layers = llava_model.config.llm_config.num_layers
        assert len(past_key_values) == num_layers
    
    def test_kv_cache_shape(self, llava_model, sample_inputs):
        """测试 KV Cache 形状"""
        input_ids = sample_inputs['input_ids_text_only']
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                use_cache=True
            )
        
        past_key_values = output['past_key_values']
        
        # 检查第一层的 KV Cache 形状
        k_cache, v_cache = past_key_values[0]
        
        num_kv_heads = llava_model.config.llm_config.num_kv_heads
        head_dim = llava_model.config.llm_config.d_model // llava_model.config.llm_config.num_heads
        
        assert k_cache.shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert v_cache.shape == (batch_size, num_kv_heads, seq_len, head_dim)
    
    def test_kv_cache_with_image(self, llava_model, sample_inputs):
        """测试带图像的 KV Cache"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        batch_size, seq_len = input_ids.shape
        
        with torch.no_grad():
            visual_tokens = llava_model.get_vision_features(pixel_values)
            num_visual_tokens = visual_tokens.shape[1]
            
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                use_cache=True
            )
        
        past_key_values = output['past_key_values']
        k_cache, v_cache = past_key_values[0]
        
        # KV Cache 的序列长度应该是新序列长度
        expected_seq_len = seq_len - 1 + num_visual_tokens
        assert k_cache.shape[2] == expected_seq_len
    
    def test_kv_cache_incremental_generation(self, llava_model, sample_inputs):
        """测试 KV Cache 增量生成"""
        input_ids = sample_inputs['input_ids_text_only']
        batch_size, seq_len = input_ids.shape
        
        # 第一次前向传播
        with torch.no_grad():
            output1 = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                use_cache=True
            )
        
        past_key_values = output1['past_key_values']
        
        # 第二次前向传播，只传入一个新 token
        new_token = torch.randint(0, 100, (batch_size, 1))
        
        with torch.no_grad():
            output2 = llava_model(
                pixel_values=None,
                input_ids=new_token,
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
    
    def test_generate_with_image(self, llava_model, sample_inputs):
        """测试带图像的生成"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        batch_size = input_ids.shape[0]
        
        with torch.no_grad():
            generated_ids = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5
            )
        
        # 输出应该是 2D 张量
        assert generated_ids.dim() == 2
        # 批次大小应该匹配
        assert generated_ids.shape[0] == batch_size
        # 长度应该大于输入长度
        assert generated_ids.shape[1] > input_ids.shape[1]
    
    def test_generate_greedy(self, llava_model, sample_inputs):
        """测试贪婪解码"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        with torch.no_grad():
            generated_ids_1 = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=False
            )
            generated_ids_2 = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=False
            )
        
        # 贪婪解码应该是确定性的
        assert torch.equal(generated_ids_1, generated_ids_2)
    
    def test_generate_with_sampling(self, llava_model, sample_inputs):
        """测试采样生成"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=1.0
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
        assert generated_ids.shape[1] > input_ids.shape[1]
    
    def test_generate_with_top_k(self, llava_model, sample_inputs):
        """测试 Top-K 采样"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=True,
                top_k=50
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
    
    def test_generate_with_top_p(self, llava_model, sample_inputs):
        """测试 Top-P 采样"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=True,
                top_p=0.9
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]
    
    def test_generate_with_temperature(self, llava_model, sample_inputs):
        """测试温度参数"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = llava_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=5,
                do_sample=True,
                temperature=0.5
            )
        
        assert generated_ids.shape[0] == input_ids.shape[0]


# ============================================================================
# 测试冻结/解冻功能（两阶段训练）(需求 9.6)
# ============================================================================

class TestFreezeUnfreeze:
    """测试冻结/解冻功能"""
    
    def test_vision_encoder_frozen_by_default(self, llava_model):
        """需求 9.6: Vision Encoder 默认应该被冻结"""
        for param in llava_model.vision_encoder.parameters():
            assert not param.requires_grad, "Vision Encoder 参数应该被冻结"
    
    def test_llm_not_frozen_by_default(self, llava_model):
        """测试 LLM 默认不被冻结"""
        # 检查 LLM 参数是否可训练
        trainable_params = sum(
            1 for param in llava_model.llm.parameters() 
            if param.requires_grad
        )
        assert trainable_params > 0, "LLM 应该有可训练参数"
    
    def test_visual_projection_trainable(self, llava_model):
        """需求 9.6: Visual Projection 应该是可训练的"""
        for param in llava_model.visual_projection.parameters():
            assert param.requires_grad, "Visual Projection 参数应该是可训练的"
    
    def test_unfreeze_vision_encoder(self, llava_model):
        """测试解冻视觉编码器"""
        llava_model.unfreeze_vision_encoder()
        
        for param in llava_model.vision_encoder.parameters():
            assert param.requires_grad, "解冻后 Vision Encoder 参数应该是可训练的"
    
    def test_freeze_llm(self, small_vision_config, small_llm_config):
        """测试冻结 LLM"""
        config = LLaVAConfig(
            vision_config=small_vision_config,
            llm_config=small_llm_config,
            projection_type='mlp',
            freeze_vision=True,
            freeze_llm=True  # 冻结 LLM
        )
        model = LLaVAModel(config)
        
        for param in model.llm.parameters():
            assert not param.requires_grad, "LLM 参数应该被冻结"
    
    def test_unfreeze_llm(self, small_vision_config, small_llm_config):
        """测试解冻 LLM"""
        config = LLaVAConfig(
            vision_config=small_vision_config,
            llm_config=small_llm_config,
            projection_type='mlp',
            freeze_vision=True,
            freeze_llm=True
        )
        model = LLaVAModel(config)
        
        # 解冻 LLM
        model.unfreeze_llm()
        
        for param in model.llm.parameters():
            assert param.requires_grad, "解冻后 LLM 参数应该是可训练的"
    
    def test_stage1_training_config(self, small_vision_config, small_llm_config):
        """需求 9.6: 测试第一阶段训练配置（仅训练 Visual Projection）"""
        config = LLaVAConfig(
            vision_config=small_vision_config,
            llm_config=small_llm_config,
            projection_type='mlp',
            freeze_vision=True,
            freeze_llm=True  # 第一阶段冻结 LLM
        )
        model = LLaVAModel(config)
        
        # 只有 Visual Projection 应该是可训练的
        trainable_modules = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_modules.append(name)
        
        # 所有可训练参数应该来自 visual_projection
        for name in trainable_modules:
            assert 'visual_projection' in name, \
                f"第一阶段只有 visual_projection 应该是可训练的，但发现 {name}"
    
    def test_stage2_training_config(self, small_vision_config, small_llm_config):
        """需求 9.6: 测试第二阶段训练配置（全参数微调）"""
        config = LLaVAConfig(
            vision_config=small_vision_config,
            llm_config=small_llm_config,
            projection_type='mlp',
            freeze_vision=False,  # 第二阶段解冻 Vision
            freeze_llm=False      # 第二阶段解冻 LLM
        )
        model = LLaVAModel(config)
        
        # 所有参数应该是可训练的
        for param in model.parameters():
            assert param.requires_grad, "第二阶段所有参数应该是可训练的"


# ============================================================================
# 测试多轮对话输入准备 (需求 9.8)
# ============================================================================

class TestMultiTurnConversation:
    """测试多轮对话输入准备"""
    
    def test_prepare_conversation_input_basic(self, llava_model):
        """需求 9.8: 测试基本多轮对话输入准备"""
        # 创建简单的 tokenizer mock
        class SimpleTokenizer:
            def encode(self, text):
                # 简单地将每个字符映射为一个 token
                return [ord(c) % 100 for c in text]
        
        tokenizer = SimpleTokenizer()
        
        conversations = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there'}
        ]
        
        input_ids, labels = llava_model.prepare_conversation_input(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        assert input_ids.dim() == 2
        assert labels.dim() == 2
        assert input_ids.shape == labels.shape
    
    def test_prepare_conversation_with_system_prompt(self, llava_model):
        """测试带系统提示的对话输入准备"""
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text]
        
        tokenizer = SimpleTokenizer()
        
        conversations = [
            {'role': 'user', 'content': 'What is this?'},
            {'role': 'assistant', 'content': 'This is an image.'}
        ]
        
        input_ids, labels = llava_model.prepare_conversation_input(
            conversations=conversations,
            tokenizer=tokenizer,
            system_prompt="You are a helpful assistant."
        )
        
        assert input_ids.dim() == 2
        assert labels.dim() == 2
    
    def test_prepare_multi_turn_conversation(self, llava_model):
        """测试多轮对话"""
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text]
        
        tokenizer = SimpleTokenizer()
        
        conversations = [
            {'role': 'user', 'content': 'What is in this image?'},
            {'role': 'assistant', 'content': 'I see a cat.'},
            {'role': 'user', 'content': 'What color is it?'},
            {'role': 'assistant', 'content': 'It is orange.'}
        ]
        
        input_ids, labels = llava_model.prepare_conversation_input(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        assert input_ids.dim() == 2
        assert labels.dim() == 2
    
    def test_user_parts_masked_in_labels(self, llava_model):
        """需求 9.7: 测试 user 部分在 labels 中被掩码"""
        class SimpleTokenizer:
            def encode(self, text):
                return [ord(c) % 100 for c in text]
        
        tokenizer = SimpleTokenizer()
        
        conversations = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello'}
        ]
        
        input_ids, labels = llava_model.prepare_conversation_input(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        # 检查是否有 -100 的掩码值（user 部分）
        has_masked = (labels == -100).any()
        assert has_masked, "User 部分应该被掩码为 -100"


# ============================================================================
# 测试不同批次大小和序列长度
# ============================================================================

class TestDifferentBatchSizesAndSequenceLengths:
    """测试不同批次大小和序列长度"""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_different_batch_sizes(self, small_llava_config, batch_size):
        """测试不同批次大小的前向传播"""
        model = LLaVAModel(small_llava_config)
        model.eval()
        
        image_size = small_llava_config.vision_config.image_size
        seq_len = 16
        vocab_size = small_llava_config.llm_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids[:, 4] = DEFAULT_IMAGE_TOKEN_ID
        
        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids)
        
        assert output['logits'].shape[0] == batch_size
    
    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_forward_different_sequence_lengths(self, small_llava_config, seq_len):
        """测试不同序列长度的前向传播"""
        model = LLaVAModel(small_llava_config)
        model.eval()
        
        batch_size = 2
        image_size = small_llava_config.vision_config.image_size
        vocab_size = small_llava_config.llm_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids[:, min(4, seq_len - 1)] = DEFAULT_IMAGE_TOKEN_ID
        
        with torch.no_grad():
            output = model(pixel_values=pixel_values, input_ids=input_ids)
        
        # 输出序列长度应该正确
        num_patches = (image_size // small_llava_config.vision_config.patch_size) ** 2
        expected_seq_len = seq_len - 1 + num_patches
        assert output['logits'].shape[1] == expected_seq_len
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_generate_different_batch_sizes(self, small_llava_config, batch_size):
        """测试不同批次大小的生成"""
        model = LLaVAModel(small_llava_config)
        model.eval()
        
        image_size = small_llava_config.vision_config.image_size
        seq_len = 8
        vocab_size = small_llava_config.llm_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        input_ids[:, 4] = DEFAULT_IMAGE_TOKEN_ID
        
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                max_new_tokens=3
            )
        
        assert generated_ids.shape[0] == batch_size


# ============================================================================
# 测试梯度流
# ============================================================================

class TestGradientFlow:
    """测试梯度流"""
    
    def test_gradient_flow_with_images(self, llava_model, sample_inputs):
        """测试带图像的梯度流"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        labels = sample_inputs['labels']
        
        llava_model.train()
        
        output = llava_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
        
        loss = output['loss']
        loss.backward()
        
        # Visual Projection 应该有梯度
        for param in llava_model.visual_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Visual Projection 参数应该有梯度"
        
        # Vision Encoder 不应该有梯度（被冻结）
        for param in llava_model.vision_encoder.parameters():
            assert param.grad is None, "Vision Encoder 参数不应该有梯度"
        
        llava_model.zero_grad()
    
    def test_gradient_flow_without_images(self, llava_model, sample_inputs):
        """测试纯文本的梯度流"""
        input_ids = sample_inputs['input_ids_text_only']
        labels = sample_inputs['labels']
        
        llava_model.train()
        
        output = llava_model(
            pixel_values=None,
            input_ids=input_ids,
            labels=labels
        )
        
        loss = output['loss']
        loss.backward()
        
        # LLM 应该有梯度
        has_grad = False
        for param in llava_model.llm.parameters():
            if param.requires_grad and param.grad is not None:
                has_grad = True
                break
        
        assert has_grad, "LLM 参数应该有梯度"
        
        llava_model.zero_grad()


# ============================================================================
# 测试输出隐藏状态
# ============================================================================

class TestOutputHiddenStates:
    """测试输出隐藏状态"""
    
    def test_output_hidden_states(self, llava_model, sample_inputs):
        """测试输出所有层的隐藏状态"""
        input_ids = sample_inputs['input_ids_text_only']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                output_hidden_states=True
            )
        
        assert 'all_hidden_states' in output
        all_hidden_states = output['all_hidden_states']
        
        # 应该有 num_layers + 1 个隐藏状态（包括输入嵌入）
        num_layers = llava_model.config.llm_config.num_layers
        assert len(all_hidden_states) == num_layers + 1
    
    def test_hidden_states_shape(self, llava_model, sample_inputs):
        """测试隐藏状态形状"""
        input_ids = sample_inputs['input_ids_text_only']
        batch_size, seq_len = input_ids.shape
        d_model = llava_model.config.llm_config.d_model
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids,
                output_hidden_states=True
            )
        
        all_hidden_states = output['all_hidden_states']
        
        for hidden_state in all_hidden_states:
            assert hidden_state.shape == (batch_size, seq_len, d_model)
    
    def test_output_hidden_states_with_image(self, llava_model, sample_inputs):
        """测试带图像时输出隐藏状态"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                output_hidden_states=True
            )
        
        assert 'all_hidden_states' in output
        all_hidden_states = output['all_hidden_states']
        
        num_layers = llava_model.config.llm_config.num_layers
        assert len(all_hidden_states) == num_layers + 1


# ============================================================================
# 测试 prepare_inputs_for_generation
# ============================================================================

class TestPrepareInputsForGeneration:
    """测试 prepare_inputs_for_generation 方法"""
    
    def test_prepare_inputs_first_call(self, llava_model, sample_inputs):
        """测试第一次调用（无 KV Cache）"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        model_inputs = llava_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=None
        )
        
        # 第一次调用应该包含完整输入
        assert torch.equal(model_inputs['input_ids'], input_ids)
        assert torch.equal(model_inputs['pixel_values'], pixel_values)
        assert model_inputs['past_key_values'] is None
        assert model_inputs['use_cache'] == True
    
    def test_prepare_inputs_subsequent_call(self, llava_model, sample_inputs):
        """测试后续调用（有 KV Cache）"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_with_image']
        
        # 模拟 KV Cache
        batch_size = input_ids.shape[0]
        num_layers = llava_model.config.llm_config.num_layers
        num_kv_heads = llava_model.config.llm_config.num_kv_heads
        head_dim = llava_model.config.llm_config.d_model // llava_model.config.llm_config.num_heads
        seq_len = 10
        
        past_key_values = [
            (
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim),
                torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
            )
            for _ in range(num_layers)
        ]
        
        model_inputs = llava_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values
        )
        
        # 后续调用应该只包含最后一个 token
        assert model_inputs['input_ids'].shape[1] == 1
        # 图像应该为 None（已经在第一次调用时处理）
        assert model_inputs['pixel_values'] is None
        assert model_inputs['past_key_values'] is not None


# ============================================================================
# 测试 extra_repr
# ============================================================================

class TestExtraRepr:
    """测试 extra_repr 方法"""
    
    def test_extra_repr(self, llava_model):
        """测试模型的额外表示信息"""
        repr_str = llava_model.extra_repr()
        
        assert 'vision_dim' in repr_str
        assert 'llm_dim' in repr_str
        assert 'projection_type' in repr_str
        assert 'freeze_vision' in repr_str
        assert 'freeze_llm' in repr_str


# ============================================================================
# 测试边界情况
# ============================================================================

class TestEdgeCases:
    """测试边界情况"""
    
    def test_image_token_at_start(self, llava_model, sample_inputs):
        """测试 <image> token 在序列开头"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_text_only'].clone()
        input_ids[:, 0] = DEFAULT_IMAGE_TOKEN_ID
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        assert 'logits' in output
    
    def test_image_token_at_end(self, llava_model, sample_inputs):
        """测试 <image> token 在序列末尾"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_text_only'].clone()
        input_ids[:, -1] = DEFAULT_IMAGE_TOKEN_ID
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        assert 'logits' in output
    
    def test_single_token_input(self, llava_model, sample_inputs):
        """测试单 token 输入"""
        pixel_values = sample_inputs['pixel_values']
        batch_size = pixel_values.shape[0]
        vocab_size = llava_model.config.llm_config.vocab_size
        
        input_ids = torch.randint(0, vocab_size, (batch_size, 1))
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=None,
                input_ids=input_ids
            )
        
        assert output['logits'].shape[0] == batch_size
        assert output['logits'].shape[1] == 1
    
    def test_custom_image_token_index(self, llava_model, sample_inputs):
        """测试自定义 <image> token ID"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids_text_only'].clone()
        
        custom_image_token_id = -999
        input_ids[:, 4] = custom_image_token_id
        
        with torch.no_grad():
            output = llava_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_token_index=custom_image_token_id
            )
        
        assert 'logits' in output
