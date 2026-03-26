"""
BLIP-2 模型单元测试

测试 BLIP-2 模型的各个组件和功能：
- 模型初始化和组件组合
- 冻结的 Vision Encoder
- Q-Former 集成
- Visual Projection
- 冻结的 LLM
- 两阶段训练前向传播
- 生成功能

需求: 6.1, 6.2, 6.6, 6.7, 6.8
"""

import pytest
import torch
import torch.nn as nn

from bert_gpt_from_scratch.config import TransformerConfig
from multimodal_models_from_scratch.config import (
    BLIP2Config,
    VisionConfig,
    LLaMAConfig
)
from multimodal_models_from_scratch.multimodal.blip2 import BLIP2Model


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
def small_qformer_config():
    """小型 Q-Former 配置"""
    return TransformerConfig(
        vocab_size=1000,
        d_model=64,
        num_heads=4,
        num_layers=2,
        d_ff=128,
        max_seq_len=64,
        dropout_rate=0.1
    )


@pytest.fixture
def small_llm_config():
    """小型 LLM 配置"""
    return LLaMAConfig(
        vocab_size=1000,
        d_model=128,
        num_heads=4,
        num_kv_heads=2,
        num_layers=2,
        d_ff=256,
        max_seq_len=64,
        dropout_rate=0.0,
        rope_theta=10000.0,
        tie_weights=False
    )


@pytest.fixture
def small_blip2_config(small_vision_config, small_qformer_config, small_llm_config):
    """小型 BLIP-2 配置"""
    return BLIP2Config(
        vision_config=small_vision_config,
        qformer_config=small_qformer_config,
        llm_config=small_llm_config,
        num_query_tokens=8,
        projection_dim=64
    )


@pytest.fixture
def blip2_model(small_blip2_config):
    """创建 BLIP-2 模型实例"""
    model = BLIP2Model(small_blip2_config)
    model.eval()
    return model


@pytest.fixture
def sample_inputs(small_blip2_config):
    """创建示例输入"""
    batch_size = 2
    image_size = small_blip2_config.vision_config.image_size
    seq_len = 16
    vocab_size = small_blip2_config.qformer_config.vocab_size
    
    pixel_values = torch.randn(batch_size, 3, image_size, image_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }


# ============================================================================
# 测试模型初始化和组件组合
# ============================================================================

class TestBLIP2ModelInitialization:
    """测试 BLIP-2 模型初始化"""
    
    def test_model_creation(self, small_blip2_config):
        """测试模型可以正常创建"""
        model = BLIP2Model(small_blip2_config)
        assert model is not None
    
    def test_model_components_exist(self, blip2_model):
        """测试模型包含所有必要组件"""
        # 需求 6.1: 冻结的 Vision Encoder
        assert hasattr(blip2_model, 'vision_encoder')
        
        # 需求 6.2: Q-Former
        assert hasattr(blip2_model, 'qformer')
        
        # 需求 6.6: Visual Projection
        assert hasattr(blip2_model, 'visual_projection')
        
        # 需求 6.7: 冻结的 LLM
        assert hasattr(blip2_model, 'llm')
        
        # 第一阶段训练组件
        assert hasattr(blip2_model, 'vision_proj_itc')
        assert hasattr(blip2_model, 'text_proj_itc')
        assert hasattr(blip2_model, 'logit_scale')
        assert hasattr(blip2_model, 'itm_head')
        assert hasattr(blip2_model, 'itg_head')
        assert hasattr(blip2_model, 'text_embedding')
    
    def test_config_stored(self, blip2_model, small_blip2_config):
        """测试配置被正确存储"""
        assert blip2_model.config == small_blip2_config
        assert blip2_model.config.num_query_tokens == 8
        assert blip2_model.config.projection_dim == 64


# ============================================================================
# 测试冻结的 Vision Encoder (需求 6.1)
# ============================================================================

class TestFrozenVisionEncoder:
    """测试冻结的 Vision Encoder"""
    
    def test_vision_encoder_frozen(self, blip2_model):
        """需求 6.1: Vision Encoder 应该被冻结"""
        for param in blip2_model.vision_encoder.parameters():
            assert not param.requires_grad, "Vision Encoder 参数应该被冻结"
    
    def test_vision_encoder_output_shape(self, blip2_model, sample_inputs):
        """测试 Vision Encoder 输出形状"""
        pixel_values = sample_inputs['pixel_values']
        
        # 编码图像
        image_features = blip2_model.encode_image(pixel_values)
        
        batch_size = pixel_values.shape[0]
        # 计算 patch 数量: (image_size / patch_size)^2
        num_patches = (64 // 16) ** 2  # = 16
        vision_d_model = blip2_model.config.vision_config.d_model
        
        assert image_features.shape == (batch_size, num_patches, vision_d_model)
    
    def test_vision_encoder_no_grad(self, blip2_model, sample_inputs):
        """测试 Vision Encoder 在前向传播时不计算梯度"""
        pixel_values = sample_inputs['pixel_values']
        pixel_values.requires_grad = True
        
        with torch.no_grad():
            image_features = blip2_model.encode_image(pixel_values)
        
        # 输出不应该有梯度
        assert not image_features.requires_grad


# ============================================================================
# 测试 Q-Former 集成 (需求 6.2)
# ============================================================================

class TestQFormerIntegration:
    """测试 Q-Former 集成"""
    
    def test_qformer_num_query_tokens(self, blip2_model, small_blip2_config):
        """需求 6.2: Q-Former 应该有正确数量的查询 token"""
        expected_num_tokens = small_blip2_config.num_query_tokens
        actual_num_tokens = blip2_model.qformer.num_query_tokens
        assert actual_num_tokens == expected_num_tokens
    
    def test_qformer_query_tokens_shape(self, blip2_model, small_blip2_config):
        """测试 Q-Former 查询 token 形状"""
        query_tokens = blip2_model.qformer.get_query_tokens()
        
        num_query_tokens = small_blip2_config.num_query_tokens
        d_model = small_blip2_config.qformer_config.d_model
        
        assert query_tokens.shape == (1, num_query_tokens, d_model)
    
    def test_qformer_output_shape(self, blip2_model, sample_inputs):
        """测试 Q-Former 输出形状"""
        pixel_values = sample_inputs['pixel_values']
        batch_size = pixel_values.shape[0]
        
        # 编码图像
        image_features = blip2_model.encode_image(pixel_values)
        
        # Q-Former 处理
        query_output = blip2_model.qformer(image_features)
        
        num_query_tokens = blip2_model.config.num_query_tokens
        qformer_d_model = blip2_model.config.qformer_config.d_model
        
        assert query_output.shape == (batch_size, num_query_tokens, qformer_d_model)
    
    def test_qformer_trainable(self, blip2_model):
        """测试 Q-Former 参数是可训练的"""
        for param in blip2_model.qformer.parameters():
            assert param.requires_grad, "Q-Former 参数应该是可训练的"


# ============================================================================
# 测试 Visual Projection (需求 6.6)
# ============================================================================

class TestVisualProjection:
    """测试 Visual Projection"""
    
    def test_visual_projection_exists(self, blip2_model):
        """需求 6.6: 应该有 Visual Projection 层"""
        assert blip2_model.visual_projection is not None
    
    def test_visual_projection_dimensions(self, blip2_model, small_blip2_config):
        """测试 Visual Projection 维度"""
        qformer_d_model = small_blip2_config.qformer_config.d_model
        llm_d_model = small_blip2_config.llm_config.d_model
        
        assert blip2_model.visual_projection.vision_dim == qformer_d_model
        assert blip2_model.visual_projection.llm_dim == llm_d_model
    
    def test_visual_projection_output_shape(self, blip2_model, sample_inputs):
        """测试 Visual Projection 输出形状"""
        pixel_values = sample_inputs['pixel_values']
        batch_size = pixel_values.shape[0]
        
        # 编码图像
        image_features = blip2_model.encode_image(pixel_values)
        
        # Q-Former 处理
        query_output = blip2_model.qformer(image_features)
        
        # Visual Projection
        visual_embeds = blip2_model.visual_projection(query_output)
        
        num_query_tokens = blip2_model.config.num_query_tokens
        llm_d_model = blip2_model.config.llm_config.d_model
        
        assert visual_embeds.shape == (batch_size, num_query_tokens, llm_d_model)
    
    def test_visual_projection_trainable(self, blip2_model):
        """测试 Visual Projection 参数是可训练的"""
        for param in blip2_model.visual_projection.parameters():
            assert param.requires_grad, "Visual Projection 参数应该是可训练的"


# ============================================================================
# 测试冻结的 LLM (需求 6.7)
# ============================================================================

class TestFrozenLLM:
    """测试冻结的 LLM"""
    
    def test_llm_frozen(self, blip2_model):
        """需求 6.7: LLM 应该被冻结"""
        for param in blip2_model.llm.parameters():
            assert not param.requires_grad, "LLM 参数应该被冻结"
    
    def test_llm_is_llama(self, blip2_model):
        """测试 LLM 是 LLaMA 模型"""
        from multimodal_models_from_scratch.llm.llama import LLaMAModel
        assert isinstance(blip2_model.llm, LLaMAModel)
    
    def test_unfreeze_llm(self, blip2_model):
        """测试可以解冻 LLM"""
        # 解冻
        blip2_model.unfreeze_llm()
        
        # 检查参数是否可训练
        for param in blip2_model.llm.parameters():
            assert param.requires_grad, "解冻后 LLM 参数应该是可训练的"
        
        # 重新冻结
        blip2_model._freeze_llm()
        
        for param in blip2_model.llm.parameters():
            assert not param.requires_grad, "重新冻结后 LLM 参数应该被冻结"


# ============================================================================
# 测试两阶段训练 (需求 6.8)
# ============================================================================

class TestTwoStageTraining:
    """测试两阶段训练"""
    
    def test_stage1_forward(self, blip2_model, sample_inputs):
        """需求 6.8: 测试第一阶段前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        attention_mask = sample_inputs['attention_mask']
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 检查输出包含 ITC 相关内容
        assert 'itc_loss' in output
        assert 'image_embeds' in output
        assert 'text_embeds' in output
        assert 'logits_per_image' in output
        assert 'logits_per_text' in output
        
        # 检查输出包含 ITM 相关内容
        assert 'itm_logits' in output
    
    def test_stage1_itc_loss(self, blip2_model, sample_inputs):
        """测试第一阶段 ITC 损失"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        itc_loss = output['itc_loss']
        
        # 损失应该是标量
        assert itc_loss.dim() == 0
        # 损失应该是正数
        assert itc_loss.item() > 0
    
    def test_stage1_itm_loss(self, blip2_model, sample_inputs):
        """测试第一阶段 ITM 损失"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        batch_size = pixel_values.shape[0]
        
        # 创建 ITM 标签
        itm_labels = torch.randint(0, 2, (batch_size,))
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids,
            itm_labels=itm_labels
        )
        
        assert 'itm_loss' in output
        itm_loss = output['itm_loss']
        
        # 损失应该是标量
        assert itm_loss.dim() == 0
        # 损失应该是正数
        assert itm_loss.item() > 0

    
    def test_stage1_itg_loss(self, blip2_model, sample_inputs):
        """测试第一阶段 ITG 损失"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids,
            itg_labels=labels
        )
        
        assert 'itg_loss' in output
        itg_loss = output['itg_loss']
        
        # 损失应该是标量
        assert itg_loss.dim() == 0
        # 损失应该是正数
        assert itg_loss.item() > 0
    
    def test_stage2_forward(self, blip2_model, sample_inputs):
        """需求 6.8: 测试第二阶段前向传播"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        attention_mask = sample_inputs['attention_mask']
        
        lm_loss = blip2_model.forward_stage2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask
        )
        
        # 损失应该是标量
        assert lm_loss.dim() == 0
        # 损失应该是正数
        assert lm_loss.item() > 0
    
    def test_stage2_loss_computation(self, blip2_model, sample_inputs):
        """测试第二阶段损失计算"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        # 多次运行应该得到相同的损失（确定性）
        blip2_model.eval()
        with torch.no_grad():
            loss1 = blip2_model.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
            loss2 = blip2_model.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
        
        assert torch.allclose(loss1, loss2)


# ============================================================================
# 测试可训练参数
# ============================================================================

class TestTrainableParameters:
    """测试可训练参数"""
    
    def test_get_trainable_params_stage1(self, blip2_model):
        """测试获取第一阶段可训练参数"""
        params = blip2_model.get_trainable_params_stage1()
        
        # 应该有参数
        assert len(params) > 0
        
        # 所有参数应该是可训练的
        for param in params:
            assert param.requires_grad
    
    def test_get_trainable_params_stage2(self, blip2_model):
        """测试获取第二阶段可训练参数"""
        params = blip2_model.get_trainable_params_stage2()
        
        # 应该有参数
        assert len(params) > 0
        
        # 所有参数应该是可训练的
        for param in params:
            assert param.requires_grad
    
    def test_stage1_params_include_qformer(self, blip2_model):
        """测试第一阶段参数包含 Q-Former"""
        stage1_params = set(blip2_model.get_trainable_params_stage1())
        qformer_params = set(blip2_model.qformer.parameters())
        
        # Q-Former 参数应该在第一阶段参数中
        assert qformer_params.issubset(stage1_params)
    
    def test_stage2_params_include_visual_projection(self, blip2_model):
        """测试第二阶段参数包含 Visual Projection"""
        stage2_params = set(blip2_model.get_trainable_params_stage2())
        visual_proj_params = set(blip2_model.visual_projection.parameters())
        
        # Visual Projection 参数应该在第二阶段参数中
        assert visual_proj_params.issubset(stage2_params)


# ============================================================================
# 测试生成功能
# ============================================================================

class TestGeneration:
    """测试生成功能"""
    
    def test_generate_basic(self, blip2_model, sample_inputs):
        """测试基本生成功能"""
        pixel_values = sample_inputs['pixel_values']
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10
            )
        
        batch_size = pixel_values.shape[0]
        
        # 输出应该是 2D 张量
        assert generated_ids.dim() == 2
        # 批次大小应该匹配
        assert generated_ids.shape[0] == batch_size
        # 长度应该大于 0
        assert generated_ids.shape[1] > 0
    
    def test_generate_with_prompt(self, blip2_model, sample_inputs):
        """测试带提示的生成"""
        pixel_values = sample_inputs['pixel_values']
        batch_size = pixel_values.shape[0]
        
        # 创建提示
        prompt_ids = torch.randint(0, 100, (batch_size, 5))
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                prompt_ids=prompt_ids,
                max_length=15
            )
        
        # 生成的序列应该以提示开头
        assert generated_ids.shape[1] >= prompt_ids.shape[1]
        assert torch.equal(generated_ids[:, :prompt_ids.shape[1]], prompt_ids)
    
    def test_generate_greedy(self, blip2_model, sample_inputs):
        """测试贪婪解码"""
        pixel_values = sample_inputs['pixel_values']
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10,
                do_sample=False
            )
        
        # 贪婪解码应该是确定性的
        with torch.no_grad():
            generated_ids_2 = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10,
                do_sample=False
            )
        
        assert torch.equal(generated_ids, generated_ids_2)

    
    def test_generate_with_sampling(self, blip2_model, sample_inputs):
        """测试采样生成"""
        pixel_values = sample_inputs['pixel_values']
        
        # 设置随机种子以获得可重复的结果
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10,
                do_sample=True,
                temperature=1.0
            )
        
        # 输出应该是有效的
        assert generated_ids.shape[0] == pixel_values.shape[0]
        assert generated_ids.shape[1] > 0
    
    def test_generate_with_top_k(self, blip2_model, sample_inputs):
        """测试 Top-K 采样"""
        pixel_values = sample_inputs['pixel_values']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10,
                do_sample=True,
                top_k=50
            )
        
        assert generated_ids.shape[0] == pixel_values.shape[0]
    
    def test_generate_with_top_p(self, blip2_model, sample_inputs):
        """测试 Top-P 采样"""
        pixel_values = sample_inputs['pixel_values']
        
        torch.manual_seed(42)
        
        with torch.no_grad():
            generated_ids = blip2_model.generate(
                pixel_values=pixel_values,
                max_length=10,
                do_sample=True,
                top_p=0.9
            )
        
        assert generated_ids.shape[0] == pixel_values.shape[0]


# ============================================================================
# 测试前向传播接口
# ============================================================================

class TestForwardInterface:
    """测试前向传播接口"""
    
    def test_forward_stage1(self, blip2_model, sample_inputs):
        """测试 forward 方法 stage=1"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        output = blip2_model.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            stage=1
        )
        
        # 应该包含 ITC 损失
        assert 'itc_loss' in output
    
    def test_forward_stage2(self, blip2_model, sample_inputs):
        """测试 forward 方法 stage=2"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        output = blip2_model.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            stage=2
        )
        
        # 应该包含损失
        assert 'loss' in output
        assert output['loss'].dim() == 0
    
    def test_forward_default_stage(self, blip2_model, sample_inputs):
        """测试 forward 方法默认阶段"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        # 默认应该是 stage=2
        output = blip2_model.forward(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
        
        assert 'loss' in output


# ============================================================================
# 测试输出形状
# ============================================================================

class TestOutputShapes:
    """测试输出形状"""
    
    def test_stage1_image_embeds_shape(self, blip2_model, sample_inputs):
        """测试第一阶段图像嵌入形状"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        batch_size = pixel_values.shape[0]
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        projection_dim = blip2_model.config.projection_dim
        assert output['image_embeds'].shape == (batch_size, projection_dim)
    
    def test_stage1_text_embeds_shape(self, blip2_model, sample_inputs):
        """测试第一阶段文本嵌入形状"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        batch_size = pixel_values.shape[0]
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        projection_dim = blip2_model.config.projection_dim
        assert output['text_embeds'].shape == (batch_size, projection_dim)
    
    def test_stage1_logits_shape(self, blip2_model, sample_inputs):
        """测试第一阶段相似度矩阵形状"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        batch_size = pixel_values.shape[0]
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        assert output['logits_per_image'].shape == (batch_size, batch_size)
        assert output['logits_per_text'].shape == (batch_size, batch_size)
    
    def test_stage1_itm_logits_shape(self, blip2_model, sample_inputs):
        """测试第一阶段 ITM logits 形状"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        batch_size = pixel_values.shape[0]
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        assert output['itm_logits'].shape == (batch_size, 2)


# ============================================================================
# 测试嵌入归一化
# ============================================================================

class TestEmbeddingNormalization:
    """测试嵌入归一化"""
    
    def test_image_embeds_normalized(self, blip2_model, sample_inputs):
        """测试图像嵌入是 L2 归一化的"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        image_embeds = output['image_embeds']
        
        # 计算 L2 范数
        norms = torch.norm(image_embeds, p=2, dim=-1)
        
        # 范数应该接近 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_text_embeds_normalized(self, blip2_model, sample_inputs):
        """测试文本嵌入是 L2 归一化的"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        text_embeds = output['text_embeds']
        
        # 计算 L2 范数
        norms = torch.norm(text_embeds, p=2, dim=-1)
        
        # 范数应该接近 1
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ============================================================================
# 测试梯度流
# ============================================================================

class TestGradientFlow:
    """测试梯度流"""
    
    def test_stage1_gradient_flow(self, blip2_model, sample_inputs):
        """测试第一阶段梯度流"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        
        blip2_model.train()
        
        output = blip2_model.forward_stage1(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        
        loss = output['itc_loss']
        loss.backward()
        
        # Q-Former 应该有梯度
        for param in blip2_model.qformer.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Q-Former 参数应该有梯度"
        
        # Vision Encoder 不应该有梯度
        for param in blip2_model.vision_encoder.parameters():
            assert param.grad is None, "Vision Encoder 参数不应该有梯度"
        
        blip2_model.zero_grad()
    
    def test_stage2_gradient_flow(self, blip2_model, sample_inputs):
        """测试第二阶段梯度流"""
        pixel_values = sample_inputs['pixel_values']
        input_ids = sample_inputs['input_ids']
        labels = sample_inputs['labels']
        
        blip2_model.train()
        
        loss = blip2_model.forward_stage2(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels
        )
        
        loss.backward()
        
        # Visual Projection 应该有梯度
        for param in blip2_model.visual_projection.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Visual Projection 参数应该有梯度"
        
        # LLM 不应该有梯度
        for param in blip2_model.llm.parameters():
            assert param.grad is None, "LLM 参数不应该有梯度"
        
        blip2_model.zero_grad()


# ============================================================================
# 测试模型表示
# ============================================================================

class TestModelRepresentation:
    """测试模型表示"""
    
    def test_extra_repr(self, blip2_model):
        """测试 extra_repr 方法"""
        repr_str = blip2_model.extra_repr()
        
        assert 'num_query_tokens' in repr_str
        assert 'projection_dim' in repr_str
    
    def test_model_str(self, blip2_model):
        """测试模型字符串表示"""
        model_str = str(blip2_model)
        
        # 应该包含主要组件
        assert 'vision_encoder' in model_str
        assert 'qformer' in model_str
        assert 'visual_projection' in model_str
        assert 'llm' in model_str


# ============================================================================
# 测试不同批次大小
# ============================================================================

class TestDifferentBatchSizes:
    """测试不同批次大小"""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_stage1_different_batch_sizes(self, small_blip2_config, batch_size):
        """测试第一阶段不同批次大小"""
        model = BLIP2Model(small_blip2_config)
        model.eval()
        
        image_size = small_blip2_config.vision_config.image_size
        seq_len = 16
        vocab_size = small_blip2_config.qformer_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model.forward_stage1(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        assert output['image_embeds'].shape[0] == batch_size
        assert output['text_embeds'].shape[0] == batch_size
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_stage2_different_batch_sizes(self, small_blip2_config, batch_size):
        """测试第二阶段不同批次大小"""
        model = BLIP2Model(small_blip2_config)
        model.eval()
        
        image_size = small_blip2_config.vision_config.image_size
        seq_len = 16
        vocab_size = small_blip2_config.qformer_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            loss = model.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
        
        assert loss.dim() == 0


# ============================================================================
# 测试不同序列长度
# ============================================================================

class TestDifferentSequenceLengths:
    """测试不同序列长度"""
    
    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_stage1_different_seq_lengths(self, small_blip2_config, seq_len):
        """测试第一阶段不同序列长度"""
        model = BLIP2Model(small_blip2_config)
        model.eval()
        
        batch_size = 2
        image_size = small_blip2_config.vision_config.image_size
        vocab_size = small_blip2_config.qformer_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            output = model.forward_stage1(
                pixel_values=pixel_values,
                input_ids=input_ids
            )
        
        # 输出形状不应该依赖于序列长度
        projection_dim = small_blip2_config.projection_dim
        assert output['image_embeds'].shape == (batch_size, projection_dim)
        assert output['text_embeds'].shape == (batch_size, projection_dim)
    
    @pytest.mark.parametrize("seq_len", [8, 16, 32])
    def test_stage2_different_seq_lengths(self, small_blip2_config, seq_len):
        """测试第二阶段不同序列长度"""
        model = BLIP2Model(small_blip2_config)
        model.eval()
        
        batch_size = 2
        image_size = small_blip2_config.vision_config.image_size
        vocab_size = small_blip2_config.qformer_config.vocab_size
        
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        with torch.no_grad():
            loss = model.forward_stage2(
                pixel_values=pixel_values,
                input_ids=input_ids,
                labels=labels
            )
        
        assert loss.dim() == 0
        assert loss.item() > 0
