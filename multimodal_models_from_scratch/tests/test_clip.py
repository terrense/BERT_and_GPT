"""
CLIP 模型单元测试

测试 CLIP 模型的各个组件和功能：
- TextEncoder: 文本编码器
- CLIPModel: 完整的 CLIP 模型
- 图像和文本编码
- 相似度矩阵计算
- L2 归一化
- 可学习的温度参数
- 对比损失函数

需求: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 17.4
"""

import pytest
import torch
import torch.nn as nn

from bert_gpt_from_scratch.config import TransformerConfig
from multimodal_models_from_scratch.config import CLIPConfig, VisionConfig
from multimodal_models_from_scratch.multimodal.clip import (
    TextEncoder,
    CLIPModel,
    contrastive_loss,
)


class TestTextEncoder:
    """TextEncoder 单元测试"""
    
    @pytest.fixture
    def text_config(self):
        """创建测试用的文本配置"""
        return TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout_rate=0.1
        )
    
    @pytest.fixture
    def text_encoder(self, text_config):
        """创建测试用的 TextEncoder"""
        return TextEncoder(text_config)
    
    def test_text_encoder_output_shape(self, text_encoder, text_config):
        """测试 TextEncoder 输出形状"""
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        
        output = text_encoder(input_ids)
        
        # 输出应该是 (batch, d_model)
        assert output.shape == (batch_size, text_config.d_model)
    
    def test_text_encoder_with_attention_mask(self, text_encoder, text_config):
        """测试带 attention_mask 的 TextEncoder"""
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        # 创建不同长度的 attention_mask
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 20:] = 0  # 第一个序列只有 20 个有效 token
        attention_mask[1, 25:] = 0  # 第二个序列只有 25 个有效 token
        
        output = text_encoder(input_ids, attention_mask)
        
        # 输出应该是 (batch, d_model)
        assert output.shape == (batch_size, text_config.d_model)
    
    def test_text_encoder_different_batch_sizes(self, text_encoder, text_config):
        """测试不同 batch size 的 TextEncoder"""
        for batch_size in [1, 2, 8]:
            seq_len = 32
            input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
            
            output = text_encoder(input_ids)
            
            assert output.shape == (batch_size, text_config.d_model)


class TestCLIPModel:
    """CLIPModel 单元测试"""
    
    @pytest.fixture
    def clip_config(self):
        """创建测试用的 CLIP 配置"""
        vision_config = VisionConfig(
            image_size=64,  # 使用较小的图像尺寸加速测试
            patch_size=8,
            in_channels=3,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            dropout_rate=0.1,
            num_classes=0
        )
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout_rate=0.1
        )
        return CLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=64,
            temperature=0.07
        )
    
    @pytest.fixture
    def clip_model(self, clip_config):
        """创建测试用的 CLIPModel"""
        return CLIPModel(clip_config)
    
    def test_clip_model_forward_output_keys(self, clip_model, clip_config):
        """测试 CLIPModel forward 输出包含所有必要的键"""
        batch_size = 4
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        output = clip_model(pixel_values, input_ids)
        
        # 检查输出包含所有必要的键
        assert 'image_embeds' in output
        assert 'text_embeds' in output
        assert 'logits_per_image' in output
        assert 'logits_per_text' in output
        assert 'temperature' in output
    
    def test_clip_model_forward_output_shapes(self, clip_model, clip_config):
        """测试 CLIPModel forward 输出形状"""
        batch_size = 4
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        output = clip_model(pixel_values, input_ids)
        
        # 检查输出形状
        assert output['image_embeds'].shape == (batch_size, clip_config.projection_dim)
        assert output['text_embeds'].shape == (batch_size, clip_config.projection_dim)
        assert output['logits_per_image'].shape == (batch_size, batch_size)
        assert output['logits_per_text'].shape == (batch_size, batch_size)
    
    def test_clip_model_encode_image(self, clip_model, clip_config):
        """测试 encode_image 方法"""
        batch_size = 4
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        
        image_embeds = clip_model.encode_image(pixel_values)
        
        # 检查输出形状
        assert image_embeds.shape == (batch_size, clip_config.projection_dim)
        
        # 检查 L2 归一化（每个向量的范数应该接近 1）
        norms = torch.norm(image_embeds, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_clip_model_encode_text(self, clip_model, clip_config):
        """测试 encode_text 方法"""
        batch_size = 4
        seq_len = 16
        
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, seq_len))
        
        text_embeds = clip_model.encode_text(input_ids)
        
        # 检查输出形状
        assert text_embeds.shape == (batch_size, clip_config.projection_dim)
        
        # 检查 L2 归一化（每个向量的范数应该接近 1）
        norms = torch.norm(text_embeds, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    
    def test_clip_model_l2_normalization(self, clip_model, clip_config):
        """测试 L2 归一化 - 需求 4.5"""
        batch_size = 4
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        output = clip_model(pixel_values, input_ids)
        
        # 检查图像嵌入的 L2 范数
        image_norms = torch.norm(output['image_embeds'], p=2, dim=-1)
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5)
        
        # 检查文本嵌入的 L2 范数
        text_norms = torch.norm(output['text_embeds'], p=2, dim=-1)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5)
    
    def test_clip_model_learnable_temperature(self, clip_model, clip_config):
        """测试可学习的温度参数 - 需求 4.6"""
        # 检查温度参数是可学习的
        assert clip_model.logit_scale.requires_grad
        
        # 检查初始温度值接近配置值
        initial_temp = clip_model.get_temperature().item()
        expected_temp = 1.0 / clip_config.temperature  # log scale 转换后
        assert abs(initial_temp - expected_temp) < 1.0  # 允许一定误差
    
    def test_clip_model_similarity_matrix(self, clip_model, clip_config):
        """测试相似度矩阵计算 - 需求 4.7"""
        batch_size = 4
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        output = clip_model(pixel_values, input_ids)
        
        # 检查相似度矩阵是 batch_size x batch_size
        assert output['logits_per_image'].shape == (batch_size, batch_size)
        assert output['logits_per_text'].shape == (batch_size, batch_size)
        
        # 检查 logits_per_text 是 logits_per_image 的转置
        assert torch.allclose(
            output['logits_per_text'],
            output['logits_per_image'].T,
            atol=1e-5
        )
    
    def test_clip_model_with_attention_mask(self, clip_model, clip_config):
        """测试带 attention_mask 的 CLIPModel"""
        batch_size = 4
        seq_len = 16
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 10:] = 0  # 第一个序列只有 10 个有效 token
        
        output = clip_model(pixel_values, input_ids, attention_mask)
        
        # 检查输出形状
        assert output['image_embeds'].shape == (batch_size, clip_config.projection_dim)
        assert output['text_embeds'].shape == (batch_size, clip_config.projection_dim)
    
    def test_clip_model_gradient_flow(self, clip_model, clip_config):
        """测试梯度流动"""
        batch_size = 2
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size,
            requires_grad=True
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        output = clip_model(pixel_values, input_ids)
        
        # 计算损失并反向传播
        loss = contrastive_loss(output['logits_per_image'], output['logits_per_text'])
        loss.backward()
        
        # 检查梯度存在
        assert pixel_values.grad is not None
        assert clip_model.logit_scale.grad is not None


class TestContrastiveLoss:
    """对比损失函数单元测试"""
    
    def test_contrastive_loss_output_scalar(self):
        """测试对比损失输出是标量"""
        batch_size = 4
        
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = logits_per_image.T
        
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 损失应该是标量
        assert loss.dim() == 0
    
    def test_contrastive_loss_positive_value(self):
        """测试对比损失是正值"""
        batch_size = 4
        
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = logits_per_image.T
        
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 损失应该是正值
        assert loss.item() > 0
    
    def test_contrastive_loss_perfect_matching(self):
        """测试完美匹配时的损失"""
        batch_size = 4
        
        # 创建完美匹配的 logits（对角线上的值最大）
        logits_per_image = torch.eye(batch_size) * 100 - 50
        logits_per_text = logits_per_image.T
        
        loss = contrastive_loss(logits_per_image, logits_per_text)
        
        # 完美匹配时损失应该接近 0
        assert loss.item() < 0.1
    
    def test_contrastive_loss_gradient_flow(self):
        """测试对比损失的梯度流动"""
        batch_size = 4
        
        logits_per_image = torch.randn(batch_size, batch_size, requires_grad=True)
        logits_per_text = logits_per_image.T
        
        loss = contrastive_loss(logits_per_image, logits_per_text)
        loss.backward()
        
        # 检查梯度存在
        assert logits_per_image.grad is not None


class TestCLIPModelIntegration:
    """CLIP 模型集成测试"""
    
    @pytest.fixture
    def clip_config(self):
        """创建测试用的 CLIP 配置"""
        vision_config = VisionConfig(
            image_size=64,
            patch_size=8,
            in_channels=3,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            dropout_rate=0.1,
            num_classes=0
        )
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout_rate=0.1
        )
        return CLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=64,
            temperature=0.07
        )
    
    def test_clip_training_step(self, clip_config):
        """测试 CLIP 训练步骤"""
        model = CLIPModel(clip_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        batch_size = 4
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        # 前向传播
        output = model(pixel_values, input_ids)
        
        # 计算损失
        loss = contrastive_loss(output['logits_per_image'], output['logits_per_text'])
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 检查损失是有效的
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_clip_eval_mode(self, clip_config):
        """测试 CLIP 评估模式"""
        model = CLIPModel(clip_config)
        model.eval()
        
        batch_size = 4
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
        
        with torch.no_grad():
            output = model(pixel_values, input_ids)
        
        # 检查输出形状
        assert output['image_embeds'].shape == (batch_size, clip_config.projection_dim)
        assert output['text_embeds'].shape == (batch_size, clip_config.projection_dim)
    
    def test_clip_different_batch_sizes(self, clip_config):
        """测试不同 batch size"""
        model = CLIPModel(clip_config)
        
        for batch_size in [1, 2, 8]:
            pixel_values = torch.randn(
                batch_size, 3,
                clip_config.vision_config.image_size,
                clip_config.vision_config.image_size
            )
            input_ids = torch.randint(0, clip_config.text_config.vocab_size, (batch_size, 16))
            
            output = model(pixel_values, input_ids)
            
            assert output['image_embeds'].shape == (batch_size, clip_config.projection_dim)
            assert output['text_embeds'].shape == (batch_size, clip_config.projection_dim)
            assert output['logits_per_image'].shape == (batch_size, batch_size)


class MockTokenizer:
    """模拟分词器，用于测试零样本分类"""
    
    def __init__(self, vocab_size: int = 1000, max_seq_len: int = 32):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
    
    def __call__(self, texts):
        """
        将文本列表转换为 token IDs
        
        Args:
            texts: 文本列表
        
        Returns:
            包含 'input_ids' 和 'attention_mask' 的字典
        """
        if isinstance(texts, str):
            texts = [texts]
        
        batch_size = len(texts)
        # 生成随机的 token IDs（模拟分词结果）
        # 使用文本长度来确定序列长度（简化模拟）
        input_ids = []
        attention_masks = []
        
        for text in texts:
            # 根据文本长度生成不同长度的序列
            seq_len = min(len(text.split()) + 2, self.max_seq_len)  # +2 for special tokens
            ids = torch.randint(1, self.vocab_size, (seq_len,))
            mask = torch.ones(seq_len)
            
            # Padding to max_seq_len
            if seq_len < self.max_seq_len:
                padding_len = self.max_seq_len - seq_len
                ids = torch.cat([ids, torch.zeros(padding_len, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(padding_len)])
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        return {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_masks)
        }


class TestZeroShotClassify:
    """零样本分类单元测试 - 需求 4.8"""
    
    @pytest.fixture
    def clip_config(self):
        """创建测试用的 CLIP 配置"""
        vision_config = VisionConfig(
            image_size=64,
            patch_size=8,
            in_channels=3,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            dropout_rate=0.1,
            num_classes=0
        )
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=32,
            dropout_rate=0.1
        )
        return CLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=64,
            temperature=0.07
        )
    
    @pytest.fixture
    def clip_model(self, clip_config):
        """创建测试用的 CLIPModel"""
        return CLIPModel(clip_config)
    
    @pytest.fixture
    def tokenizer(self, clip_config):
        """创建测试用的分词器"""
        return MockTokenizer(
            vocab_size=clip_config.text_config.vocab_size,
            max_seq_len=clip_config.text_config.max_seq_len
        )
    
    def test_zero_shot_classify_output_shapes(self, clip_model, clip_config, tokenizer):
        """测试零样本分类输出形状"""
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        predicted_labels, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查输出形状
        assert predicted_labels.shape == (batch_size,)
        assert probabilities.shape == (batch_size, num_labels)
    
    def test_zero_shot_classify_probabilities_sum_to_one(self, clip_model, clip_config, tokenizer):
        """测试零样本分类概率和为 1"""
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        _, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查每个样本的概率和为 1
        prob_sums = probabilities.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)
    
    def test_zero_shot_classify_probabilities_non_negative(self, clip_model, clip_config, tokenizer):
        """测试零样本分类概率非负"""
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        _, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查所有概率非负
        assert (probabilities >= 0).all()
    
    def test_zero_shot_classify_predicted_labels_valid(self, clip_model, clip_config, tokenizer):
        """测试预测标签在有效范围内"""
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        predicted_labels, _ = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查预测标签在有效范围内 [0, num_labels)
        assert (predicted_labels >= 0).all()
        assert (predicted_labels < num_labels).all()
    
    def test_zero_shot_classify_single_image(self, clip_model, clip_config, tokenizer):
        """测试单张图像的零样本分类"""
        batch_size = 1
        num_labels = 3
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = ["a cat", "a dog", "a bird"]
        
        predicted_labels, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查输出形状
        assert predicted_labels.shape == (batch_size,)
        assert probabilities.shape == (batch_size, num_labels)
    
    def test_zero_shot_classify_two_labels(self, clip_model, clip_config, tokenizer):
        """测试只有两个标签的零样本分类"""
        batch_size = 4
        num_labels = 2
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = ["positive", "negative"]
        
        predicted_labels, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查输出形状
        assert predicted_labels.shape == (batch_size,)
        assert probabilities.shape == (batch_size, num_labels)
    
    def test_zero_shot_classify_many_labels(self, clip_model, clip_config, tokenizer):
        """测试多个标签的零样本分类"""
        batch_size = 2
        num_labels = 100
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"class {i}" for i in range(num_labels)]
        
        predicted_labels, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查输出形状
        assert predicted_labels.shape == (batch_size,)
        assert probabilities.shape == (batch_size, num_labels)
    
    def test_zero_shot_classify_eval_mode(self, clip_model, clip_config, tokenizer):
        """测试评估模式下的零样本分类"""
        clip_model.eval()
        
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        with torch.no_grad():
            predicted_labels, probabilities = clip_model.zero_shot_classify(
                pixel_values, text_labels, tokenizer
            )
        
        # 检查输出形状
        assert predicted_labels.shape == (batch_size,)
        assert probabilities.shape == (batch_size, num_labels)
    
    def test_zero_shot_classify_predicted_label_matches_max_probability(self, clip_model, clip_config, tokenizer):
        """测试预测标签与最大概率一致"""
        batch_size = 4
        num_labels = 5
        
        pixel_values = torch.randn(
            batch_size, 3,
            clip_config.vision_config.image_size,
            clip_config.vision_config.image_size
        )
        text_labels = [f"a photo of class {i}" for i in range(num_labels)]
        
        predicted_labels, probabilities = clip_model.zero_shot_classify(
            pixel_values, text_labels, tokenizer
        )
        
        # 检查预测标签与最大概率索引一致
        expected_labels = torch.argmax(probabilities, dim=-1)
        assert torch.equal(predicted_labels, expected_labels)
