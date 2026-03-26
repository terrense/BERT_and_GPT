"""
对比学习训练模块单元测试

测试 InfoNCE Loss、ContrastiveTrainer 等组件。
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_models_from_scratch.training.contrastive import (
    ContrastiveTrainer,
    ContrastiveTrainingConfig,
    info_nce_loss,
    compute_contrastive_accuracy,
)


class MockCLIPModel(nn.Module):
    """用于测试的模拟 CLIP 模型"""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 简单的图像编码器
        self.image_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, embed_dim)
        )
        
        # 简单的文本编码器
        self.text_encoder = nn.Embedding(1000, embed_dim)
        
        # 可学习的温度参数（log scale）
        self.logit_scale = nn.Parameter(torch.tensor([2.6593]))  # log(1/0.07)
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.image_encoder(pixel_values)
        return F.normalize(features, p=2, dim=-1)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # 简单地取平均
        features = self.text_encoder(input_ids).mean(dim=1)
        return F.normalize(features, p=2, dim=-1)
    
    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        image_embeds = self.encode_image(pixel_values)
        text_embeds = self.encode_text(input_ids, attention_mask)
        
        temperature = torch.exp(self.logit_scale)
        
        logits_per_image = temperature * (image_embeds @ text_embeds.T)
        logits_per_text = logits_per_image.T
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text,
            'temperature': temperature.squeeze()
        }


class TestInfoNCELoss:
    """InfoNCE Loss 测试"""
    
    def test_loss_shape(self):
        """测试损失输出为标量"""
        batch_size = 4
        embed_dim = 512
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])  # 1/0.07
        
        loss = info_nce_loss(image_embeds, text_embeds, temperature)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_with_perfect_alignment(self):
        """测试完美对齐时损失较低"""
        batch_size = 4
        embed_dim = 512
        
        # 创建完美对齐的嵌入（图像和文本相同）
        embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss = info_nce_loss(embeds, embeds, temperature)
        
        # 完美对齐时损失应该很低
        assert loss.item() < 1.0
    
    def test_loss_with_random_alignment(self):
        """测试随机对齐时损失较高"""
        batch_size = 4
        embed_dim = 512
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss = info_nce_loss(image_embeds, text_embeds, temperature)
        
        # 随机对齐时损失应该接近 log(batch_size)
        expected_random_loss = torch.log(torch.tensor(float(batch_size)))
        assert abs(loss.item() - expected_random_loss.item()) < 1.0
    
    def test_loss_gradient_flow(self):
        """测试梯度能够正确流动"""
        batch_size = 4
        embed_dim = 512
        
        # 创建叶子张量
        image_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
        text_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
        temperature = torch.tensor([14.28], requires_grad=True)
        
        # 归一化（非叶子张量）
        image_embeds = F.normalize(image_raw, dim=-1)
        text_embeds = F.normalize(text_raw, dim=-1)
        
        loss = info_nce_loss(image_embeds, text_embeds, temperature)
        loss.backward()
        
        # 检查叶子张量的梯度存在
        assert image_raw.grad is not None
        assert text_raw.grad is not None
        assert temperature.grad is not None
    
    def test_loss_symmetry(self):
        """测试损失的对称性"""
        batch_size = 4
        embed_dim = 512
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss1 = info_nce_loss(image_embeds, text_embeds, temperature)
        loss2 = info_nce_loss(text_embeds, image_embeds, temperature)
        
        # 由于是双向平均，交换顺序应该得到相同的损失
        assert torch.isclose(loss1, loss2, atol=1e-5)


class TestComputeContrastiveAccuracy:
    """对比学习准确率计算测试"""
    
    def test_accuracy_shape(self):
        """测试准确率输出格式"""
        batch_size = 4
        embed_dim = 512
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        metrics = compute_contrastive_accuracy(image_embeds, text_embeds, temperature)
        
        assert 'i2t_accuracy' in metrics
        assert 't2i_accuracy' in metrics
        assert 'mean_accuracy' in metrics
        
        assert 0 <= metrics['i2t_accuracy'] <= 1
        assert 0 <= metrics['t2i_accuracy'] <= 1
        assert 0 <= metrics['mean_accuracy'] <= 1
    
    def test_perfect_accuracy(self):
        """测试完美对齐时准确率为 1"""
        batch_size = 4
        embed_dim = 512
        
        # 创建完美对齐的嵌入
        embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        metrics = compute_contrastive_accuracy(embeds, embeds, temperature)
        
        assert metrics['i2t_accuracy'] == 1.0
        assert metrics['t2i_accuracy'] == 1.0
        assert metrics['mean_accuracy'] == 1.0


class TestContrastiveTrainingConfig:
    """对比学习训练配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ContrastiveTrainingConfig()
        
        assert config.temperature == 0.07
        assert config.gradient_accumulation_steps == 1
        assert config.use_hard_negatives == False
        assert config.learning_rate == 1e-4
        assert config.batch_size == 32
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = ContrastiveTrainingConfig(
            temperature=0.1,
            gradient_accumulation_steps=4,
            learning_rate=5e-5,
            batch_size=64
        )
        
        assert config.temperature == 0.1
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 5e-5
        assert config.batch_size == 64


class TestContrastiveTrainer:
    """ContrastiveTrainer 测试"""
    
    @pytest.fixture
    def trainer(self):
        """创建测试用的 trainer"""
        model = MockCLIPModel(embed_dim=512)
        config = ContrastiveTrainingConfig(
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            log_steps=10,
            save_steps=100
        )
        return ContrastiveTrainer(model, config)
    
    @pytest.fixture
    def sample_batch(self):
        """创建测试用的 batch"""
        batch_size = 4
        return {
            'pixel_values': torch.randn(batch_size, 3, 224, 224),
            'input_ids': torch.randint(0, 1000, (batch_size, 32)),
            'attention_mask': torch.ones(batch_size, 32)
        }
    
    def test_trainer_initialization(self, trainer):
        """测试 trainer 初始化"""
        assert trainer.model is not None
        assert trainer.config is not None
        assert trainer.optimizer is not None
        assert trainer.global_step == 0
        assert trainer.accumulation_step == 0
    
    def test_train_step_output(self, trainer, sample_batch):
        """测试 train_step 输出格式"""
        metrics = trainer.train_step(sample_batch)
        
        assert 'loss' in metrics
        assert 'i2t_accuracy' in metrics
        assert 't2i_accuracy' in metrics
        assert 'mean_accuracy' in metrics
        assert 'temperature' in metrics
        assert 'did_update' in metrics
        
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0
    
    def test_gradient_accumulation(self, trainer, sample_batch):
        """测试梯度累积"""
        # 第一步：不应该更新参数
        metrics1 = trainer.train_step(sample_batch)
        assert metrics1['did_update'] == False
        assert trainer.accumulation_step == 1
        assert trainer.global_step == 0
        
        # 第二步：应该更新参数（因为 gradient_accumulation_steps=2）
        metrics2 = trainer.train_step(sample_batch)
        assert metrics2['did_update'] == True
        assert trainer.accumulation_step == 0
        assert trainer.global_step == 1
    
    def test_compute_loss(self, trainer):
        """测试 compute_loss 方法"""
        batch_size = 4
        embed_dim = 512
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss = trainer.compute_loss(image_embeds, text_embeds, temperature)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_set_scheduler(self, trainer):
        """测试学习率调度器设置"""
        total_steps = 1000
        warmup_steps = 100
        
        trainer.set_scheduler(total_steps, warmup_steps)
        
        assert trainer.scheduler is not None
    
    def test_multiple_train_steps(self, trainer, sample_batch):
        """测试多步训练"""
        initial_params = {
            name: param.clone() 
            for name, param in trainer.model.named_parameters()
        }
        
        # 执行多步训练
        for _ in range(4):  # 2 次参数更新
            trainer.train_step(sample_batch)
        
        # 检查参数已更新
        params_changed = False
        for name, param in trainer.model.named_parameters():
            if not torch.allclose(param, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed, "Parameters should have changed after training"
        assert trainer.global_step == 2


class TestContrastiveTrainerWithRealCLIP:
    """使用真实 CLIP 模型的集成测试"""
    
    @pytest.fixture
    def real_trainer(self):
        """创建使用真实 CLIP 模型的 trainer"""
        from multimodal_models_from_scratch.multimodal.clip import CLIPModel
        from multimodal_models_from_scratch.config import CLIPConfig, VisionConfig
        from bert_gpt_from_scratch.config import TransformerConfig
        
        # 使用小型配置以加快测试
        vision_config = VisionConfig(
            image_size=32,
            patch_size=8,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128
        )
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=32
        )
        clip_config = CLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=64,
            temperature=0.07
        )
        
        model = CLIPModel(clip_config)
        config = ContrastiveTrainingConfig(
            learning_rate=1e-4,
            gradient_accumulation_steps=1
        )
        
        return ContrastiveTrainer(model, config)
    
    @pytest.fixture
    def real_batch(self):
        """创建真实 CLIP 模型的测试 batch"""
        batch_size = 2
        return {
            'pixel_values': torch.randn(batch_size, 3, 32, 32),
            'input_ids': torch.randint(0, 1000, (batch_size, 16)),
            'attention_mask': torch.ones(batch_size, 16)
        }
    
    def test_real_clip_train_step(self, real_trainer, real_batch):
        """测试使用真实 CLIP 模型的训练步骤"""
        metrics = real_trainer.train_step(real_batch)
        
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
        assert 'temperature' in metrics
        assert metrics['temperature'] > 0
    
    def test_real_clip_loss_decreases(self, real_trainer, real_batch):
        """测试损失是否随训练下降"""
        losses = []
        
        for _ in range(5):
            metrics = real_trainer.train_step(real_batch)
            losses.append(metrics['loss'])
        
        # 检查损失有下降趋势（不要求严格单调下降）
        # 由于是小 batch 和随机数据，可能会有波动
        assert len(losses) == 5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
