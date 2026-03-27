"""
多模态预训练模块单元测试

测试 ITC Loss、ITM Loss、ITG Loss、Hard Negative Mining、MultimodalPreTrainer 等组件。
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_models_from_scratch.training.multimodal_pretrain import (
    MultimodalPreTrainer,
    MultimodalPreTrainingConfig,
    compute_itc_loss,
    compute_itm_loss,
    compute_itg_loss,
    sample_hard_negatives,
    compute_itc_accuracy,
    compute_itm_accuracy,
)


class MockBLIPModel(nn.Module):
    """用于测试的模拟 BLIP 模型"""
    
    def __init__(self, embed_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 简单的图像编码器
        self.vision_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, embed_dim)
        )
        
        # 简单的文本编码器
        self.text_encoder = nn.Embedding(vocab_size, embed_dim)
        
        # 投影层
        self.visual_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)
        
        # ITM 头
        self.itm_head = nn.Linear(embed_dim, 2)
        
        # LM 头
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        
        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.tensor([2.6593]))
    
    def encode_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.vision_encoder(pixel_values)
        return features
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        features = self.text_encoder(input_ids).mean(dim=1)
        return features
    
    def forward_itc(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        
        image_embeds = self.visual_projection(image_features)
        text_embeds = self.text_projection(text_features)
        
        image_embeds = F.normalize(image_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        temperature = torch.exp(self.logit_scale)
        logits_per_image = temperature * (image_embeds @ text_embeds.T)
        logits_per_text = logits_per_image.T
        
        return {
            'image_embeds': image_embeds,
            'text_embeds': text_embeds,
            'logits_per_image': logits_per_image,
            'logits_per_text': logits_per_text
        }
    
    def forward_itm(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        image_features = self.encode_image(pixel_values)
        text_features = self.encode_text(input_ids, attention_mask)
        
        # 简单地将图像和文本特征相加
        combined = image_features + text_features
        itm_logits = self.itm_head(combined)
        
        return itm_logits
    
    def forward_itg(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        image_features = self.encode_image(pixel_values)
        text_features = self.text_encoder(input_ids)
        
        # 简单地将图像特征加到文本特征上
        combined = text_features + image_features.unsqueeze(1)
        logits = self.lm_head(combined)
        
        # 计算语言模型损失
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )
        
        return loss


class TestComputeITCLoss:
    """ITC Loss 测试"""
    
    def test_loss_shape(self):
        """测试损失输出为标量"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss = compute_itc_loss(image_embeds, text_embeds, temperature)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_with_perfect_alignment(self):
        """测试完美对齐时损失较低"""
        batch_size = 4
        embed_dim = 256
        
        embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        loss = compute_itc_loss(embeds, embeds, temperature)
        
        assert loss.item() < 1.0
    
    def test_loss_gradient_flow(self):
        """测试梯度能够正确流动"""
        batch_size = 4
        embed_dim = 256
        
        image_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
        text_raw = torch.randn(batch_size, embed_dim, requires_grad=True)
        temperature = torch.tensor([14.28], requires_grad=True)
        
        image_embeds = F.normalize(image_raw, dim=-1)
        text_embeds = F.normalize(text_raw, dim=-1)
        
        loss = compute_itc_loss(image_embeds, text_embeds, temperature)
        loss.backward()
        
        assert image_raw.grad is not None
        assert text_raw.grad is not None
        assert temperature.grad is not None


class TestComputeITMLoss:
    """ITM Loss 测试"""
    
    def test_loss_shape(self):
        """测试损失输出为标量"""
        batch_size = 4
        
        itm_logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        
        loss = compute_itm_loss(itm_logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_with_correct_predictions(self):
        """测试正确预测时损失较低"""
        batch_size = 4
        
        # 创建明确的预测
        itm_logits = torch.zeros(batch_size, 2)
        itm_logits[:, 1] = 10.0  # 强烈预测类别 1
        labels = torch.ones(batch_size, dtype=torch.long)
        
        loss = compute_itm_loss(itm_logits, labels)
        
        assert loss.item() < 0.1
    
    def test_loss_gradient_flow(self):
        """测试梯度能够正确流动"""
        batch_size = 4
        
        itm_logits = torch.randn(batch_size, 2, requires_grad=True)
        labels = torch.randint(0, 2, (batch_size,))
        
        loss = compute_itm_loss(itm_logits, labels)
        loss.backward()
        
        assert itm_logits.grad is not None


class TestComputeITGLoss:
    """ITG Loss 测试"""
    
    def test_loss_shape(self):
        """测试损失输出为标量"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        lm_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_itg_loss(lm_logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_ignores_padding(self):
        """测试忽略 padding token"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        lm_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, seq_len // 2:] = -100  # 后半部分为 padding
        
        loss = compute_itg_loss(lm_logits, labels, ignore_index=-100)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_gradient_flow(self):
        """测试梯度能够正确流动"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        lm_logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_itg_loss(lm_logits, labels)
        loss.backward()
        
        assert lm_logits.grad is not None


class TestSampleHardNegatives:
    """Hard Negative Mining 测试"""
    
    def test_output_shapes(self):
        """测试输出形状"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        neg_image_indices, neg_text_indices, labels = sample_hard_negatives(
            image_embeds, text_embeds, hard_negative_ratio=0.5
        )
        
        assert neg_image_indices.shape == (batch_size,)
        assert neg_text_indices.shape == (batch_size,)
        assert labels.shape == (batch_size * 3,)
    
    def test_labels_distribution(self):
        """测试标签分布"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        _, _, labels = sample_hard_negatives(
            image_embeds, text_embeds, hard_negative_ratio=0.5
        )
        
        # 前 batch_size 个应该是正样本 (1)
        assert (labels[:batch_size] == 1).all()
        # 后 2*batch_size 个应该是负样本 (0)
        assert (labels[batch_size:] == 0).all()
    
    def test_negative_indices_not_diagonal(self):
        """测试负样本索引不在对角线上"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        neg_image_indices, neg_text_indices, _ = sample_hard_negatives(
            image_embeds, text_embeds, hard_negative_ratio=1.0
        )
        
        # 负样本索引不应该等于自身索引
        for i in range(batch_size):
            assert neg_text_indices[i].item() != i
            assert neg_image_indices[i].item() != i
    
    def test_hard_negative_ratio(self):
        """测试困难负样本比例"""
        batch_size = 8
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        # 测试不同的比例
        for ratio in [0.0, 0.5, 1.0]:
            neg_image_indices, neg_text_indices, labels = sample_hard_negatives(
                image_embeds, text_embeds, hard_negative_ratio=ratio
            )
            
            assert neg_image_indices.shape == (batch_size,)
            assert neg_text_indices.shape == (batch_size,)


class TestComputeITCAccuracy:
    """ITC 准确率计算测试"""
    
    def test_accuracy_shape(self):
        """测试准确率输出格式"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        metrics = compute_itc_accuracy(image_embeds, text_embeds, temperature)
        
        assert 'i2t_accuracy' in metrics
        assert 't2i_accuracy' in metrics
        assert 'mean_accuracy' in metrics
        
        assert 0 <= metrics['i2t_accuracy'] <= 1
        assert 0 <= metrics['t2i_accuracy'] <= 1
        assert 0 <= metrics['mean_accuracy'] <= 1
    
    def test_perfect_accuracy(self):
        """测试完美对齐时准确率为 1"""
        batch_size = 4
        embed_dim = 256
        
        embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        temperature = torch.tensor([14.28])
        
        metrics = compute_itc_accuracy(embeds, embeds, temperature)
        
        assert metrics['i2t_accuracy'] == 1.0
        assert metrics['t2i_accuracy'] == 1.0
        assert metrics['mean_accuracy'] == 1.0


class TestComputeITMAccuracy:
    """ITM 准确率计算测试"""
    
    def test_accuracy_range(self):
        """测试准确率范围"""
        batch_size = 4
        
        itm_logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        
        accuracy = compute_itm_accuracy(itm_logits, labels)
        
        assert 0 <= accuracy <= 1
    
    def test_perfect_accuracy(self):
        """测试完美预测时准确率为 1"""
        batch_size = 4
        
        itm_logits = torch.zeros(batch_size, 2)
        itm_logits[:, 1] = 10.0
        labels = torch.ones(batch_size, dtype=torch.long)
        
        accuracy = compute_itm_accuracy(itm_logits, labels)
        
        assert accuracy == 1.0


class TestMultimodalPreTrainingConfig:
    """多模态预训练配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = MultimodalPreTrainingConfig()
        
        assert config.lambda_itc == 1.0
        assert config.lambda_itm == 1.0
        assert config.lambda_itg == 1.0
        assert config.hard_negative_ratio == 0.5
        assert config.freeze_vision_encoder == False
        assert config.gradient_accumulation_steps == 1
        assert config.temperature == 0.07
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = MultimodalPreTrainingConfig(
            lambda_itc=0.5,
            lambda_itm=1.5,
            lambda_itg=2.0,
            hard_negative_ratio=0.8,
            freeze_vision_encoder=True,
            learning_rate=5e-5
        )
        
        assert config.lambda_itc == 0.5
        assert config.lambda_itm == 1.5
        assert config.lambda_itg == 2.0
        assert config.hard_negative_ratio == 0.8
        assert config.freeze_vision_encoder == True
        assert config.learning_rate == 5e-5


class TestMultimodalPreTrainer:
    """MultimodalPreTrainer 测试"""
    
    @pytest.fixture
    def trainer(self):
        """创建测试用的 trainer"""
        model = MockBLIPModel(embed_dim=256, vocab_size=1000)
        config = MultimodalPreTrainingConfig(
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            log_steps=10,
            save_steps=100,
            lambda_itc=1.0,
            lambda_itm=1.0,
            lambda_itg=1.0
        )
        return MultimodalPreTrainer(model, config)
    
    @pytest.fixture
    def sample_batch(self):
        """创建测试用的 batch"""
        batch_size = 4
        seq_len = 16
        return {
            'pixel_values': torch.randn(batch_size, 3, 32, 32),
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': torch.randint(0, 1000, (batch_size, seq_len))
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
        
        assert 'total_loss' in metrics
        assert 'itc_loss' in metrics
        assert 'itm_loss' in metrics
        assert 'itg_loss' in metrics
        assert 'did_update' in metrics
        
        assert isinstance(metrics['total_loss'], float)
        assert metrics['total_loss'] >= 0
    
    def test_gradient_accumulation(self, trainer, sample_batch):
        """测试梯度累积"""
        # 第一步：不应该更新参数
        metrics1 = trainer.train_step(sample_batch)
        assert metrics1['did_update'] == False
        assert trainer.accumulation_step == 1
        assert trainer.global_step == 0
        
        # 第二步：应该更新参数
        metrics2 = trainer.train_step(sample_batch)
        assert metrics2['did_update'] == True
        assert trainer.accumulation_step == 0
        assert trainer.global_step == 1
    
    def test_loss_weights(self, trainer, sample_batch):
        """测试损失权重"""
        # 修改权重
        trainer.config.lambda_itc = 2.0
        trainer.config.lambda_itm = 0.5
        trainer.config.lambda_itg = 1.5
        
        metrics = trainer.train_step(sample_batch)
        
        # 验证总损失是加权和
        expected_total = (
            2.0 * metrics['itc_loss'] +
            0.5 * metrics['itm_loss'] +
            1.5 * metrics['itg_loss']
        )
        
        assert abs(metrics['total_loss'] - expected_total) < 0.01
    
    def test_compute_itc_loss_method(self, trainer):
        """测试 compute_itc_loss 方法"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        loss = trainer.compute_itc_loss(image_embeds, text_embeds)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_compute_itm_loss_method(self, trainer):
        """测试 compute_itm_loss 方法"""
        batch_size = 4
        
        itm_logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        
        loss = trainer.compute_itm_loss(itm_logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_compute_itg_loss_method(self, trainer):
        """测试 compute_itg_loss 方法"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        lm_logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = trainer.compute_itg_loss(lm_logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_sample_hard_negatives_method(self, trainer):
        """测试 sample_hard_negatives 方法"""
        batch_size = 4
        embed_dim = 256
        
        image_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        text_embeds = F.normalize(torch.randn(batch_size, embed_dim), dim=-1)
        
        neg_image_indices, neg_text_indices, labels = trainer.sample_hard_negatives(
            image_embeds, text_embeds
        )
        
        assert neg_image_indices.shape == (batch_size,)
        assert neg_text_indices.shape == (batch_size,)
        assert labels.shape == (batch_size * 3,)
    
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
    
    def test_accuracy_metrics(self, trainer, sample_batch):
        """测试准确率指标"""
        metrics = trainer.train_step(sample_batch)
        
        # 检查准确率指标存在
        assert 'itc_accuracy' in metrics
        assert 'itm_accuracy' in metrics
        
        # 检查准确率范围
        assert 0 <= metrics['itc_accuracy'] <= 1
        assert 0 <= metrics['itm_accuracy'] <= 1


class TestMultimodalPreTrainerWithRealBLIP:
    """使用真实 BLIP 模型的集成测试"""
    
    @pytest.fixture
    def real_trainer(self):
        """创建使用真实 BLIP 模型的 trainer"""
        from multimodal_models_from_scratch.multimodal.blip import BLIPModel
        from multimodal_models_from_scratch.config import BLIPConfig, VisionConfig
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
        blip_config = BLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=64
        )
        
        model = BLIPModel(blip_config)
        config = MultimodalPreTrainingConfig(
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            lambda_itc=1.0,
            lambda_itm=1.0,
            lambda_itg=1.0
        )
        
        return MultimodalPreTrainer(model, config)
    
    @pytest.fixture
    def real_batch(self):
        """创建真实 BLIP 模型的测试 batch"""
        batch_size = 2
        seq_len = 16
        return {
            'pixel_values': torch.randn(batch_size, 3, 32, 32),
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': torch.randint(0, 1000, (batch_size, seq_len))
        }
    
    def test_real_blip_train_step(self, real_trainer, real_batch):
        """测试使用真实 BLIP 模型的训练步骤"""
        metrics = real_trainer.train_step(real_batch)
        
        assert 'total_loss' in metrics
        assert 'itc_loss' in metrics
        assert 'itm_loss' in metrics
        assert 'itg_loss' in metrics
        
        assert metrics['total_loss'] >= 0
        assert metrics['itc_loss'] >= 0
        assert metrics['itm_loss'] >= 0
        assert metrics['itg_loss'] >= 0
    
    def test_real_blip_multiple_steps(self, real_trainer, real_batch):
        """测试多步训练"""
        losses = []
        
        for _ in range(3):
            metrics = real_trainer.train_step(real_batch)
            losses.append(metrics['total_loss'])
        
        assert len(losses) == 3
        assert all(loss >= 0 for loss in losses)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
