"""
BLIP 模型测试

测试 BLIP 模型的各个组件和功能：
- CrossAttentionLayer
- TextEncoderWithCrossAttention
- TextDecoderWithCrossAttention
- BLIPModel
- ITC、ITM、ITG 任务
"""

import pytest
import torch
import torch.nn as nn

from bert_gpt_from_scratch.config import TransformerConfig
from multimodal_models_from_scratch.config import BLIPConfig, VisionConfig
from multimodal_models_from_scratch.multimodal.blip import (
    CrossAttentionLayer,
    TextEncoderWithCrossAttention,
    TextDecoderWithCrossAttention,
    BLIPModel,
    itc_loss,
    itm_loss,
)


class TestCrossAttentionLayer:
    """CrossAttentionLayer 测试"""
    
    def test_cross_attention_output_shape(self):
        """测试交叉注意力层输出形状"""
        batch_size = 2
        seq_len = 16
        num_patches = 49
        d_model = 256
        
        layer = CrossAttentionLayer(d_model=d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        output = layer(hidden_states, encoder_hidden_states)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_cross_attention_with_mask(self):
        """测试带掩码的交叉注意力"""
        batch_size = 2
        seq_len = 16
        num_patches = 49
        d_model = 256
        
        layer = CrossAttentionLayer(d_model=d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        encoder_attention_mask = torch.ones(batch_size, num_patches)
        encoder_attention_mask[:, -10:] = 0  # 掩码最后 10 个位置
        
        output = layer(hidden_states, encoder_hidden_states, encoder_attention_mask)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTextEncoderWithCrossAttention:
    """TextEncoderWithCrossAttention 测试"""
    
    @pytest.fixture
    def text_config(self):
        return TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout_rate=0.1
        )
    
    def test_encoder_output_shape(self, text_config):
        """测试编码器输出形状"""
        batch_size = 2
        seq_len = 32
        num_patches = 49
        
        encoder = TextEncoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, num_patches, text_config.d_model)
        
        output = encoder(input_ids, encoder_hidden_states=encoder_hidden_states)
        
        assert output.shape == (batch_size, seq_len, text_config.d_model)
    
    def test_encoder_without_cross_attention(self, text_config):
        """测试不使用交叉注意力的编码器"""
        batch_size = 2
        seq_len = 32
        
        encoder = TextEncoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        
        # 不提供 encoder_hidden_states
        output = encoder(input_ids)
        
        assert output.shape == (batch_size, seq_len, text_config.d_model)
    
    def test_encoder_with_attention_mask(self, text_config):
        """测试带注意力掩码的编码器"""
        batch_size = 2
        seq_len = 32
        num_patches = 49
        
        encoder = TextEncoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, -5:] = 0  # 掩码最后 5 个位置
        encoder_hidden_states = torch.randn(batch_size, num_patches, text_config.d_model)
        
        output = encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states
        )
        
        assert output.shape == (batch_size, seq_len, text_config.d_model)


class TestTextDecoderWithCrossAttention:
    """TextDecoderWithCrossAttention 测试"""
    
    @pytest.fixture
    def text_config(self):
        return TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout_rate=0.1
        )
    
    def test_decoder_output_shape(self, text_config):
        """测试解码器输出形状"""
        batch_size = 2
        seq_len = 32
        num_patches = 49
        
        decoder = TextDecoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, num_patches, text_config.d_model)
        
        output = decoder(input_ids, encoder_hidden_states=encoder_hidden_states)
        
        assert 'logits' in output
        assert output['logits'].shape == (batch_size, seq_len, text_config.vocab_size)
    
    def test_decoder_with_labels(self, text_config):
        """测试带标签的解码器（计算损失）"""
        batch_size = 2
        seq_len = 32
        num_patches = 49
        
        decoder = TextDecoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, num_patches, text_config.d_model)
        
        output = decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            labels=labels
        )
        
        assert 'logits' in output
        assert 'loss' in output
        assert output['loss'].dim() == 0  # scalar
    
    def test_decoder_causal_mask(self, text_config):
        """测试解码器的因果掩码"""
        batch_size = 1
        seq_len = 8
        num_patches = 16
        
        decoder = TextDecoderWithCrossAttention(text_config)
        
        input_ids = torch.randint(0, text_config.vocab_size, (batch_size, seq_len))
        encoder_hidden_states = torch.randn(batch_size, num_patches, text_config.d_model)
        
        # 前向传播两次，第二次只改变最后一个 token
        output1 = decoder(input_ids, encoder_hidden_states=encoder_hidden_states)
        
        input_ids_modified = input_ids.clone()
        input_ids_modified[:, -1] = (input_ids[:, -1] + 1) % text_config.vocab_size
        output2 = decoder(input_ids_modified, encoder_hidden_states=encoder_hidden_states)
        
        # 由于因果掩码，前面位置的 logits 应该相同
        # 注意：由于 dropout，需要在 eval 模式下测试
        decoder.eval()
        with torch.no_grad():
            output1 = decoder(input_ids, encoder_hidden_states=encoder_hidden_states)
            output2 = decoder(input_ids_modified, encoder_hidden_states=encoder_hidden_states)
        
        # 前 seq_len-1 个位置的 logits 应该相同
        assert torch.allclose(
            output1['logits'][:, :-1, :],
            output2['logits'][:, :-1, :],
            atol=1e-5
        )


class TestBLIPModel:
    """BLIPModel 测试"""
    
    @pytest.fixture
    def blip_config(self):
        vision_config = VisionConfig(
            image_size=64,
            patch_size=8,
            in_channels=3,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=512,
            dropout_rate=0.1,
            num_classes=0
        )
        text_config = TransformerConfig(
            vocab_size=1000,
            d_model=256,
            num_heads=8,
            num_layers=2,
            d_ff=512,
            max_seq_len=64,
            dropout_rate=0.1
        )
        return BLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=128
        )
    
    def test_model_initialization(self, blip_config):
        """测试模型初始化"""
        model = BLIPModel(blip_config)
        
        assert model.vision_encoder is not None
        assert model.text_encoder is not None
        assert model.text_decoder is not None
        assert model.visual_projection is not None
        assert model.text_projection is not None
        assert model.itm_head is not None
    
    def test_encode_image(self, blip_config):
        """测试图像编码"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        
        output = model.encode_image(pixel_values)
        
        num_patches = (blip_config.vision_config.image_size // blip_config.vision_config.patch_size) ** 2
        
        assert 'image_features' in output
        assert 'pooler_output' in output
        assert output['image_features'].shape == (batch_size, num_patches, blip_config.vision_config.d_model)
        assert output['pooler_output'].shape == (batch_size, blip_config.vision_config.d_model)
    
    def test_forward_itc(self, blip_config):
        """测试 ITC 前向传播"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 16
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        output = model.forward_itc(pixel_values, input_ids)
        
        assert 'image_embeds' in output
        assert 'text_embeds' in output
        assert 'logits_per_image' in output
        assert 'logits_per_text' in output
        
        assert output['image_embeds'].shape == (batch_size, blip_config.projection_dim)
        assert output['text_embeds'].shape == (batch_size, blip_config.projection_dim)
        assert output['logits_per_image'].shape == (batch_size, batch_size)
        assert output['logits_per_text'].shape == (batch_size, batch_size)
        
        # 检查 L2 归一化
        image_norms = torch.norm(output['image_embeds'], p=2, dim=-1)
        text_norms = torch.norm(output['text_embeds'], p=2, dim=-1)
        assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5)
        assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5)
    
    def test_forward_itm(self, blip_config):
        """测试 ITM 前向传播"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 16
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        itm_logits = model.forward_itm(pixel_values, input_ids)
        
        assert itm_logits.shape == (batch_size, 2)
    
    def test_forward_itg(self, blip_config):
        """测试 ITG 前向传播"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 16
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        lm_loss = model.forward_itg(pixel_values, input_ids, labels)
        
        assert lm_loss.dim() == 0  # scalar
        assert lm_loss.item() > 0  # 损失应该为正
    
    def test_forward_all_tasks(self, blip_config):
        """测试完整前向传播（所有任务）"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 16
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        output = model(pixel_values, input_ids, labels=labels)
        
        assert 'itc_output' in output
        assert 'itm_logits' in output
        assert 'itg_loss' in output
    
    def test_generate_caption(self, blip_config):
        """测试图像描述生成"""
        model = BLIPModel(blip_config)
        model.eval()
        
        batch_size = 2
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        
        with torch.no_grad():
            generated_ids = model.generate_caption(
                pixel_values,
                max_length=10,
                bos_token_id=1,
                eos_token_id=2
            )
        
        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] <= 10
        assert generated_ids[:, 0].tolist() == [1, 1]  # BOS token
    
    def test_visual_question_answering(self, blip_config):
        """测试视觉问答"""
        model = BLIPModel(blip_config)
        model.eval()
        
        batch_size = 2
        question_len = 8
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size
        )
        question_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, question_len))
        
        with torch.no_grad():
            answer_ids = model.visual_question_answering(
                pixel_values,
                question_ids,
                max_length=10,
                bos_token_id=1,
                eos_token_id=2
            )
        
        assert answer_ids.shape[0] == batch_size
        assert answer_ids.shape[1] <= 10


class TestLossFunctions:
    """损失函数测试"""
    
    def test_itc_loss(self):
        """测试 ITC 损失"""
        batch_size = 4
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = logits_per_image.T
        
        loss = itc_loss(logits_per_image, logits_per_text)
        
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # 损失应该为正
    
    def test_itc_loss_perfect_match(self):
        """测试完美匹配时的 ITC 损失"""
        batch_size = 4
        # 创建对角线为高值的相似度矩阵
        logits_per_image = torch.eye(batch_size) * 100 - 50
        logits_per_text = logits_per_image.T
        
        loss = itc_loss(logits_per_image, logits_per_text)
        
        # 完美匹配时损失应该接近 0
        assert loss.item() < 0.1
    
    def test_itm_loss(self):
        """测试 ITM 损失"""
        batch_size = 4
        itm_logits = torch.randn(batch_size, 2)
        labels = torch.randint(0, 2, (batch_size,))
        
        loss = itm_loss(itm_logits, labels)
        
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # 损失应该为正
    
    def test_itm_loss_perfect_prediction(self):
        """测试完美预测时的 ITM 损失"""
        batch_size = 4
        labels = torch.tensor([0, 1, 0, 1])
        # 创建完美预测的 logits
        itm_logits = torch.zeros(batch_size, 2)
        itm_logits[labels == 0, 0] = 100
        itm_logits[labels == 1, 1] = 100
        
        loss = itm_loss(itm_logits, labels)
        
        # 完美预测时损失应该接近 0
        assert loss.item() < 0.01


class TestBLIPGradients:
    """BLIP 模型梯度测试"""
    
    @pytest.fixture
    def blip_config(self):
        vision_config = VisionConfig(
            image_size=32,
            patch_size=8,
            in_channels=3,
            d_model=64,
            num_heads=4,
            num_layers=1,
            d_ff=128,
            dropout_rate=0.0,
            num_classes=0
        )
        text_config = TransformerConfig(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=1,
            d_ff=128,
            max_seq_len=32,
            dropout_rate=0.0
        )
        return BLIPConfig(
            vision_config=vision_config,
            text_config=text_config,
            projection_dim=32
        )
    
    def test_itc_gradients(self, blip_config):
        """测试 ITC 任务的梯度流"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 8
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size,
            requires_grad=True
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        output = model.forward_itc(pixel_values, input_ids)
        loss = itc_loss(output['logits_per_image'], output['logits_per_text'])
        loss.backward()
        
        # 检查梯度是否存在
        assert pixel_values.grad is not None
        assert model.visual_projection.weight.grad is not None
        assert model.text_projection.weight.grad is not None
    
    def test_itm_gradients(self, blip_config):
        """测试 ITM 任务的梯度流"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 8
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size,
            requires_grad=True
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, 2, (batch_size,))
        
        itm_logits = model.forward_itm(pixel_values, input_ids)
        loss = itm_loss(itm_logits, labels)
        loss.backward()
        
        # 检查梯度是否存在
        assert pixel_values.grad is not None
        assert model.itm_head.weight.grad is not None
    
    def test_itg_gradients(self, blip_config):
        """测试 ITG 任务的梯度流"""
        model = BLIPModel(blip_config)
        
        batch_size = 2
        seq_len = 8
        pixel_values = torch.randn(
            batch_size, 3,
            blip_config.vision_config.image_size,
            blip_config.vision_config.image_size,
            requires_grad=True
        )
        input_ids = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, blip_config.text_config.vocab_size, (batch_size, seq_len))
        
        loss = model.forward_itg(pixel_values, input_ids, labels)
        loss.backward()
        
        # 检查梯度是否存在
        assert pixel_values.grad is not None
        assert model.text_decoder.lm_head.weight.grad is not None
