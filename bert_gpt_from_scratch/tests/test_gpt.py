"""
GPT 模型单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.config import GPTConfig
from bert_gpt_from_scratch.models.gpt import GPTModel, LMHead


class TestGPTModel:
    """GPT 模型测试"""
    
    @pytest.fixture
    def config(self):
        """创建小型 GPT 配置用于测试"""
        return GPTConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout_rate=0.0,
            tie_weights=True
        )
    
    @pytest.fixture
    def model(self, config):
        """创建 GPT 模型"""
        return GPTModel(config)
    
    def test_forward_output_shape(self, model, config):
        """测试前向传播输出形状"""
        batch, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len)
        
        outputs = model(input_ids, attention_mask)
        
        assert outputs['hidden_states'].shape == (batch, seq_len, config.d_model)
        assert outputs['logits'].shape == (batch, seq_len, config.vocab_size)
    
    def test_logits_shape(self, model, config):
        """测试 logits 形状为 (batch, seq_len, vocab_size)"""
        batch, seq_len = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        outputs = model(input_ids)
        
        assert outputs['logits'].shape == (batch, seq_len, config.vocab_size)
    
    def test_weight_tying(self, config):
        """测试权重绑定正确性"""
        config.tie_weights = True
        model = GPTModel(config)
        
        # LM Head 的权重应该与 Token Embedding 的权重相同
        assert model.lm_head.decoder.weight is model.token_embedding.weight
    
    def test_no_weight_tying(self, config):
        """测试不绑定权重"""
        config.tie_weights = False
        model = GPTModel(config)
        
        # LM Head 的权重应该与 Token Embedding 的权重不同
        assert model.lm_head.decoder.weight is not model.token_embedding.weight
    
    def test_with_attention_mask(self, model, config):
        """测试带 attention mask 的前向传播"""
        batch, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[0, 20:] = 0  # 第一个样本后 12 个位置是 padding
        
        outputs = model(input_ids, attention_mask)
        
        assert outputs['hidden_states'].shape == (batch, seq_len, config.d_model)
    
    def test_causal_attention(self, model, config):
        """测试因果注意力（自回归特性）"""
        batch, seq_len = 1, 10
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        model.eval()
        
        # 完整序列的输出
        full_outputs = model(input_ids)
        
        # 只用前 5 个 token 的输出
        partial_outputs = model(input_ids[:, :5])
        
        # 前 5 个位置的 logits 应该相同（因为因果注意力）
        assert torch.allclose(
            full_outputs['logits'][:, :5, :],
            partial_outputs['logits'],
            atol=1e-5
        )


class TestLMHead:
    """LM Head 测试"""
    
    def test_output_shape(self):
        """测试输出形状"""
        d_model, vocab_size = 64, 1000
        head = LMHead(d_model, vocab_size)
        
        batch, seq_len = 2, 32
        hidden_states = torch.randn(batch, seq_len, d_model)
        
        logits = head(hidden_states)
        
        assert logits.shape == (batch, seq_len, vocab_size)
