"""
BERT 模型单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.config import BERTConfig
from bert_gpt_from_scratch.models.bert import BERTModel, MLMHead, NSPHead


class TestBERTModel:
    """BERT 模型测试"""
    
    @pytest.fixture
    def config(self):
        """创建小型 BERT 配置用于测试"""
        return BERTConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=4,
            num_layers=2,
            d_ff=256,
            max_seq_len=128,
            dropout_rate=0.0,
            num_segments=2
        )
    
    @pytest.fixture
    def model(self, config):
        """创建 BERT 模型"""
        return BERTModel(config)
    
    def test_forward_output_shape(self, model, config):
        """测试前向传播输出形状"""
        batch, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch, seq_len)
        
        outputs = model(input_ids, segment_ids, attention_mask)
        
        assert outputs['hidden_states'].shape == (batch, seq_len, config.d_model)
        assert outputs['mlm_logits'].shape == (batch, seq_len, config.vocab_size)
        assert outputs['nsp_logits'].shape == (batch, 2)
    
    def test_mlm_logits_shape(self, model, config):
        """测试 MLM logits 形状为 (batch, seq_len, vocab_size)"""
        batch, seq_len = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        
        outputs = model(input_ids, segment_ids)
        
        assert outputs['mlm_logits'].shape == (batch, seq_len, config.vocab_size)
    
    def test_nsp_logits_shape(self, model, config):
        """测试 NSP logits 形状为 (batch, 2)"""
        batch, seq_len = 4, 64
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        
        outputs = model(input_ids, segment_ids)
        
        assert outputs['nsp_logits'].shape == (batch, 2)
    
    def test_with_attention_mask(self, model, config):
        """测试带 attention mask 的前向传播"""
        batch, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        segment_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch, seq_len)
        attention_mask[0, 20:] = 0  # 第一个样本后 12 个位置是 padding
        
        outputs = model(input_ids, segment_ids, attention_mask)
        
        assert outputs['hidden_states'].shape == (batch, seq_len, config.d_model)
    
    def test_segment_embedding(self, model, config):
        """测试 segment embedding 正确应用"""
        batch, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
        
        # 不同的 segment_ids 应该产生不同的输出
        segment_ids_0 = torch.zeros(batch, seq_len, dtype=torch.long)
        segment_ids_1 = torch.ones(batch, seq_len, dtype=torch.long)
        
        model.eval()
        outputs_0 = model(input_ids, segment_ids_0)
        outputs_1 = model(input_ids, segment_ids_1)
        
        assert not torch.allclose(outputs_0['hidden_states'], outputs_1['hidden_states'])


class TestMLMHead:
    """MLM Head 测试"""
    
    def test_output_shape(self):
        """测试输出形状"""
        d_model, vocab_size = 64, 1000
        head = MLMHead(d_model, vocab_size)
        
        batch, seq_len = 2, 32
        hidden_states = torch.randn(batch, seq_len, d_model)
        
        logits = head(hidden_states)
        
        assert logits.shape == (batch, seq_len, vocab_size)


class TestNSPHead:
    """NSP Head 测试"""
    
    def test_output_shape(self):
        """测试输出形状"""
        d_model = 64
        head = NSPHead(d_model)
        
        batch = 4
        cls_hidden_state = torch.randn(batch, d_model)
        
        logits = head(cls_hidden_state)
        
        assert logits.shape == (batch, 2)
