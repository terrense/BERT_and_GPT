"""
FeedForwardNetwork 单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.core.feedforward import FeedForwardNetwork


class TestFeedForwardNetwork:
    """FeedForwardNetwork 测试"""
    
    def test_output_shape(self):
        """测试输出形状与输入形状一致"""
        d_model, d_ff = 64, 256
        ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate=0.0)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output = ffn(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_dropout_training_mode(self):
        """测试训练模式下 dropout 生效"""
        d_model, d_ff = 64, 256
        ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate=0.5)
        ffn.train()
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        # 多次前向传播，结果应该不同（因为 dropout）
        torch.manual_seed(42)
        output1 = ffn(x)
        torch.manual_seed(43)
        output2 = ffn(x)
        
        assert not torch.allclose(output1, output2)
    
    def test_dropout_eval_mode(self):
        """测试推理模式下 dropout 不生效"""
        d_model, d_ff = 64, 256
        ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate=0.5)
        ffn.eval()
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        # 推理模式下，多次前向传播结果应该相同
        output1 = ffn(x)
        output2 = ffn(x)
        
        assert torch.allclose(output1, output2)
    
    def test_zero_dropout(self):
        """测试 dropout=0 时结果确定"""
        d_model, d_ff = 64, 256
        ffn = FeedForwardNetwork(d_model, d_ff, dropout_rate=0.0)
        ffn.train()
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output1 = ffn(x)
        output2 = ffn(x)
        
        assert torch.allclose(output1, output2)
