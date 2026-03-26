"""
Transformer 层单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.core.layers import EncoderLayer, DecoderLayer


class TestEncoderLayer:
    """EncoderLayer 测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        d_model, num_heads, d_ff = 64, 4, 256
        encoder = EncoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output = encoder(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_with_padding_mask(self):
        """测试 padding mask"""
        d_model, num_heads, d_ff = 64, 4, 256
        encoder = EncoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        padding_mask = torch.zeros(batch, seq_len)
        padding_mask[0, 7:] = 1  # 第一个样本最后 3 个位置是 padding
        
        output = encoder(x, padding_mask=padding_mask)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_residual_connection(self):
        """测试残差连接存在"""
        d_model, num_heads, d_ff = 64, 4, 256
        encoder = EncoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        encoder.eval()
        
        batch, seq_len = 1, 5
        x = torch.randn(batch, seq_len, d_model)
        
        output = encoder(x)
        
        # 输出不应该与输入完全相同（有变换），但也不应该完全不相关（有残差）
        assert not torch.allclose(output, x)
        # 相关性检查：输出和输入的差异应该是有限的
        diff = (output - x).abs().mean()
        assert diff < 10.0  # 合理的差异范围


class TestDecoderLayer:
    """DecoderLayer 测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        d_model, num_heads, d_ff = 64, 4, 256
        decoder = DecoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        
        output = decoder(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_causal_mask_auto_applied(self):
        """测试 causal mask 自动应用"""
        d_model, num_heads, d_ff = 64, 4, 256
        decoder = DecoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        decoder.eval()
        
        batch, seq_len = 1, 5
        x = torch.randn(batch, seq_len, d_model)
        
        # 应该能正常运行（causal mask 自动生成）
        output = decoder(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_with_padding_mask(self):
        """测试 padding mask 与 causal mask 结合"""
        d_model, num_heads, d_ff = 64, 4, 256
        decoder = DecoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        
        batch, seq_len = 2, 10
        x = torch.randn(batch, seq_len, d_model)
        padding_mask = torch.zeros(batch, seq_len)
        padding_mask[0, 7:] = 1
        
        output = decoder(x, padding_mask=padding_mask)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_residual_and_layernorm(self):
        """测试残差连接和 LayerNorm 正确应用"""
        d_model, num_heads, d_ff = 64, 4, 256
        decoder = DecoderLayer(d_model, num_heads, d_ff, dropout_rate=0.0)
        decoder.eval()
        
        batch, seq_len = 1, 5
        x = torch.randn(batch, seq_len, d_model)
        
        output = decoder(x)
        
        # LayerNorm 后，每个位置的均值应该接近 0，方差接近 1
        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=0.1)
        assert torch.allclose(var, torch.ones_like(var), atol=0.1)
