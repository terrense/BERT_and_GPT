"""
位置编码单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.core.position import (
    SinusoidalPositionEncoding,
    LearnablePositionEmbedding
)


class TestSinusoidalPositionEncoding:
    """正弦余弦位置编码测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        d_model, max_seq_len = 64, 100
        pe = SinusoidalPositionEncoding(d_model, max_seq_len)
        
        batch, seq_len = 2, 50
        x = torch.randn(batch, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_position_encoding_range(self):
        """测试位置编码值范围在 [-1, 1]"""
        d_model, max_seq_len = 64, 100
        pe = SinusoidalPositionEncoding(d_model, max_seq_len)
        
        # 位置编码应该在 [-1, 1] 范围内（sin/cos 的值域）
        assert pe.pe.min() >= -1.0
        assert pe.pe.max() <= 1.0
    
    def test_different_positions_have_different_encodings(self):
        """测试不同位置有不同的编码"""
        d_model, max_seq_len = 64, 100
        pe = SinusoidalPositionEncoding(d_model, max_seq_len)
        
        # 位置 0 和位置 1 的编码应该不同
        assert not torch.allclose(pe.pe[0, 0, :], pe.pe[0, 1, :])
    
    def test_encoding_is_deterministic(self):
        """测试编码是确定性的（不可学习）"""
        d_model, max_seq_len = 64, 100
        pe1 = SinusoidalPositionEncoding(d_model, max_seq_len)
        pe2 = SinusoidalPositionEncoding(d_model, max_seq_len)
        
        assert torch.allclose(pe1.pe, pe2.pe)


class TestLearnablePositionEmbedding:
    """可学习位置嵌入测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        d_model, max_seq_len = 64, 100
        pe = LearnablePositionEmbedding(d_model, max_seq_len)
        
        batch, seq_len = 2, 50
        x = torch.randn(batch, seq_len, d_model)
        
        output = pe(x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_embedding_is_learnable(self):
        """测试嵌入是可学习的"""
        d_model, max_seq_len = 64, 100
        pe = LearnablePositionEmbedding(d_model, max_seq_len)
        
        # 检查参数需要梯度
        assert pe.position_embedding.weight.requires_grad
    
    def test_different_initializations(self):
        """测试不同实例有不同的初始化"""
        d_model, max_seq_len = 64, 100
        pe1 = LearnablePositionEmbedding(d_model, max_seq_len)
        pe2 = LearnablePositionEmbedding(d_model, max_seq_len)
        
        # 随机初始化应该不同
        assert not torch.allclose(
            pe1.position_embedding.weight, 
            pe2.position_embedding.weight
        )
