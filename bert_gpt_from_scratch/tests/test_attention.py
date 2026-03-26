"""
Multi-Head Attention 单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.core.attention import (
    scaled_dot_product_attention,
    MultiHeadAttention
)


class TestScaledDotProductAttention:
    """Scaled Dot-Product Attention 测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        batch, heads, seq_len, d_k = 2, 4, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)
        
        output, weights = scaled_dot_product_attention(q, k, v)
        
        assert output.shape == (batch, heads, seq_len, d_k)
        assert weights.shape == (batch, heads, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """测试注意力权重和为 1"""
        batch, heads, seq_len, d_k = 2, 4, 10, 64
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)
        
        _, weights = scaled_dot_product_attention(q, k, v)
        
        # 每行的权重和应该为 1
        row_sums = weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    
    def test_mask_application(self):
        """测试掩码正确应用"""
        batch, heads, seq_len, d_k = 1, 1, 4, 8
        q = torch.randn(batch, heads, seq_len, d_k)
        k = torch.randn(batch, heads, seq_len, d_k)
        v = torch.randn(batch, heads, seq_len, d_k)
        
        # 创建掩码：掩盖最后两个位置
        mask = torch.zeros(batch, 1, 1, seq_len)
        mask[:, :, :, 2:] = 1
        
        _, weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 被掩码位置的权重应该接近 0
        assert torch.allclose(weights[:, :, :, 2:], torch.zeros_like(weights[:, :, :, 2:]), atol=1e-5)


class TestMultiHeadAttention:
    """Multi-Head Attention 测试"""
    
    @pytest.fixture
    def mha(self):
        """创建 MHA 实例"""
        return MultiHeadAttention(d_model=64, num_heads=4, dropout_rate=0.0)
    
    def test_output_shape(self, mha):
        """测试输出形状正确性"""
        batch, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch, seq_len, d_model)
        
        output = mha(x, x, x)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_causal_mask_blocks_future(self):
        """测试 causal mask 阻止未来位置信息"""
        d_model, num_heads = 64, 4
        mha = MultiHeadAttention(d_model, num_heads, dropout_rate=0.0)
        
        batch, seq_len = 1, 5
        x = torch.randn(batch, seq_len, d_model)
        
        # 创建 causal mask
        causal_mask = MultiHeadAttention.create_causal_mask(seq_len)
        
        # 使用 causal mask
        output_masked = mha(x, x, x, mask=causal_mask)
        
        # 输出形状应该正确
        assert output_masked.shape == (batch, seq_len, d_model)
    
    def test_padding_mask(self, mha):
        """测试 padding mask 正确应用"""
        batch, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch, seq_len, d_model)
        
        # 创建 padding mask：第一个样本最后 3 个位置是 padding
        padding_mask = torch.zeros(batch, seq_len)
        padding_mask[0, 7:] = 1
        
        output = mha(x, x, x, mask=padding_mask)
        
        assert output.shape == (batch, seq_len, d_model)
    
    def test_causal_mask_creation(self):
        """测试 causal mask 创建"""
        seq_len = 5
        mask = MultiHeadAttention.create_causal_mask(seq_len)
        
        # 应该是上三角矩阵
        assert mask.shape == (seq_len, seq_len)
        assert mask[0, 0] == False  # 对角线为 False
        assert mask[0, 1] == True   # 上三角为 True
        assert mask[4, 0] == False  # 下三角为 False
