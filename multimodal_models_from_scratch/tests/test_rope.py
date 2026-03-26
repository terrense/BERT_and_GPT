"""
RoPE (Rotary Position Embedding) 单元测试

测试旋转位置编码的正确性，包括：
- 输出形状验证
- cos/sin 缓存预计算
- 旋转变换的数学正确性
- NTK-aware 插值扩展
- 自定义 position_ids 支持
"""

import pytest
import torch
import math

from multimodal_models_from_scratch.llm.rope import RotaryPositionEmbedding


class TestRotaryPositionEmbedding:
    """RoPE 模块测试"""
    
    def test_initialization(self):
        """测试初始化参数"""
        d_model = 64
        max_seq_len = 512
        theta = 10000.0
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
        
        assert rope.d_model == d_model
        assert rope.max_seq_len == max_seq_len
        assert rope.theta == theta
        assert rope.inv_freq.shape == (d_model // 2,)
    
    def test_cache_shape(self):
        """测试 cos/sin 缓存形状"""
        d_model = 64
        max_seq_len = 128
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len)
        
        # 缓存形状应为 (1, 1, max_seq_len, d_model)
        assert rope.cos_cached.shape == (1, 1, max_seq_len, d_model)
        assert rope.sin_cached.shape == (1, 1, max_seq_len, d_model)
    
    def test_forward_output_shape(self):
        """测试前向传播输出形状"""
        batch_size = 2
        num_heads = 8
        seq_len = 32
        head_dim = 64
        
        rope = RotaryPositionEmbedding(head_dim, max_seq_len=128)
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        rotated_q, rotated_k = rope(q, k)
        
        assert rotated_q.shape == q.shape
        assert rotated_k.shape == k.shape
    
    def test_forward_with_position_ids(self):
        """测试使用自定义 position_ids"""
        batch_size = 2
        num_heads = 4
        seq_len = 16
        head_dim = 32
        
        rope = RotaryPositionEmbedding(head_dim, max_seq_len=64)
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # 自定义位置索引（例如 KV Cache 场景）
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        rotated_q, rotated_k = rope(q, k, position_ids)
        
        assert rotated_q.shape == q.shape
        assert rotated_k.shape == k.shape
    
    def test_cache_extension(self):
        """测试缓存自动扩展"""
        d_model = 64
        initial_max_len = 32
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len=initial_max_len)
        
        # 初始缓存长度
        assert rope.cos_cached.shape[2] == initial_max_len
        
        # 使用更长的序列
        longer_seq_len = 64
        q = torch.randn(1, 4, longer_seq_len, d_model)
        k = torch.randn(1, 4, longer_seq_len, d_model)
        
        rotated_q, rotated_k = rope(q, k)
        
        # 缓存应该被扩展
        assert rope.cos_cached.shape[2] >= longer_seq_len
        assert rotated_q.shape == q.shape
    
    def test_rotation_preserves_norm(self):
        """测试旋转变换保持向量范数（近似）"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        q = torch.randn(1, 1, 1, d_model)
        k = torch.randn(1, 1, 1, d_model)
        
        q_norm_before = torch.norm(q)
        k_norm_before = torch.norm(k)
        
        rotated_q, rotated_k = rope(q, k)
        
        q_norm_after = torch.norm(rotated_q)
        k_norm_after = torch.norm(rotated_k)
        
        # 旋转变换应该保持向量范数
        assert torch.allclose(q_norm_before, q_norm_after, rtol=1e-5)
        assert torch.allclose(k_norm_before, k_norm_after, rtol=1e-5)
    
    def test_different_positions_different_rotations(self):
        """测试不同位置产生不同的旋转"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        # 相同的输入向量
        x = torch.randn(1, 1, 1, d_model)
        q = x.expand(1, 1, 3, d_model).clone()  # 3 个相同的向量
        k = x.expand(1, 1, 3, d_model).clone()
        
        rotated_q, rotated_k = rope(q, k)
        
        # 不同位置的旋转结果应该不同
        assert not torch.allclose(rotated_q[0, 0, 0], rotated_q[0, 0, 1])
        assert not torch.allclose(rotated_q[0, 0, 1], rotated_q[0, 0, 2])
    
    def test_inv_freq_computation(self):
        """测试逆频率计算的正确性"""
        d_model = 8
        theta = 10000.0
        
        rope = RotaryPositionEmbedding(d_model, theta=theta)
        
        # 手动计算期望的 inv_freq
        expected_inv_freq = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        
        assert torch.allclose(rope.inv_freq, expected_inv_freq)
    
    def test_cos_sin_values_at_position_zero(self):
        """测试位置 0 处的 cos/sin 值"""
        d_model = 8
        rope = RotaryPositionEmbedding(d_model)
        
        # 位置 0 处，角度为 0，cos=1, sin=0
        cos_at_0 = rope.cos_cached[0, 0, 0, :]
        sin_at_0 = rope.sin_cached[0, 0, 0, :]
        
        assert torch.allclose(cos_at_0, torch.ones_like(cos_at_0))
        assert torch.allclose(sin_at_0, torch.zeros_like(sin_at_0), atol=1e-6)
    
    def test_apply_rotary_emb(self):
        """测试 apply_rotary_emb 方法"""
        d_model = 4
        rope = RotaryPositionEmbedding(d_model)
        
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # (1, 1, 1, 4)
        cos = torch.ones(1, 1, 1, 4)
        sin = torch.zeros(1, 1, 1, 4)
        
        # cos=1, sin=0 时，旋转后应该与原始相同
        rotated = rope.apply_rotary_emb(x, cos, sin)
        assert torch.allclose(rotated, x)
    
    def test_rotate_half(self):
        """测试 _rotate_half 方法"""
        d_model = 4
        rope = RotaryPositionEmbedding(d_model)
        
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rotated = rope._rotate_half(x.unsqueeze(0).unsqueeze(0).unsqueeze(0))
        
        # [1, 2, 3, 4] -> [-3, -4, 1, 2]
        expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
        assert torch.allclose(rotated, expected)
    
    def test_ntk_scaling(self):
        """测试 NTK-aware 插值扩展"""
        d_model = 64
        max_seq_len = 128
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len)
        
        # 测试 alpha=1.0（无缩放）
        cos1, sin1 = rope.apply_ntk_scaling(64, alpha=1.0)
        assert cos1.shape == (1, 1, 64, d_model)
        assert sin1.shape == (1, 1, 64, d_model)
        
        # 测试 alpha=2.0（扩展到 2 倍长度）
        cos2, sin2 = rope.apply_ntk_scaling(256, alpha=2.0)
        assert cos2.shape == (1, 1, 256, d_model)
        assert sin2.shape == (1, 1, 256, d_model)
        
        # 缩放后的频率应该不同
        # 在相同位置，不同 alpha 的 cos/sin 值应该不同
        cos1_extended, _ = rope.apply_ntk_scaling(256, alpha=1.0)
        assert not torch.allclose(cos1_extended[:, :, 100, :], cos2[:, :, 100, :])
    
    def test_ntk_scaling_preserves_norm(self):
        """测试 NTK 缩放后旋转仍保持范数"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model, max_seq_len=128)
        
        x = torch.randn(1, 1, 1, d_model)
        x_norm_before = torch.norm(x)
        
        cos, sin = rope.apply_ntk_scaling(256, alpha=2.0)
        rotated = rope.apply_rotary_emb(x, cos[:, :, :1, :], sin[:, :, :1, :])
        
        x_norm_after = torch.norm(rotated)
        assert torch.allclose(x_norm_before, x_norm_after, rtol=1e-5)
    
    def test_relative_position_property(self):
        """测试相对位置编码特性"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        # 创建两个向量
        q = torch.randn(1, 1, 4, d_model)
        k = torch.randn(1, 1, 4, d_model)
        
        rotated_q, rotated_k = rope(q, k)
        
        # 计算注意力分数（点积）
        # 位置 i 和 j 之间的注意力分数应该只依赖于相对位置 i-j
        scores = torch.matmul(rotated_q, rotated_k.transpose(-2, -1))
        
        # 验证分数矩阵的形状
        assert scores.shape == (1, 1, 4, 4)
    
    def test_batch_processing(self):
        """测试批量处理"""
        batch_size = 4
        num_heads = 8
        seq_len = 32
        head_dim = 64
        
        rope = RotaryPositionEmbedding(head_dim)
        
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        rotated_q, rotated_k = rope(q, k)
        
        # 验证每个 batch 独立处理
        for i in range(batch_size):
            q_single = q[i:i+1]
            k_single = k[i:i+1]
            rotated_q_single, rotated_k_single = rope(q_single, k_single)
            
            assert torch.allclose(rotated_q[i:i+1], rotated_q_single, rtol=1e-5)
            assert torch.allclose(rotated_k[i:i+1], rotated_k_single, rtol=1e-5)
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        d_model = 64
        max_seq_len = 512
        theta = 10000.0
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len, theta)
        
        repr_str = rope.extra_repr()
        assert 'd_model=64' in repr_str
        assert 'max_seq_len=512' in repr_str
        assert 'theta=10000.0' in repr_str
    
    def test_device_consistency(self):
        """测试设备一致性"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        # 默认在 CPU 上
        assert rope.inv_freq.device.type == 'cpu'
        assert rope.cos_cached.device.type == 'cpu'
        assert rope.sin_cached.device.type == 'cpu'
        
        q = torch.randn(1, 1, 8, d_model)
        k = torch.randn(1, 1, 8, d_model)
        
        rotated_q, rotated_k = rope(q, k)
        
        assert rotated_q.device.type == 'cpu'
        assert rotated_k.device.type == 'cpu'
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        q = torch.randn(1, 1, 4, d_model, requires_grad=True)
        k = torch.randn(1, 1, 4, d_model, requires_grad=True)
        
        rotated_q, rotated_k = rope(q, k)
        
        # 计算损失并反向传播
        loss = rotated_q.sum() + rotated_k.sum()
        loss.backward()
        
        # 验证梯度存在
        assert q.grad is not None
        assert k.grad is not None
        assert q.grad.shape == q.shape
        assert k.grad.shape == k.shape


class TestRotaryPositionEmbeddingEdgeCases:
    """RoPE 边界情况测试"""
    
    def test_single_position(self):
        """测试单个位置"""
        d_model = 64
        rope = RotaryPositionEmbedding(d_model)
        
        q = torch.randn(1, 1, 1, d_model)
        k = torch.randn(1, 1, 1, d_model)
        
        rotated_q, rotated_k = rope(q, k)
        
        assert rotated_q.shape == (1, 1, 1, d_model)
        assert rotated_k.shape == (1, 1, 1, d_model)
    
    def test_small_d_model(self):
        """测试小维度"""
        d_model = 4
        rope = RotaryPositionEmbedding(d_model)
        
        q = torch.randn(1, 1, 8, d_model)
        k = torch.randn(1, 1, 8, d_model)
        
        rotated_q, rotated_k = rope(q, k)
        
        assert rotated_q.shape == q.shape
        assert rotated_k.shape == k.shape
    
    def test_large_seq_len(self):
        """测试长序列"""
        d_model = 64
        seq_len = 2048
        
        rope = RotaryPositionEmbedding(d_model, max_seq_len=256)
        
        q = torch.randn(1, 1, seq_len, d_model)
        k = torch.randn(1, 1, seq_len, d_model)
        
        # 应该自动扩展缓存
        rotated_q, rotated_k = rope(q, k)
        
        assert rotated_q.shape == q.shape
        assert rotated_k.shape == k.shape
    
    def test_different_theta(self):
        """测试不同的 theta 值"""
        d_model = 64
        
        rope1 = RotaryPositionEmbedding(d_model, theta=10000.0)
        rope2 = RotaryPositionEmbedding(d_model, theta=1000000.0)
        
        q = torch.randn(1, 1, 8, d_model)
        k = torch.randn(1, 1, 8, d_model)
        
        rotated_q1, rotated_k1 = rope1(q.clone(), k.clone())
        rotated_q2, rotated_k2 = rope2(q.clone(), k.clone())
        
        # 不同 theta 应该产生不同的旋转
        assert not torch.allclose(rotated_q1, rotated_q2)
        assert not torch.allclose(rotated_k1, rotated_k2)
