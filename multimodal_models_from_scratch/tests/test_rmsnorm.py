"""
RMSNorm 单元测试

测试 Root Mean Square Layer Normalization 的正确性
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.llm.rmsnorm import RMSNorm


class TestRMSNorm:
    """RMSNorm 测试类"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        batch_size = 2
        seq_len = 10
        d_model = 64
        
        rmsnorm = RMSNorm(d_model=d_model)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = rmsnorm(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_output_shape_various_sizes(self):
        """测试不同输入尺寸的输出形状"""
        test_cases = [
            (1, 1, 32),
            (4, 128, 256),
            (8, 512, 768),
        ]
        
        for batch_size, seq_len, d_model in test_cases:
            rmsnorm = RMSNorm(d_model=d_model)
            x = torch.randn(batch_size, seq_len, d_model)
            output = rmsnorm(x)
            
            assert output.shape == (batch_size, seq_len, d_model), \
                f"Failed for shape ({batch_size}, {seq_len}, {d_model})"
    
    def test_weight_initialization(self):
        """测试权重初始化为 1"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        assert rmsnorm.weight.shape == (d_model,)
        assert torch.allclose(rmsnorm.weight, torch.ones(d_model))
    
    def test_formula_correctness(self):
        """测试 RMSNorm 公式正确性: x * weight / sqrt(mean(x^2) + eps)"""
        d_model = 4
        eps = 1e-6
        
        rmsnorm = RMSNorm(d_model=d_model, eps=eps)
        # 设置权重为已知值
        rmsnorm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])  # (1, 1, 4)
        
        # 手动计算期望值
        # mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
        # rms = sqrt(7.5 + eps) ≈ 2.7386
        # x_normalized = x / rms
        # output = weight * x_normalized
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(variance + eps)
        expected = rmsnorm.weight * (x / rms)
        
        output = rmsnorm(x)
        
        assert torch.allclose(output, expected, atol=1e-5)
    
    def test_numerical_stability(self):
        """测试数值稳定性（小值输入）"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        # 非常小的输入值
        x = torch.randn(2, 10, d_model) * 1e-8
        output = rmsnorm(x)
        
        # 输出不应包含 NaN 或 Inf
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_gradient_flow(self):
        """测试梯度能够正确传播"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        x = torch.randn(2, 10, d_model, requires_grad=True)
        output = rmsnorm(x)
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert x.grad is not None
        assert rmsnorm.weight.grad is not None
        
        # 检查梯度不为零
        assert x.grad.abs().sum() > 0
        assert rmsnorm.weight.grad.abs().sum() > 0
    
    def test_different_eps_values(self):
        """测试不同 eps 值"""
        d_model = 64
        x = torch.randn(2, 10, d_model)
        
        eps_values = [1e-8, 1e-6, 1e-4]
        outputs = []
        
        for eps in eps_values:
            rmsnorm = RMSNorm(d_model=d_model, eps=eps)
            outputs.append(rmsnorm(x))
        
        # 不同 eps 应该产生略微不同的输出
        # 但差异应该很小
        for i in range(len(outputs) - 1):
            diff = (outputs[i] - outputs[i + 1]).abs().max()
            assert diff < 0.01, f"Difference too large between eps={eps_values[i]} and eps={eps_values[i+1]}"
    
    def test_learnable_weight(self):
        """测试权重是可学习的"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        # 检查权重是 Parameter
        assert isinstance(rmsnorm.weight, nn.Parameter)
        assert rmsnorm.weight.requires_grad
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        d_model = 128
        eps = 1e-5
        rmsnorm = RMSNorm(d_model=d_model, eps=eps)
        
        repr_str = rmsnorm.extra_repr()
        assert 'd_model=128' in repr_str
        assert 'eps=1e-05' in repr_str
    
    def test_batch_independence(self):
        """测试批次之间的独立性"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        x1 = torch.randn(1, 10, d_model)
        x2 = torch.randn(1, 10, d_model)
        
        # 分别处理
        out1 = rmsnorm(x1)
        out2 = rmsnorm(x2)
        
        # 合并处理
        x_combined = torch.cat([x1, x2], dim=0)
        out_combined = rmsnorm(x_combined)
        
        # 结果应该相同
        assert torch.allclose(out1, out_combined[0:1], atol=1e-6)
        assert torch.allclose(out2, out_combined[1:2], atol=1e-6)
    
    def test_sequence_independence(self):
        """测试序列位置之间的独立性"""
        d_model = 64
        rmsnorm = RMSNorm(d_model=d_model)
        
        x = torch.randn(2, 10, d_model)
        output = rmsnorm(x)
        
        # 单独处理每个位置
        for i in range(10):
            x_single = x[:, i:i+1, :]
            out_single = rmsnorm(x_single)
            assert torch.allclose(out_single, output[:, i:i+1, :], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
