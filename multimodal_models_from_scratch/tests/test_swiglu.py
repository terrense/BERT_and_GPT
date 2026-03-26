"""
SwiGLU 模块单元测试

测试 SwiGLU 激活函数的前馈网络实现。

需求: 7.3, 10.3
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.llm.swiglu import SwiGLU


class TestSwiGLU:
    """SwiGLU 单元测试"""
    
    def test_output_shape(self):
        """测试输出形状正确性"""
        batch_size = 2
        seq_len = 10
        d_model = 64
        d_ff = 256
        
        swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = swiglu(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_output_shape_various_sizes(self):
        """测试不同输入尺寸的输出形状"""
        test_cases = [
            (1, 1, 32, 128),      # 最小输入
            (4, 128, 256, 1024),  # 较大输入
            (8, 512, 768, 3072),  # 类似 BERT-base 配置
        ]
        
        for batch_size, seq_len, d_model, d_ff in test_cases:
            swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
            x = torch.randn(batch_size, seq_len, d_model)
            
            output = swiglu(x)
            
            assert output.shape == (batch_size, seq_len, d_model), \
                f"Failed for batch={batch_size}, seq={seq_len}, d_model={d_model}, d_ff={d_ff}"
    
    def test_no_bias_in_linear_layers(self):
        """测试线性层没有偏置（符合 LLaMA 设计）"""
        swiglu = SwiGLU(d_model=64, d_ff=256)
        
        assert swiglu.w_gate.bias is None
        assert swiglu.w_up.bias is None
        assert swiglu.w_down.bias is None
    
    def test_dropout_applied(self):
        """测试 dropout 被正确应用"""
        swiglu = SwiGLU(d_model=64, d_ff=256, dropout_rate=0.5)
        x = torch.randn(4, 10, 64)
        
        # 训练模式下，dropout 应该有效果
        swiglu.train()
        outputs_train = [swiglu(x) for _ in range(5)]
        
        # 检查多次前向传播结果不完全相同（因为 dropout）
        all_same = all(torch.allclose(outputs_train[0], out) for out in outputs_train[1:])
        assert not all_same, "Dropout should cause different outputs in training mode"
        
        # 评估模式下，dropout 应该被禁用
        swiglu.eval()
        output1 = swiglu(x)
        output2 = swiglu(x)
        
        assert torch.allclose(output1, output2), "Outputs should be identical in eval mode"
    
    def test_no_dropout_when_rate_zero(self):
        """测试 dropout_rate=0 时没有 dropout"""
        swiglu = SwiGLU(d_model=64, d_ff=256, dropout_rate=0.0)
        x = torch.randn(4, 10, 64)
        
        swiglu.train()
        output1 = swiglu(x)
        output2 = swiglu(x)
        
        assert torch.allclose(output1, output2), "Outputs should be identical when dropout_rate=0"
    
    def test_gradient_flow(self):
        """测试梯度能够正确流动"""
        swiglu = SwiGLU(d_model=64, d_ff=256)
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        output = swiglu(x)
        loss = output.sum()
        loss.backward()
        
        # 检查输入梯度存在
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # 检查所有参数都有梯度
        for name, param in swiglu.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
    
    def test_swish_activation_behavior(self):
        """测试 Swish 激活函数的行为"""
        # Swish(x) = x * sigmoid(x)
        # 对于正数，Swish 接近 x
        # 对于负数，Swish 接近 0 但不完全为 0
        
        swiglu = SwiGLU(d_model=64, d_ff=256)
        
        # 创建一个简单的输入
        x = torch.randn(1, 1, 64)
        
        # 前向传播应该成功
        output = swiglu(x)
        
        # 输出应该是有限值
        assert torch.isfinite(output).all()
    
    def test_deterministic_output(self):
        """测试相同输入产生相同输出（eval 模式）"""
        swiglu = SwiGLU(d_model=64, d_ff=256, dropout_rate=0.1)
        swiglu.eval()
        
        x = torch.randn(2, 10, 64)
        
        output1 = swiglu(x)
        output2 = swiglu(x)
        
        assert torch.allclose(output1, output2)
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        swiglu = SwiGLU(d_model=64, d_ff=256)
        repr_str = swiglu.extra_repr()
        
        assert 'd_model=64' in repr_str
        assert 'd_ff=256' in repr_str
    
    def test_parameter_count(self):
        """测试参数数量正确"""
        d_model = 64
        d_ff = 256
        
        swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
        
        # 计算预期参数数量
        # w_gate: d_model * d_ff (无偏置)
        # w_up: d_model * d_ff (无偏置)
        # w_down: d_ff * d_model (无偏置)
        expected_params = d_model * d_ff + d_model * d_ff + d_ff * d_model
        
        actual_params = sum(p.numel() for p in swiglu.parameters())
        
        assert actual_params == expected_params, \
            f"Expected {expected_params} parameters, got {actual_params}"
    
    def test_llama_like_configuration(self):
        """测试类似 LLaMA 的配置"""
        # LLaMA-7B 配置: d_model=4096, d_ff=11008
        # 这里使用缩小版本进行测试
        d_model = 256
        d_ff = 688  # 约 2.7 * d_model，类似 LLaMA 的比例
        
        swiglu = SwiGLU(d_model=d_model, d_ff=d_ff)
        x = torch.randn(2, 32, d_model)
        
        output = swiglu(x)
        
        assert output.shape == (2, 32, d_model)
        assert torch.isfinite(output).all()


class TestSwiGLUIntegration:
    """SwiGLU 集成测试"""
    
    def test_with_rmsnorm(self):
        """测试与 RMSNorm 的集成"""
        from multimodal_models_from_scratch.llm.rmsnorm import RMSNorm
        
        d_model = 64
        d_ff = 256
        
        rmsnorm = RMSNorm(d_model)
        swiglu = SwiGLU(d_model, d_ff)
        
        x = torch.randn(2, 10, d_model)
        
        # Pre-Norm 架构: x + SwiGLU(RMSNorm(x))
        normalized = rmsnorm(x)
        ffn_output = swiglu(normalized)
        output = x + ffn_output
        
        assert output.shape == x.shape
        assert torch.isfinite(output).all()
    
    def test_sequential_layers(self):
        """测试多层堆叠"""
        d_model = 64
        d_ff = 256
        num_layers = 4
        
        layers = nn.ModuleList([
            SwiGLU(d_model, d_ff) for _ in range(num_layers)
        ])
        
        x = torch.randn(2, 10, d_model)
        
        for layer in layers:
            x = x + layer(x)  # 残差连接
        
        assert x.shape == (2, 10, d_model)
        assert torch.isfinite(x).all()
