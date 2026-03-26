"""
Visual Projection 模块单元测试

测试 VisualProjection 类的线性投影和 MLP 投影两种模式。

需求: 4.3, 6.6, 9.2
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.multimodal.visual_projection import VisualProjection


class TestLinearProjection:
    """测试线性投影模式"""
    
    def test_linear_projection_output_shape(self):
        """测试线性投影输出形状: (batch, num_tokens, llm_dim)"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        visual_features = torch.randn(2, 196, 768)
        output = proj(visual_features)
        
        assert output.shape == (2, 196, 4096)
    
    def test_linear_projection_single_token(self):
        """测试单个 token 的线性投影"""
        proj = VisualProjection(
            vision_dim=512,
            llm_dim=2048,
            projection_type='linear'
        )
        
        visual_features = torch.randn(1, 1, 512)
        output = proj(visual_features)
        
        assert output.shape == (1, 1, 2048)
    
    def test_linear_projection_is_single_layer(self):
        """测试线性投影是单层 Linear"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        assert isinstance(proj.projection, nn.Linear)
        assert proj.projection.in_features == 768
        assert proj.projection.out_features == 4096
    
    def test_linear_projection_different_dims(self):
        """测试不同维度配置的线性投影"""
        test_cases = [
            (256, 512),
            (512, 1024),
            (768, 4096),
            (1024, 4096),
            (1280, 5120),
        ]
        
        for vision_dim, llm_dim in test_cases:
            proj = VisualProjection(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                projection_type='linear'
            )
            
            visual_features = torch.randn(1, 100, vision_dim)
            output = proj(visual_features)
            
            assert output.shape == (1, 100, llm_dim)


class TestMLPProjection:
    """测试 MLP 投影模式"""
    
    def test_mlp_projection_output_shape(self):
        """测试 MLP 投影输出形状: (batch, num_tokens, llm_dim)"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        visual_features = torch.randn(2, 196, 768)
        output = proj(visual_features)
        
        assert output.shape == (2, 196, 4096)
    
    def test_mlp_projection_default_depth(self):
        """测试 MLP 投影默认深度为 2"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        assert proj.mlp_depth == 2
    
    def test_mlp_projection_structure_depth_2(self):
        """测试 depth=2 的 MLP 结构: Linear -> GELU -> Linear"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=2
        )
        
        assert isinstance(proj.projection, nn.Sequential)
        assert len(proj.projection) == 3  # Linear, GELU, Linear
        
        assert isinstance(proj.projection[0], nn.Linear)
        assert proj.projection[0].in_features == 768
        assert proj.projection[0].out_features == 4096
        
        assert isinstance(proj.projection[1], nn.GELU)
        
        assert isinstance(proj.projection[2], nn.Linear)
        assert proj.projection[2].in_features == 4096
        assert proj.projection[2].out_features == 4096
    
    def test_mlp_projection_structure_depth_3(self):
        """测试 depth=3 的 MLP 结构: Linear -> GELU -> Linear -> GELU -> Linear"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=3
        )
        
        assert isinstance(proj.projection, nn.Sequential)
        assert len(proj.projection) == 5  # Linear, GELU, Linear, GELU, Linear
        
        # 第一层
        assert isinstance(proj.projection[0], nn.Linear)
        assert proj.projection[0].in_features == 768
        assert proj.projection[0].out_features == 4096
        
        assert isinstance(proj.projection[1], nn.GELU)
        
        # 中间层
        assert isinstance(proj.projection[2], nn.Linear)
        assert proj.projection[2].in_features == 4096
        assert proj.projection[2].out_features == 4096
        
        assert isinstance(proj.projection[3], nn.GELU)
        
        # 最后一层
        assert isinstance(proj.projection[4], nn.Linear)
        assert proj.projection[4].in_features == 4096
        assert proj.projection[4].out_features == 4096
    
    def test_mlp_projection_custom_depth(self):
        """测试自定义深度的 MLP 投影"""
        for depth in [2, 3, 4, 5]:
            proj = VisualProjection(
                vision_dim=768,
                llm_dim=4096,
                projection_type='mlp',
                mlp_depth=depth
            )
            
            visual_features = torch.randn(1, 100, 768)
            output = proj(visual_features)
            
            assert output.shape == (1, 100, 4096)
            assert proj.mlp_depth == depth
    
    def test_mlp_projection_different_dims(self):
        """测试不同维度配置的 MLP 投影"""
        test_cases = [
            (256, 512),
            (512, 1024),
            (768, 4096),
            (1024, 4096),
            (1280, 5120),
        ]
        
        for vision_dim, llm_dim in test_cases:
            proj = VisualProjection(
                vision_dim=vision_dim,
                llm_dim=llm_dim,
                projection_type='mlp'
            )
            
            visual_features = torch.randn(1, 100, vision_dim)
            output = proj(visual_features)
            
            assert output.shape == (1, 100, llm_dim)


class TestProjectionTypeValidation:
    """测试投影类型验证"""
    
    def test_invalid_projection_type_raises_error(self):
        """测试无效的投影类型抛出错误"""
        with pytest.raises(ValueError, match="Unknown projection_type"):
            VisualProjection(
                vision_dim=768,
                llm_dim=4096,
                projection_type='invalid'
            )
    
    def test_mlp_depth_less_than_2_raises_error(self):
        """测试 mlp_depth < 2 抛出错误"""
        with pytest.raises(ValueError, match="mlp_depth must be at least 2"):
            VisualProjection(
                vision_dim=768,
                llm_dim=4096,
                projection_type='mlp',
                mlp_depth=1
            )


class TestBatchProcessing:
    """测试批量处理"""
    
    def test_single_sample(self):
        """测试单个样本处理"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        visual_features = torch.randn(1, 196, 768)
        output = proj(visual_features)
        
        assert output.shape == (1, 196, 4096)
    
    def test_batch_processing(self):
        """测试批量样本处理"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        batch_sizes = [2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            visual_features = torch.randn(batch_size, 196, 768)
            output = proj(visual_features)
            
            assert output.shape == (batch_size, 196, 4096)
    
    def test_batch_consistency(self):
        """测试批量处理的一致性"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        proj.eval()
        
        # 创建相同的输入
        x1 = torch.randn(1, 196, 768)
        x2 = x1.clone()
        
        # 单独处理
        with torch.no_grad():
            output1 = proj(x1)
            output2 = proj(x2)
        
        # 结果应该相同
        assert torch.allclose(output1, output2)
    
    def test_batch_vs_individual_processing(self):
        """测试批量处理与单独处理的一致性"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        proj.eval()
        
        # 创建两个不同的输入
        x1 = torch.randn(1, 196, 768)
        x2 = torch.randn(1, 196, 768)
        
        # 批量处理
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = proj(x_batch)
        
        # 单独处理
        with torch.no_grad():
            output1 = proj(x1)
            output2 = proj(x2)
        
        # 批量处理的结果应该与单独处理一致
        assert torch.allclose(output_batch[0], output1[0], atol=1e-5)
        assert torch.allclose(output_batch[1], output2[0], atol=1e-5)


class TestDifferentTokenCounts:
    """测试不同 token 数量"""
    
    def test_different_num_tokens(self):
        """测试不同数量的视觉 token"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        # 不同的 token 数量（对应不同的图像尺寸和 patch 大小）
        token_counts = [49, 196, 256, 576, 1024]
        
        for num_tokens in token_counts:
            visual_features = torch.randn(2, num_tokens, 768)
            output = proj(visual_features)
            
            assert output.shape == (2, num_tokens, 4096)


class TestGradientFlow:
    """测试梯度流动"""
    
    def test_gradients_flow_through_linear(self):
        """测试梯度能够正确流过线性投影"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        visual_features = torch.randn(2, 196, 768, requires_grad=True)
        output = proj(visual_features)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert visual_features.grad is not None
    
    def test_gradients_flow_through_mlp(self):
        """测试梯度能够正确流过 MLP 投影"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        visual_features = torch.randn(2, 196, 768, requires_grad=True)
        output = proj(visual_features)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert visual_features.grad is not None
    
    def test_all_parameters_have_gradients_linear(self):
        """测试线性投影所有参数都有梯度"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        visual_features = torch.randn(1, 196, 768)
        output = proj(visual_features)
        loss = output.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in proj.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
    
    def test_all_parameters_have_gradients_mlp(self):
        """测试 MLP 投影所有参数都有梯度"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=3
        )
        
        visual_features = torch.randn(1, 196, 768)
        output = proj(visual_features)
        loss = output.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in proj.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestDeviceCompatibility:
    """测试设备兼容性"""
    
    def test_cpu_forward(self):
        """测试 CPU 前向传播"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        visual_features = torch.randn(1, 196, 768)
        output = proj(visual_features)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """测试 CUDA 前向传播"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        ).cuda()
        
        visual_features = torch.randn(1, 196, 768).cuda()
        output = proj(visual_features)
        
        assert output.device.type == 'cuda'
    
    def test_output_dtype(self):
        """测试输出数据类型"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp'
        )
        
        visual_features = torch.randn(1, 196, 768)
        output = proj(visual_features)
        
        assert output.dtype == torch.float32


class TestExtraRepr:
    """测试 extra_repr 方法"""
    
    def test_extra_repr_linear(self):
        """测试线性投影的 extra_repr"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        repr_str = proj.extra_repr()
        
        assert "vision_dim=768" in repr_str
        assert "llm_dim=4096" in repr_str
        assert "projection_type='linear'" in repr_str
        assert "mlp_depth" not in repr_str
    
    def test_extra_repr_mlp(self):
        """测试 MLP 投影的 extra_repr"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=3
        )
        
        repr_str = proj.extra_repr()
        
        assert "vision_dim=768" in repr_str
        assert "llm_dim=4096" in repr_str
        assert "projection_type='mlp'" in repr_str
        assert "mlp_depth=3" in repr_str


class TestParameterCount:
    """测试参数数量"""
    
    def test_linear_parameter_count(self):
        """测试线性投影的参数数量"""
        vision_dim = 768
        llm_dim = 4096
        
        proj = VisualProjection(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            projection_type='linear'
        )
        
        # Linear: weight (vision_dim * llm_dim) + bias (llm_dim)
        expected_params = vision_dim * llm_dim + llm_dim
        actual_params = sum(p.numel() for p in proj.parameters())
        
        assert actual_params == expected_params
    
    def test_mlp_depth_2_parameter_count(self):
        """测试 depth=2 MLP 投影的参数数量"""
        vision_dim = 768
        llm_dim = 4096
        
        proj = VisualProjection(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            projection_type='mlp',
            mlp_depth=2
        )
        
        # Layer 1: weight (vision_dim * llm_dim) + bias (llm_dim)
        # Layer 2: weight (llm_dim * llm_dim) + bias (llm_dim)
        expected_params = (
            vision_dim * llm_dim + llm_dim +  # Layer 1
            llm_dim * llm_dim + llm_dim       # Layer 2
        )
        actual_params = sum(p.numel() for p in proj.parameters())
        
        assert actual_params == expected_params
    
    def test_mlp_depth_3_parameter_count(self):
        """测试 depth=3 MLP 投影的参数数量"""
        vision_dim = 768
        llm_dim = 4096
        
        proj = VisualProjection(
            vision_dim=vision_dim,
            llm_dim=llm_dim,
            projection_type='mlp',
            mlp_depth=3
        )
        
        # Layer 1: weight (vision_dim * llm_dim) + bias (llm_dim)
        # Layer 2: weight (llm_dim * llm_dim) + bias (llm_dim)
        # Layer 3: weight (llm_dim * llm_dim) + bias (llm_dim)
        expected_params = (
            vision_dim * llm_dim + llm_dim +  # Layer 1
            llm_dim * llm_dim + llm_dim +     # Layer 2
            llm_dim * llm_dim + llm_dim       # Layer 3
        )
        actual_params = sum(p.numel() for p in proj.parameters())
        
        assert actual_params == expected_params


class TestLLaVAStyleProjection:
    """测试 LLaVA 风格的投影（MLP 投影）"""
    
    def test_llava_typical_config(self):
        """测试 LLaVA 典型配置: ViT-L/14 (1024) -> LLaMA-7B (4096)"""
        proj = VisualProjection(
            vision_dim=1024,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=2
        )
        
        # ViT-L/14 with 224x224 image: 256 patches
        visual_features = torch.randn(2, 256, 1024)
        output = proj(visual_features)
        
        assert output.shape == (2, 256, 4096)
    
    def test_llava_1_5_config(self):
        """测试 LLaVA-1.5 配置: CLIP ViT-L/14@336 (1024) -> Vicuna-7B (4096)"""
        proj = VisualProjection(
            vision_dim=1024,
            llm_dim=4096,
            projection_type='mlp',
            mlp_depth=2
        )
        
        # CLIP ViT-L/14@336: 576 patches
        visual_features = torch.randn(1, 576, 1024)
        output = proj(visual_features)
        
        assert output.shape == (1, 576, 4096)


class TestCLIPStyleProjection:
    """测试 CLIP 风格的投影（线性投影）"""
    
    def test_clip_typical_config(self):
        """测试 CLIP 典型配置: ViT-B/16 (768) -> projection_dim (512)"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=512,
            projection_type='linear'
        )
        
        # [CLS] token only
        visual_features = torch.randn(4, 1, 768)
        output = proj(visual_features)
        
        assert output.shape == (4, 1, 512)


class TestBLIP2StyleProjection:
    """测试 BLIP-2 风格的投影"""
    
    def test_blip2_typical_config(self):
        """测试 BLIP-2 典型配置: Q-Former (768) -> LLM (4096)"""
        proj = VisualProjection(
            vision_dim=768,
            llm_dim=4096,
            projection_type='linear'
        )
        
        # Q-Former output: 32 query tokens
        visual_features = torch.randn(2, 32, 768)
        output = proj(visual_features)
        
        assert output.shape == (2, 32, 4096)
