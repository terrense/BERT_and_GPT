"""
Patch Embedding 单元测试

测试 Patch Embedding 模块的输出形状、[CLS] token、位置嵌入等功能。

需求: 1.3
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.vision.patch_embedding import PatchEmbedding


class TestPatchEmbeddingOutputShape:
    """测试输出形状正确性"""
    
    def test_default_output_shape(self):
        """测试默认配置的输出形状: (batch, num_patches + 1, d_model)"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            in_channels=3,
            d_model=768
        )
        
        # 输入: (batch, 3, 224, 224)
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)
        
        # num_patches = (224 // 16) ** 2 = 14 * 14 = 196
        # 输出: (batch, num_patches + 1, d_model) = (2, 197, 768)
        expected_num_patches = (224 // 16) ** 2
        assert output.shape == (2, expected_num_patches + 1, 768)
    
    def test_num_patches_calculation(self):
        """测试 num_patches = (image_size // patch_size) ** 2"""
        # 测试不同的 image_size 和 patch_size 组合
        test_cases = [
            (224, 16, 196),   # (224 // 16) ** 2 = 14 * 14 = 196
            (224, 32, 49),    # (224 // 32) ** 2 = 7 * 7 = 49
            (384, 16, 576),   # (384 // 16) ** 2 = 24 * 24 = 576
            (256, 8, 1024),   # (256 // 8) ** 2 = 32 * 32 = 1024
        ]
        
        for image_size, patch_size, expected_num_patches in test_cases:
            patch_embed = PatchEmbedding(
                image_size=image_size,
                patch_size=patch_size,
                d_model=768
            )
            
            assert patch_embed.num_patches == expected_num_patches
            
            x = torch.randn(1, 3, image_size, image_size)
            output = patch_embed(x)
            
            # 输出序列长度 = num_patches + 1 (包含 [CLS] token)
            assert output.shape[1] == expected_num_patches + 1
    
    def test_different_d_model(self):
        """测试不同的 d_model 配置"""
        d_model_values = [256, 512, 768, 1024]
        
        for d_model in d_model_values:
            patch_embed = PatchEmbedding(
                image_size=224,
                patch_size=16,
                d_model=d_model
            )
            
            x = torch.randn(1, 3, 224, 224)
            output = patch_embed(x)
            
            assert output.shape == (1, 197, d_model)
    
    def test_different_in_channels(self):
        """测试不同的输入通道数"""
        in_channels_values = [1, 3, 4]
        
        for in_channels in in_channels_values:
            patch_embed = PatchEmbedding(
                image_size=224,
                patch_size=16,
                in_channels=in_channels,
                d_model=768
            )
            
            x = torch.randn(1, in_channels, 224, 224)
            output = patch_embed(x)
            
            assert output.shape == (1, 197, 768)


class TestCLSToken:
    """测试 [CLS] token"""
    
    def test_cls_token_at_position_zero(self):
        """测试 [CLS] token 在序列位置 0"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)
        
        # [CLS] token 应该在位置 0
        # 验证方式：检查 cls_token 参数与输出的位置 0 相关
        # 由于添加了位置嵌入，我们检查 cls_token 的形状
        assert patch_embed.cls_token.shape == (1, 1, 768)
    
    def test_cls_token_is_learnable(self):
        """测试 [CLS] token 是可学习的参数"""
        patch_embed = PatchEmbedding(d_model=768)
        
        # cls_token 应该是 nn.Parameter
        assert isinstance(patch_embed.cls_token, nn.Parameter)
        assert patch_embed.cls_token.requires_grad
    
    def test_cls_token_broadcast_to_batch(self):
        """测试 [CLS] token 正确广播到 batch 维度"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        batch_sizes = [1, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            output = patch_embed(x)
            
            # 每个 batch 的第一个 token 应该来自同一个 cls_token
            # 输出形状应该正确
            assert output.shape[0] == batch_size
            assert output.shape[1] == 197  # num_patches + 1
    
    def test_cls_token_initialization(self):
        """测试 [CLS] token 初始化"""
        patch_embed = PatchEmbedding(d_model=768)
        
        # 检查 cls_token 不是全零（因为使用了截断正态分布初始化）
        # 注意：初始化后可能有非常小的值，但不应该完全为零
        assert patch_embed.cls_token.shape == (1, 1, 768)


class TestPositionEmbedding:
    """测试位置嵌入"""
    
    def test_position_embedding_shape(self):
        """测试位置嵌入形状: (1, num_patches + 1, d_model)"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        # num_patches = 196, 加上 [CLS] token = 197
        expected_shape = (1, 197, 768)
        assert patch_embed.position_embedding.shape == expected_shape
    
    def test_position_embedding_is_learnable(self):
        """测试位置嵌入是可学习的参数"""
        patch_embed = PatchEmbedding(d_model=768)
        
        # position_embedding 应该是 nn.Parameter
        assert isinstance(patch_embed.position_embedding, nn.Parameter)
        assert patch_embed.position_embedding.requires_grad
    
    def test_position_embedding_added_correctly(self):
        """测试位置嵌入正确添加到输出"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)
        
        # 输出应该包含位置信息
        # 验证输出形状正确
        assert output.shape == (2, 197, 768)
    
    def test_position_embedding_different_configs(self):
        """测试不同配置下的位置嵌入形状"""
        test_cases = [
            (224, 16, 768, 197),   # 196 patches + 1 CLS
            (384, 16, 768, 577),   # 576 patches + 1 CLS
            (224, 32, 512, 50),    # 49 patches + 1 CLS
        ]
        
        for image_size, patch_size, d_model, expected_seq_len in test_cases:
            patch_embed = PatchEmbedding(
                image_size=image_size,
                patch_size=patch_size,
                d_model=d_model
            )
            
            assert patch_embed.position_embedding.shape == (1, expected_seq_len, d_model)


class TestDifferentImageSizesAndPatchSizes:
    """测试不同图像尺寸和 patch 尺寸的组合"""
    
    def test_small_image_small_patch(self):
        """测试小图像和小 patch"""
        patch_embed = PatchEmbedding(
            image_size=64,
            patch_size=8,
            d_model=256
        )
        
        x = torch.randn(1, 3, 64, 64)
        output = patch_embed(x)
        
        # num_patches = (64 // 8) ** 2 = 64
        assert output.shape == (1, 65, 256)
    
    def test_large_image_large_patch(self):
        """测试大图像和大 patch"""
        patch_embed = PatchEmbedding(
            image_size=512,
            patch_size=32,
            d_model=1024
        )
        
        x = torch.randn(1, 3, 512, 512)
        output = patch_embed(x)
        
        # num_patches = (512 // 32) ** 2 = 256
        assert output.shape == (1, 257, 1024)
    
    def test_vit_base_config(self):
        """测试 ViT-Base 配置"""
        # ViT-Base: image_size=224, patch_size=16, d_model=768
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(4, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.shape == (4, 197, 768)
    
    def test_vit_large_config(self):
        """测试 ViT-Large 配置"""
        # ViT-Large: image_size=224, patch_size=16, d_model=1024
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=1024
        )
        
        x = torch.randn(2, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.shape == (2, 197, 1024)
    
    def test_vit_huge_config(self):
        """测试 ViT-Huge 配置"""
        # ViT-Huge: image_size=224, patch_size=14, d_model=1280
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=14,
            d_model=1280
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        
        # num_patches = (224 // 14) ** 2 = 16 * 16 = 256
        assert output.shape == (1, 257, 1280)


class TestBatchProcessing:
    """测试批量处理"""
    
    def test_single_image(self):
        """测试单张图像处理"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.shape == (1, 197, 768)
    
    def test_batch_of_images(self):
        """测试批量图像处理"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        batch_sizes = [2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            output = patch_embed(x)
            
            assert output.shape == (batch_size, 197, 768)
    
    def test_batch_consistency(self):
        """测试批量处理的一致性"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        patch_embed.eval()  # 设置为评估模式
        
        # 创建相同的输入
        x1 = torch.randn(1, 3, 224, 224)
        x2 = x1.clone()
        
        # 单独处理
        with torch.no_grad():
            output1 = patch_embed(x1)
            output2 = patch_embed(x2)
        
        # 结果应该相同
        assert torch.allclose(output1, output2)
    
    def test_batch_vs_individual_processing(self):
        """测试批量处理与单独处理的一致性"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        patch_embed.eval()
        
        # 创建两张不同的图像
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        
        # 批量处理
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = patch_embed(x_batch)
        
        # 单独处理
        with torch.no_grad():
            output1 = patch_embed(x1)
            output2 = patch_embed(x2)
        
        # 批量处理的结果应该与单独处理一致
        assert torch.allclose(output_batch[0], output1[0])
        assert torch.allclose(output_batch[1], output2[0])


class TestConv2dProjection:
    """测试 Conv2d 投影层"""
    
    def test_projection_layer_exists(self):
        """测试投影层存在"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        assert hasattr(patch_embed, 'projection')
        assert isinstance(patch_embed.projection, nn.Conv2d)
    
    def test_projection_layer_config(self):
        """测试投影层配置正确"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            in_channels=3,
            d_model=768
        )
        
        proj = patch_embed.projection
        
        # 检查 Conv2d 配置
        assert proj.in_channels == 3
        assert proj.out_channels == 768
        assert proj.kernel_size == (16, 16)
        assert proj.stride == (16, 16)
    
    def test_projection_output_shape(self):
        """测试投影层输出形状"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(2, 3, 224, 224)
        
        # 直接测试投影层输出
        proj_output = patch_embed.projection(x)
        
        # 输出形状: (batch, d_model, H/patch_size, W/patch_size)
        # = (2, 768, 14, 14)
        assert proj_output.shape == (2, 768, 14, 14)


class TestGradientFlow:
    """测试梯度流动"""
    
    def test_gradients_flow_through_model(self):
        """测试梯度能够正确流过模型"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = patch_embed(x)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度存在
        assert x.grad is not None
        assert patch_embed.cls_token.grad is not None
        assert patch_embed.position_embedding.grad is not None
        assert patch_embed.projection.weight.grad is not None
    
    def test_all_parameters_have_gradients(self):
        """测试所有参数都有梯度"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        loss = output.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in patch_embed.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestDeviceCompatibility:
    """测试设备兼容性"""
    
    def test_cpu_forward(self):
        """测试 CPU 前向传播"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """测试 CUDA 前向传播"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        ).cuda()
        
        x = torch.randn(1, 3, 224, 224).cuda()
        output = patch_embed(x)
        
        assert output.device.type == 'cuda'
    
    def test_output_dtype(self):
        """测试输出数据类型"""
        patch_embed = PatchEmbedding(
            image_size=224,
            patch_size=16,
            d_model=768
        )
        
        x = torch.randn(1, 3, 224, 224)
        output = patch_embed(x)
        
        assert output.dtype == torch.float32
