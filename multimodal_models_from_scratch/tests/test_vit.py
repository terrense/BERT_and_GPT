"""
ViT 模型单元测试

测试 ViT 模型的前向传播输出形状、分类头输出、get_image_features 方法等功能。

需求: 2.6
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.vision.vit import ViTModel
from multimodal_models_from_scratch.config import VisionConfig


class TestViTForwardPassOutputShapes:
    """测试前向传播输出形状"""
    
    def test_last_hidden_state_shape(self):
        """测试 last_hidden_state 输出形状: (batch, num_patches + 1, d_model)"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        # num_patches = (224 // 16) ** 2 = 196
        # last_hidden_state: (batch, num_patches + 1, d_model) = (2, 197, 768)
        assert 'last_hidden_state' in output
        assert output['last_hidden_state'].shape == (2, 197, 768)

    def test_pooler_output_shape(self):
        """测试 pooler_output 输出形状: (batch, d_model)"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(4, 3, 224, 224)
        output = model(x)
        
        # pooler_output: (batch, d_model) = (4, 768)
        assert 'pooler_output' in output
        assert output['pooler_output'].shape == (4, 768)
    
    def test_logits_shape_with_classifier(self):
        """测试 logits 输出形状: (batch, num_classes)"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        # logits: (batch, num_classes) = (2, 1000)
        assert 'logits' in output
        assert output['logits'].shape == (2, 1000)
    
    def test_no_logits_without_classifier(self):
        """测试 num_classes=0 时没有 logits 输出"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=0  # 不创建分类头
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        # 没有分类头时不应该有 logits
        assert 'logits' not in output
        assert model.classifier is None


class TestClassificationHead:
    """测试分类头输出"""
    
    def test_classifier_exists_when_num_classes_positive(self):
        """测试 num_classes > 0 时分类头存在"""
        config = VisionConfig(num_classes=1000)
        model = ViTModel(config)
        
        assert model.classifier is not None
        assert isinstance(model.classifier, nn.Linear)
    
    def test_classifier_output_dimension(self):
        """测试分类头输出维度正确"""
        num_classes_values = [10, 100, 1000, 21843]
        
        for num_classes in num_classes_values:
            config = VisionConfig(
                image_size=224,
                patch_size=16,
                d_model=768,
                num_heads=12,
                num_layers=2,
                d_ff=3072,
                num_classes=num_classes
            )
            model = ViTModel(config)
            
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            assert output['logits'].shape == (1, num_classes)
    
    def test_classifier_input_from_cls_token(self):
        """测试分类头输入来自 [CLS] token"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=10
        )
        model = ViTModel(config)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        # pooler_output 应该是 last_hidden_state 的第一个 token
        pooler_output = output['pooler_output']
        cls_from_hidden = output['last_hidden_state'][:, 0]
        
        assert torch.allclose(pooler_output, cls_from_hidden)


class TestGetImageFeatures:
    """测试 get_image_features 方法"""
    
    def test_get_image_features_output_shape(self):
        """测试 get_image_features 输出形状: (batch, num_patches, d_model)"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        features = model.get_image_features(x)
        
        # num_patches = (224 // 16) ** 2 = 196
        # 输出: (batch, num_patches, d_model) = (2, 196, 768)
        # 注意：不包含 [CLS] token
        assert features.shape == (2, 196, 768)
    
    def test_get_image_features_excludes_cls_token(self):
        """测试 get_image_features 不包含 [CLS] token"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=0
        )
        model = ViTModel(config)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
            features = model.get_image_features(x)
        
        # features 应该等于 last_hidden_state[:, 1:]（去掉 [CLS] token）
        expected_features = output['last_hidden_state'][:, 1:]
        assert torch.allclose(features, expected_features)
    
    def test_get_image_features_different_configs(self):
        """测试不同配置下的 get_image_features 输出形状"""
        test_cases = [
            (224, 16, 768, 196),   # (224 // 16) ** 2 = 196
            (384, 16, 768, 576),   # (384 // 16) ** 2 = 576
            (224, 32, 512, 49),    # (224 // 32) ** 2 = 49
            (256, 8, 256, 1024),   # (256 // 8) ** 2 = 1024
        ]
        
        for image_size, patch_size, d_model, expected_num_patches in test_cases:
            config = VisionConfig(
                image_size=image_size,
                patch_size=patch_size,
                d_model=d_model,
                num_heads=8,
                num_layers=2,
                d_ff=d_model * 4,
                num_classes=0
            )
            model = ViTModel(config)
            
            x = torch.randn(1, 3, image_size, image_size)
            features = model.get_image_features(x)
            
            assert features.shape == (1, expected_num_patches, d_model)


class TestOutputHiddenStates:
    """测试 output_hidden_states=True 返回所有隐藏状态"""
    
    def test_hidden_states_returned_when_enabled(self):
        """测试 output_hidden_states=True 时返回 hidden_states"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x, output_hidden_states=True)
        
        assert 'hidden_states' in output
        assert output['hidden_states'] is not None
    
    def test_hidden_states_not_returned_when_disabled(self):
        """测试 output_hidden_states=False 时不返回 hidden_states"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x, output_hidden_states=False)
        
        assert 'hidden_states' not in output
    
    def test_hidden_states_count(self):
        """测试 hidden_states 数量 = num_layers + 1（包含初始嵌入）"""
        num_layers = 6
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=num_layers,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x, output_hidden_states=True)
        
        # hidden_states 应该包含：初始嵌入 + 每层输出
        # 总共 num_layers + 1 个
        assert len(output['hidden_states']) == num_layers + 1
    
    def test_hidden_states_shapes(self):
        """测试每个 hidden_state 的形状正确"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=4,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 224, 224)
        output = model(x, output_hidden_states=True)
        
        # 每个 hidden_state 形状: (batch, num_patches + 1, d_model)
        expected_shape = (batch_size, 197, 768)
        for hidden_state in output['hidden_states']:
            assert hidden_state.shape == expected_shape


class TestDifferentVisionConfigs:
    """测试不同 VisionConfig 配置"""
    
    def test_vit_tiny_config(self):
        """测试 ViT-Tiny 配置"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=192,
            num_heads=3,
            num_layers=12,
            d_ff=768,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].shape == (1, 197, 192)
        assert output['pooler_output'].shape == (1, 192)
        assert output['logits'].shape == (1, 1000)
    
    def test_vit_small_config(self):
        """测试 ViT-Small 配置"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=384,
            num_heads=6,
            num_layers=12,
            d_ff=1536,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].shape == (1, 197, 384)
        assert output['pooler_output'].shape == (1, 384)
        assert output['logits'].shape == (1, 1000)
    
    def test_vit_base_config(self):
        """测试 ViT-Base 配置"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=12,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].shape == (1, 197, 768)
        assert output['pooler_output'].shape == (1, 768)
        assert output['logits'].shape == (1, 1000)
    
    def test_vit_large_config(self):
        """测试 ViT-Large 配置"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=1024,
            num_heads=16,
            num_layers=2,  # 减少层数以加快测试
            d_ff=4096,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].shape == (1, 197, 1024)
        assert output['pooler_output'].shape == (1, 1024)
        assert output['logits'].shape == (1, 1000)
    
    def test_different_image_sizes(self):
        """测试不同图像尺寸"""
        image_sizes = [224, 256, 384, 512]
        
        for image_size in image_sizes:
            config = VisionConfig(
                image_size=image_size,
                patch_size=16,
                d_model=768,
                num_heads=12,
                num_layers=2,
                d_ff=3072,
                num_classes=100
            )
            model = ViTModel(config)
            
            x = torch.randn(1, 3, image_size, image_size)
            output = model(x)
            
            num_patches = (image_size // 16) ** 2
            assert output['last_hidden_state'].shape == (1, num_patches + 1, 768)
    
    def test_different_patch_sizes(self):
        """测试不同 patch 尺寸"""
        patch_sizes = [8, 14, 16, 32]
        
        for patch_size in patch_sizes:
            config = VisionConfig(
                image_size=224,
                patch_size=patch_size,
                d_model=768,
                num_heads=12,
                num_layers=2,
                d_ff=3072,
                num_classes=100
            )
            model = ViTModel(config)
            
            x = torch.randn(1, 3, 224, 224)
            output = model(x)
            
            num_patches = (224 // patch_size) ** 2
            assert output['last_hidden_state'].shape == (1, num_patches + 1, 768)


class TestBatchProcessing:
    """测试批量处理"""
    
    def test_single_image(self):
        """测试单张图像处理"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].shape == (1, 197, 768)
        assert output['pooler_output'].shape == (1, 768)
        assert output['logits'].shape == (1, 1000)
    
    def test_batch_of_images(self):
        """测试批量图像处理"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=1000
        )
        model = ViTModel(config)
        
        batch_sizes = [2, 4, 8]
        
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, 224, 224)
            output = model(x)
            
            assert output['last_hidden_state'].shape == (batch_size, 197, 768)
            assert output['pooler_output'].shape == (batch_size, 768)
            assert output['logits'].shape == (batch_size, 1000)
    
    def test_batch_consistency(self):
        """测试批量处理的一致性"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        model.eval()
        
        # 创建相同的输入
        x1 = torch.randn(1, 3, 224, 224)
        x2 = x1.clone()
        
        # 单独处理
        with torch.no_grad():
            output1 = model(x1)
            output2 = model(x2)
        
        # 结果应该相同
        assert torch.allclose(output1['last_hidden_state'], output2['last_hidden_state'])
        assert torch.allclose(output1['pooler_output'], output2['pooler_output'])
        assert torch.allclose(output1['logits'], output2['logits'])
    
    def test_batch_vs_individual_processing(self):
        """测试批量处理与单独处理的一致性"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        model.eval()
        
        # 创建两张不同的图像
        x1 = torch.randn(1, 3, 224, 224)
        x2 = torch.randn(1, 3, 224, 224)
        
        # 批量处理
        x_batch = torch.cat([x1, x2], dim=0)
        with torch.no_grad():
            output_batch = model(x_batch)
        
        # 单独处理
        with torch.no_grad():
            output1 = model(x1)
            output2 = model(x2)
        
        # 批量处理的结果应该与单独处理一致（使用较宽松的容差）
        assert torch.allclose(output_batch['last_hidden_state'][0], output1['last_hidden_state'][0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(output_batch['last_hidden_state'][1], output2['last_hidden_state'][0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(output_batch['logits'][0], output1['logits'][0], atol=1e-5, rtol=1e-4)
        assert torch.allclose(output_batch['logits'][1], output2['logits'][0], atol=1e-5, rtol=1e-4)


class TestModelComponents:
    """测试模型组件"""
    
    def test_patch_embedding_exists(self):
        """测试 patch_embedding 组件存在"""
        config = VisionConfig()
        model = ViTModel(config)
        
        assert hasattr(model, 'patch_embedding')
        assert model.patch_embedding is not None
    
    def test_encoder_layers_exist(self):
        """测试 encoder_layers 组件存在"""
        config = VisionConfig(num_layers=6)
        model = ViTModel(config)
        
        assert hasattr(model, 'encoder_layers')
        assert len(model.encoder_layers) == 6
    
    def test_layer_norm_exists(self):
        """测试 LayerNorm 组件存在"""
        config = VisionConfig()
        model = ViTModel(config)
        
        assert hasattr(model, 'norm')
        assert isinstance(model.norm, nn.LayerNorm)
    
    def test_dropout_exists(self):
        """测试 Dropout 组件存在"""
        config = VisionConfig(dropout_rate=0.1)
        model = ViTModel(config)
        
        assert hasattr(model, 'dropout')
        assert isinstance(model.dropout, nn.Dropout)


class TestGradientFlow:
    """测试梯度流动"""
    
    def test_gradients_flow_through_model(self):
        """测试梯度能够正确流过模型"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)
        
        # 计算损失并反向传播
        loss = output['logits'].sum()
        loss.backward()
        
        # 检查梯度存在
        assert x.grad is not None
    
    def test_all_parameters_have_gradients(self):
        """测试所有参数都有梯度"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        loss = output['logits'].sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"


class TestDeviceCompatibility:
    """测试设备兼容性"""
    
    def test_cpu_forward(self):
        """测试 CPU 前向传播"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].device.type == 'cpu'
        assert output['pooler_output'].device.type == 'cpu'
        assert output['logits'].device.type == 'cpu'
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """测试 CUDA 前向传播"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config).cuda()
        
        x = torch.randn(1, 3, 224, 224).cuda()
        output = model(x)
        
        assert output['last_hidden_state'].device.type == 'cuda'
        assert output['pooler_output'].device.type == 'cuda'
        assert output['logits'].device.type == 'cuda'
    
    def test_output_dtype(self):
        """测试输出数据类型"""
        config = VisionConfig(
            image_size=224,
            patch_size=16,
            d_model=768,
            num_heads=12,
            num_layers=2,
            d_ff=3072,
            num_classes=100
        )
        model = ViTModel(config)
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        
        assert output['last_hidden_state'].dtype == torch.float32
        assert output['pooler_output'].dtype == torch.float32
        assert output['logits'].dtype == torch.float32
