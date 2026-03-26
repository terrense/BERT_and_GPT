"""
ImageProcessor 单元测试

测试图像预处理模块的各种输入格式处理、输出形状和归一化值。

需求: 16.5
"""

import pytest
import numpy as np
import torch
from PIL import Image

from multimodal_models_from_scratch.vision.image_processor import ImageProcessor


class TestImageProcessorInputFormats:
    """测试不同输入格式的处理"""
    
    @pytest.fixture
    def processor(self):
        """创建 ImageProcessor 实例"""
        return ImageProcessor(image_size=224)
    
    def test_pil_image_input(self, processor):
        """测试 PIL Image 输入处理"""
        # 创建 RGB PIL Image
        pil_image = Image.new('RGB', (256, 256), color=(128, 64, 192))
        
        result = processor(pil_image)
        
        assert 'pixel_values' in result
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_pil_image_rgba_conversion(self, processor):
        """测试 RGBA PIL Image 自动转换为 RGB"""
        # 创建 RGBA PIL Image
        pil_image = Image.new('RGBA', (256, 256), color=(128, 64, 192, 255))
        
        result = processor(pil_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_numpy_array_hwc_input(self, processor):
        """测试 numpy array (H, W, 3) 格式输入"""
        # 创建 (H, W, 3) 格式的 numpy array
        np_image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        result = processor(np_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_numpy_array_chw_input(self, processor):
        """测试 numpy array (3, H, W) 格式输入"""
        # 创建 (3, H, W) 格式的 numpy array
        np_image = np.random.randint(0, 256, (3, 256, 256), dtype=np.uint8)
        
        result = processor(np_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_numpy_array_float_input(self, processor):
        """测试 float 类型 numpy array 输入 (值范围 0-1)"""
        # 创建 float 类型的 numpy array，值范围 [0, 1]
        np_image = np.random.rand(256, 256, 3).astype(np.float32)
        
        result = processor(np_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_torch_tensor_chw_input(self, processor):
        """测试 torch Tensor (3, H, W) 格式输入"""
        # 创建 (3, H, W) 格式的 torch Tensor
        tensor_image = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        
        result = processor(tensor_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_torch_tensor_hwc_input(self, processor):
        """测试 torch Tensor (H, W, 3) 格式输入"""
        # 创建 (H, W, 3) 格式的 torch Tensor
        tensor_image = torch.randint(0, 256, (256, 256, 3), dtype=torch.uint8)
        
        result = processor(tensor_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32
    
    def test_torch_tensor_float_input(self, processor):
        """测试 float 类型 torch Tensor 输入"""
        # 创建 float 类型的 torch Tensor，值范围 [0, 1]
        tensor_image = torch.rand(3, 256, 256, dtype=torch.float32)
        
        result = processor(tensor_image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
        assert result['pixel_values'].dtype == torch.float32


class TestImageProcessorBatchProcessing:
    """测试批量图像处理"""
    
    @pytest.fixture
    def processor(self):
        """创建 ImageProcessor 实例"""
        return ImageProcessor(image_size=224)
    
    def test_list_of_pil_images(self, processor):
        """测试 PIL Image 列表输入"""
        images = [
            Image.new('RGB', (256, 256), color=(128, 64, 192)),
            Image.new('RGB', (128, 128), color=(64, 128, 64)),
            Image.new('RGB', (512, 512), color=(192, 192, 64)),
        ]
        
        result = processor(images)
        
        assert result['pixel_values'].shape == (3, 3, 224, 224)
    
    def test_list_of_numpy_arrays(self, processor):
        """测试 numpy array 列表输入"""
        images = [
            np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8),
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
        ]
        
        result = processor(images)
        
        assert result['pixel_values'].shape == (2, 3, 224, 224)
    
    def test_list_of_torch_tensors(self, processor):
        """测试 torch Tensor 列表输入"""
        images = [
            torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8),
            torch.randint(0, 256, (3, 128, 128), dtype=torch.uint8),
        ]
        
        result = processor(images)
        
        assert result['pixel_values'].shape == (2, 3, 224, 224)
    
    def test_mixed_input_formats(self, processor):
        """测试混合输入格式"""
        images = [
            Image.new('RGB', (256, 256), color=(128, 64, 192)),
            np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8),
            torch.randint(0, 256, (3, 200, 200), dtype=torch.uint8),
        ]
        
        result = processor(images)
        
        assert result['pixel_values'].shape == (3, 3, 224, 224)


class TestImageProcessorOutputShape:
    """测试输出形状正确性"""
    
    def test_default_image_size(self):
        """测试默认图像尺寸 224x224"""
        processor = ImageProcessor()
        image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_custom_image_size(self):
        """测试自定义图像尺寸"""
        processor = ImageProcessor(image_size=384)
        image = Image.new('RGB', (512, 512), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 384, 384)
    
    def test_small_image_upscaling(self):
        """测试小图像放大"""
        processor = ImageProcessor(image_size=224)
        image = Image.new('RGB', (64, 64), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_non_square_image(self):
        """测试非正方形图像处理"""
        processor = ImageProcessor(image_size=224)
        image = Image.new('RGB', (640, 480), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_batch_output_shape(self):
        """测试批量处理输出形状"""
        processor = ImageProcessor(image_size=224)
        images = [Image.new('RGB', (256, 256)) for _ in range(5)]
        
        result = processor(images)
        
        assert result['pixel_values'].shape == (5, 3, 224, 224)


class TestImageProcessorNormalization:
    """测试归一化值正确性"""
    
    def test_imagenet_mean_std_default(self):
        """测试默认使用 ImageNet 均值和标准差"""
        processor = ImageProcessor()
        
        assert processor.mean == (0.485, 0.456, 0.406)
        assert processor.std == (0.229, 0.224, 0.225)
    
    def test_custom_mean_std(self):
        """测试自定义均值和标准差"""
        custom_mean = (0.5, 0.5, 0.5)
        custom_std = (0.5, 0.5, 0.5)
        processor = ImageProcessor(mean=custom_mean, std=custom_std)
        
        assert processor.mean == custom_mean
        assert processor.std == custom_std
    
    def test_normalization_values_range(self):
        """测试归一化后的值范围"""
        processor = ImageProcessor()
        # 创建纯色图像
        image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        result = processor(image)
        pixel_values = result['pixel_values']
        
        # 归一化后的值应该在合理范围内
        # 对于 ImageNet 归一化，典型范围约为 [-2.5, 2.5]
        assert pixel_values.min() >= -3.0
        assert pixel_values.max() <= 3.0
    
    def test_normalization_formula(self):
        """测试归一化公式正确性: (x - mean) / std"""
        processor = ImageProcessor()
        
        # 创建已知像素值的图像
        # 使用纯白色图像 (255, 255, 255)
        white_image = Image.new('RGB', (224, 224), color=(255, 255, 255))
        result = processor(white_image)
        pixel_values = result['pixel_values']
        
        # 白色像素值为 1.0（归一化到 [0, 1] 后）
        # 归一化后: (1.0 - mean) / std
        expected_r = (1.0 - 0.485) / 0.229
        expected_g = (1.0 - 0.456) / 0.224
        expected_b = (1.0 - 0.406) / 0.225
        
        # 检查每个通道的归一化值
        assert torch.allclose(pixel_values[0, 0], torch.full((224, 224), expected_r), atol=1e-4)
        assert torch.allclose(pixel_values[0, 1], torch.full((224, 224), expected_g), atol=1e-4)
        assert torch.allclose(pixel_values[0, 2], torch.full((224, 224), expected_b), atol=1e-4)
    
    def test_black_image_normalization(self):
        """测试黑色图像归一化"""
        processor = ImageProcessor()
        
        # 创建纯黑色图像 (0, 0, 0)
        black_image = Image.new('RGB', (224, 224), color=(0, 0, 0))
        result = processor(black_image)
        pixel_values = result['pixel_values']
        
        # 黑色像素值为 0.0（归一化到 [0, 1] 后）
        # 归一化后: (0.0 - mean) / std
        expected_r = (0.0 - 0.485) / 0.229
        expected_g = (0.0 - 0.456) / 0.224
        expected_b = (0.0 - 0.406) / 0.225
        
        assert torch.allclose(pixel_values[0, 0], torch.full((224, 224), expected_r), atol=1e-4)
        assert torch.allclose(pixel_values[0, 1], torch.full((224, 224), expected_g), atol=1e-4)
        assert torch.allclose(pixel_values[0, 2], torch.full((224, 224), expected_b), atol=1e-4)
    
    def test_channel_order_rgb(self):
        """测试通道顺序为 RGB"""
        processor = ImageProcessor()
        
        # 创建红色图像 (255, 0, 0)
        red_image = Image.new('RGB', (224, 224), color=(255, 0, 0))
        result = processor(red_image)
        pixel_values = result['pixel_values']
        
        # R 通道应该有最大值，G 和 B 通道应该有最小值
        r_channel_mean = pixel_values[0, 0].mean().item()
        g_channel_mean = pixel_values[0, 1].mean().item()
        b_channel_mean = pixel_values[0, 2].mean().item()
        
        # R 通道归一化后应该是正值（因为 1.0 > mean）
        # G 和 B 通道归一化后应该是负值（因为 0.0 < mean）
        assert r_channel_mean > 0
        assert g_channel_mean < 0
        assert b_channel_mean < 0


class TestImageProcessorEdgeCases:
    """测试边界情况"""
    
    @pytest.fixture
    def processor(self):
        """创建 ImageProcessor 实例"""
        return ImageProcessor(image_size=224)
    
    def test_single_pixel_image(self, processor):
        """测试单像素图像"""
        image = Image.new('RGB', (1, 1), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_very_large_image(self, processor):
        """测试大图像缩小"""
        image = Image.new('RGB', (2048, 2048), color=(128, 128, 128))
        
        result = processor(image)
        
        assert result['pixel_values'].shape == (1, 3, 224, 224)
    
    def test_empty_list_raises_error(self, processor):
        """测试空列表输入"""
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            processor([])
    
    def test_unsupported_return_tensors(self, processor):
        """测试不支持的 return_tensors 参数"""
        image = Image.new('RGB', (224, 224))
        
        with pytest.raises(ValueError):
            processor(image, return_tensors='np')
    
    def test_unsupported_input_type(self, processor):
        """测试不支持的输入类型"""
        # 字符串被视为文件路径，无效路径会抛出 FileNotFoundError
        with pytest.raises(FileNotFoundError):
            processor("not_a_valid_path_or_image")
    
    def test_unsupported_object_type(self, processor):
        """测试不支持的对象类型"""
        # 传入不支持的对象类型应该抛出 ValueError
        with pytest.raises(ValueError):
            processor(12345)  # 整数不是支持的类型
    
    def test_grayscale_image_conversion(self, processor):
        """测试灰度图像自动转换为 RGB"""
        # 创建灰度图像
        gray_image = Image.new('L', (224, 224), color=128)
        
        result = processor(gray_image)
        
        # 应该自动转换为 RGB
        assert result['pixel_values'].shape == (1, 3, 224, 224)
