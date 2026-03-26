"""
图像预处理模块

实现 ImageProcessor 类，支持图像缩放、归一化。
支持从文件路径、PIL Image、numpy array、torch Tensor 加载图像。
使用 ImageNet 均值和标准差进行归一化。

需求: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from PIL import Image


class ImageProcessor:
    """图像预处理器
    
    将输入图像转换为模型可接受的格式，包括：
    - 图像缩放到指定尺寸
    - 使用 ImageNet 均值和标准差进行归一化
    - 支持多种输入格式（文件路径、PIL Image、numpy array、torch Tensor）
    
    Attributes:
        image_size: 目标图像尺寸
        mean: 归一化均值（ImageNet 默认值）
        std: 归一化标准差（ImageNet 默认值）
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225)
    ):
        """初始化图像预处理器
        
        Args:
            image_size: 目标图像尺寸（正方形）
            mean: ImageNet 均值，用于归一化
            std: ImageNet 标准差，用于归一化
        """
        self.image_size = image_size
        self.mean = mean
        self.std = std
    
    def __call__(
        self,
        images: Union[str, Image.Image, np.ndarray, torch.Tensor, List],
        return_tensors: str = 'pt'
    ) -> Dict[str, torch.Tensor]:
        """预处理图像
        
        Args:
            images: 单张或多张图像，支持以下格式：
                - str: 图像文件路径
                - PIL.Image.Image: PIL 图像对象
                - np.ndarray: numpy 数组，形状为 (H, W, 3) 或 (3, H, W)
                - torch.Tensor: PyTorch 张量，形状为 (3, H, W) 或 (H, W, 3)
                - List: 上述任意格式的列表
            return_tensors: 返回格式，目前仅支持 'pt'（PyTorch）
        
        Returns:
            包含 'pixel_values' 键的字典，值为形状 (batch, 3, H, W) 的归一化张量
        
        Raises:
            ValueError: 当输入格式不支持或 return_tensors 不是 'pt' 时
        """
        if return_tensors != 'pt':
            raise ValueError(f"Unsupported return_tensors: {return_tensors}. Only 'pt' is supported.")
        
        # 处理单张图像或图像列表
        if not isinstance(images, list):
            images = [images]
        
        # 处理每张图像
        processed_images = []
        for image in images:
            processed = self._process_single_image(image)
            processed_images.append(processed)
        
        # 堆叠为 batch
        pixel_values = torch.stack(processed_images, dim=0)
        
        return {'pixel_values': pixel_values}
    
    def _process_single_image(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """处理单张图像
        
        Args:
            image: 单张图像，支持多种格式
        
        Returns:
            形状为 (3, H, W) 的归一化张量
        """
        # 1. 加载图像为 PIL Image
        pil_image = self._load_image(image)
        
        # 2. 缩放到目标尺寸
        pil_image = self._resize_image(pil_image)
        
        # 3. 转换为张量
        tensor = self._to_tensor(pil_image)
        
        # 4. 归一化
        tensor = self._normalize(tensor)
        
        return tensor
    
    def _load_image(
        self,
        image: Union[str, Image.Image, np.ndarray, torch.Tensor]
    ) -> Image.Image:
        """加载图像为 PIL Image
        
        Args:
            image: 输入图像，支持多种格式
        
        Returns:
            RGB 格式的 PIL Image
        
        Raises:
            ValueError: 当输入格式不支持时
        """
        if isinstance(image, str):
            # 从文件路径加载
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            # 已经是 PIL Image
            pil_image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            # 从 numpy array 加载
            pil_image = self._numpy_to_pil(image)
        elif isinstance(image, torch.Tensor):
            # 从 torch Tensor 加载
            pil_image = self._tensor_to_pil(image)
        else:
            raise ValueError(
                f"Unsupported image type: {type(image)}. "
                "Supported types: str, PIL.Image.Image, np.ndarray, torch.Tensor"
            )
        
        return pil_image
    
    def _numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """将 numpy array 转换为 PIL Image
        
        Args:
            array: numpy 数组，形状为 (H, W, 3) 或 (3, H, W)
        
        Returns:
            RGB 格式的 PIL Image
        """
        # 处理不同的通道顺序
        if array.ndim == 3:
            if array.shape[0] == 3:
                # (3, H, W) -> (H, W, 3)
                array = np.transpose(array, (1, 2, 0))
            elif array.shape[2] != 3:
                raise ValueError(f"Expected 3 channels, got shape {array.shape}")
        else:
            raise ValueError(f"Expected 3D array, got {array.ndim}D array")
        
        # 处理数据类型和范围
        if array.dtype == np.float32 or array.dtype == np.float64:
            # 假设范围是 [0, 1]，转换为 [0, 255]
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        return Image.fromarray(array, mode='RGB')
    
    def _tensor_to_pil(self, tensor: torch.Tensor) -> Image.Image:
        """将 torch Tensor 转换为 PIL Image
        
        Args:
            tensor: PyTorch 张量，形状为 (3, H, W) 或 (H, W, 3)
        
        Returns:
            RGB 格式的 PIL Image
        """
        # 转换为 numpy array
        array = tensor.detach().cpu().numpy()
        return self._numpy_to_pil(array)
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """缩放图像到目标尺寸
        
        Args:
            image: PIL Image
        
        Returns:
            缩放后的 PIL Image
        """
        # 兼容不同版本的 PIL
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        
        return image.resize(
            (self.image_size, self.image_size),
            resample=resample
        )
    
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """将 PIL Image 转换为张量
        
        Args:
            image: PIL Image
        
        Returns:
            形状为 (3, H, W) 的张量，值范围 [0, 1]
        """
        # 转换为 numpy array
        array = np.array(image, dtype=np.float32)
        
        # 归一化到 [0, 1]
        array = array / 255.0
        
        # (H, W, 3) -> (3, H, W)
        array = np.transpose(array, (2, 0, 1))
        
        return torch.from_numpy(array)
    
    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """使用 ImageNet 均值和标准差归一化张量
        
        Args:
            tensor: 形状为 (3, H, W) 的张量，值范围 [0, 1]
        
        Returns:
            归一化后的张量
        """
        mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
        
        return (tensor - mean) / std
