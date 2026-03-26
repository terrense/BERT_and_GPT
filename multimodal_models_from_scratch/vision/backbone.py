"""
CNN Backbone (ResNet) for DETR

实现简化版 ResNet 作为 DETR 的 backbone，用于提取图像特征。
支持 ResNet-18、ResNet-34、ResNet-50 等不同深度配置。

需求: 3.1
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenBatchNorm2d(nn.Module):
    """冻结的 BatchNorm2d
    
    在 DETR 训练中，backbone 的 BatchNorm 通常被冻结，
    使用预训练时的统计量（running_mean 和 running_var）。
    
    Args:
        num_features: 特征通道数
        eps: 数值稳定性参数
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # 注册为 buffer 而非 parameter，不参与梯度更新
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (batch, num_features, H, W)
        
        Returns:
            normalized: (batch, num_features, H, W)
        """
        # 使用冻结的统计量进行归一化
        # 重塑 weight, bias, mean, var 以便广播
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        running_mean = self.running_mean.view(1, -1, 1, 1)
        running_var = self.running_var.view(1, -1, 1, 1)
        
        # 归一化公式: (x - mean) / sqrt(var + eps) * weight + bias
        return (x - running_mean) / torch.sqrt(running_var + self.eps) * weight + bias


class BasicBlock(nn.Module):
    """ResNet 基础残差块（用于 ResNet-18, ResNet-34）
    
    结构:
    - Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN
    - 残差连接（如果维度不匹配则使用 1x1 卷积）
    - ReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 输出通道数
        stride: 卷积步长（用于下采样）
        norm_layer: 归一化层类型
    """
    
    expansion = 1  # 输出通道数相对于 out_channels 的倍数
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 第一个 3x3 卷积
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        
        # 第二个 3x3 卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        # 残差连接（如果维度不匹配）
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_channels * self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (batch, in_channels, H, W)
        
        Returns:
            output: (batch, out_channels, H', W')
        """
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class Bottleneck(nn.Module):
    """ResNet 瓶颈残差块（用于 ResNet-50, ResNet-101, ResNet-152）
    
    结构:
    - Conv1x1 -> BN -> ReLU -> Conv3x3 -> BN -> ReLU -> Conv1x1 -> BN
    - 残差连接（如果维度不匹配则使用 1x1 卷积）
    - ReLU
    
    Args:
        in_channels: 输入通道数
        out_channels: 中间层通道数（输出通道数为 out_channels * expansion）
        stride: 卷积步长（用于下采样）
        norm_layer: 归一化层类型
    """
    
    expansion = 4  # 输出通道数相对于 out_channels 的倍数
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 1x1 卷积（降维）
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )
        self.bn1 = norm_layer(out_channels)
        
        # 3x3 卷积
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        # 1x1 卷积（升维）
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion,
            kernel_size=1, bias=False
        )
        self.bn3 = norm_layer(out_channels * self.expansion)
        
        # 残差连接（如果维度不匹配）
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                norm_layer(out_channels * self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        Args:
            x: (batch, in_channels, H, W)
        
        Returns:
            output: (batch, out_channels * expansion, H', W')
        """
        identity = x
        
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out



class PositionEmbeddingSine(nn.Module):
    """2D 正弦位置编码
    
    为 2D 特征图生成正弦位置编码，用于 DETR 的 Transformer。
    
    Args:
        d_model: 位置编码维度（必须为偶数）
        temperature: 温度参数，控制频率范围
        normalize: 是否归一化位置坐标
        scale: 归一化时的缩放因子
    """
    
    def __init__(
        self,
        d_model: int = 256,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.temperature = temperature
        self.normalize = normalize
        
        if scale is None:
            scale = 2 * 3.141592653589793  # 2 * pi
        self.scale = scale
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """生成 2D 位置编码
        
        Args:
            x: 特征图 (batch, channels, H, W)
            mask: 可选的掩码 (batch, H, W)，True 表示被掩码的位置
        
        Returns:
            pos_embed: 位置编码 (batch, d_model, H, W)
        """
        batch_size, _, h, w = x.shape
        
        if mask is None:
            mask = torch.zeros((batch_size, h, w), dtype=torch.bool, device=x.device)
        
        # 计算非掩码位置的累积和
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        
        # 归一化
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        
        # 生成频率
        dim_t = torch.arange(self.d_model // 2, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * dim_t / self.d_model)
        
        # 计算位置编码
        # x 方向
        pos_x = x_embed[:, :, :, None] / dim_t  # (batch, H, W, d_model/2)
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()], dim=4)
        pos_x = pos_x.flatten(3)  # (batch, H, W, d_model/2)
        
        # y 方向
        pos_y = y_embed[:, :, :, None] / dim_t  # (batch, H, W, d_model/2)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4)
        pos_y = pos_y.flatten(3)  # (batch, H, W, d_model/2)
        
        # 拼接 x 和 y 方向的位置编码
        pos = torch.cat([pos_y, pos_x], dim=3)  # (batch, H, W, d_model)
        pos = pos.permute(0, 3, 1, 2)  # (batch, d_model, H, W)
        
        return pos


class ResNet(nn.Module):
    """ResNet Backbone
    
    实现简化版 ResNet，支持 ResNet-18、ResNet-34、ResNet-50 等配置。
    用于 DETR 的图像特征提取。
    
    Args:
        block: 残差块类型（BasicBlock 或 Bottleneck）
        layers: 每个 stage 的残差块数量列表
        in_channels: 输入图像通道数
        frozen_bn: 是否使用冻结的 BatchNorm
        return_intermediate: 是否返回中间层特征
    
    Attributes:
        conv1: 初始 7x7 卷积
        bn1: 初始 BatchNorm
        maxpool: 最大池化
        layer1-4: 四个 stage 的残差块
        out_channels: 输出特征通道数
    """
    
    def __init__(
        self,
        block: type,
        layers: List[int],
        in_channels: int = 3,
        frozen_bn: bool = True,
        return_intermediate: bool = False
    ):
        super().__init__()
        
        self.return_intermediate = return_intermediate
        
        # 选择归一化层类型
        if frozen_bn:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
        
        self._norm_layer = norm_layer
        self.inplanes = 64  # 当前通道数
        
        # 初始卷积层
        # 输入: (batch, 3, H, W) -> 输出: (batch, 64, H/2, W/2)
        self.conv1 = nn.Conv2d(
            in_channels, self.inplanes,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        # 最大池化
        # (batch, 64, H/2, W/2) -> (batch, 64, H/4, W/4)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个 stage
        # layer1: (batch, 64, H/4, W/4) -> (batch, 64*expansion, H/4, W/4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        # layer2: (batch, 64*expansion, H/4, W/4) -> (batch, 128*expansion, H/8, W/8)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        # layer3: (batch, 128*expansion, H/8, W/8) -> (batch, 256*expansion, H/16, W/16)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        # layer4: (batch, 256*expansion, H/16, W/16) -> (batch, 512*expansion, H/32, W/32)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 输出通道数
        self.out_channels = 512 * block.expansion
        
        # 初始化权重
        self._init_weights()
    
    def _make_layer(
        self,
        block: type,
        out_channels: int,
        num_blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        """构建一个 stage 的残差块序列
        
        Args:
            block: 残差块类型
            out_channels: 输出通道数
            num_blocks: 残差块数量
            stride: 第一个残差块的步长
        
        Returns:
            Sequential: 残差块序列
        """
        layers = []
        
        # 第一个残差块（可能有下采样）
        layers.append(
            block(self.inplanes, out_channels, stride, self._norm_layer)
        )
        self.inplanes = out_channels * block.expansion
        
        # 后续残差块
        for _ in range(1, num_blocks):
            layers.append(
                block(self.inplanes, out_channels, 1, self._norm_layer)
            )
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    if isinstance(m.weight, nn.Parameter):
                        nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    if isinstance(m.bias, nn.Parameter):
                        nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入图像 (batch, 3, H, W)
            mask: 可选的掩码 (batch, H, W)
        
        Returns:
            Dict 包含:
            - 'features': 最后一层特征图 (batch, out_channels, H/32, W/32)
            - 'mask': 下采样后的掩码 (batch, H/32, W/32)
            - 'intermediate_features': 中间层特征列表（如果 return_intermediate=True）
        """
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # 收集中间特征
        intermediate_features = []
        
        # 四个 stage
        x = self.layer1(x)
        if self.return_intermediate:
            intermediate_features.append(x)
        
        x = self.layer2(x)
        if self.return_intermediate:
            intermediate_features.append(x)
        
        x = self.layer3(x)
        if self.return_intermediate:
            intermediate_features.append(x)
        
        x = self.layer4(x)
        if self.return_intermediate:
            intermediate_features.append(x)
        
        # 下采样掩码
        output_mask = None
        if mask is not None:
            # 将掩码下采样到特征图大小
            output_mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=x.shape[-2:],
                mode='nearest'
            ).squeeze(1).bool()
        
        output = {
            'features': x,
            'mask': output_mask,
        }
        
        if self.return_intermediate:
            output['intermediate_features'] = intermediate_features
        
        return output
    
    def get_feature_channels(self) -> int:
        """获取输出特征通道数"""
        return self.out_channels


class ResNetBackbone(nn.Module):
    """DETR 的 ResNet Backbone 封装
    
    封装 ResNet 和位置编码，提供 DETR 所需的接口。
    
    Args:
        name: ResNet 变体名称 ('resnet18', 'resnet34', 'resnet50')
        frozen_bn: 是否使用冻结的 BatchNorm
        return_intermediate: 是否返回中间层特征
        d_model: 位置编码维度
    """
    
    # ResNet 配置
    CONFIGS = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3]),
    }
    
    def __init__(
        self,
        name: str = 'resnet50',
        frozen_bn: bool = True,
        return_intermediate: bool = False,
        d_model: int = 256
    ):
        super().__init__()
        
        if name not in self.CONFIGS:
            raise ValueError(f"Unknown ResNet variant: {name}. "
                           f"Available: {list(self.CONFIGS.keys())}")
        
        block, layers = self.CONFIGS[name]
        
        # ResNet backbone
        self.backbone = ResNet(
            block=block,
            layers=layers,
            frozen_bn=frozen_bn,
            return_intermediate=return_intermediate
        )
        
        # 位置编码
        self.position_embedding = PositionEmbeddingSine(d_model=d_model)
        
        # 输出通道数
        self.out_channels = self.backbone.out_channels
        self.d_model = d_model
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """前向传播
        
        Args:
            x: 输入图像 (batch, 3, H, W)
            mask: 可选的掩码 (batch, H, W)
        
        Returns:
            Dict 包含:
            - 'features': 特征图 (batch, out_channels, H/32, W/32)
            - 'mask': 下采样后的掩码 (batch, H/32, W/32)
            - 'pos_embed': 位置编码 (batch, d_model, H/32, W/32)
            - 'intermediate_features': 中间层特征列表（如果 return_intermediate=True）
        """
        # 提取特征
        backbone_output = self.backbone(x, mask)
        features = backbone_output['features']
        output_mask = backbone_output.get('mask')
        
        # 生成位置编码
        pos_embed = self.position_embedding(features, output_mask)
        
        output = {
            'features': features,
            'mask': output_mask,
            'pos_embed': pos_embed,
        }
        
        if 'intermediate_features' in backbone_output:
            output['intermediate_features'] = backbone_output['intermediate_features']
        
        return output
    
    def get_feature_channels(self) -> int:
        """获取输出特征通道数"""
        return self.out_channels
    
    def freeze_bn(self):
        """冻结所有 BatchNorm 层"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, FrozenBatchNorm2d)):
                m.eval()


def build_backbone(
    name: str = 'resnet50',
    frozen_bn: bool = True,
    return_intermediate: bool = False,
    d_model: int = 256
) -> ResNetBackbone:
    """构建 ResNet Backbone
    
    便捷函数，用于创建 DETR 的 backbone。
    
    Args:
        name: ResNet 变体名称 ('resnet18', 'resnet34', 'resnet50')
        frozen_bn: 是否使用冻结的 BatchNorm
        return_intermediate: 是否返回中间层特征
        d_model: 位置编码维度
    
    Returns:
        ResNetBackbone: 配置好的 backbone
    """
    return ResNetBackbone(
        name=name,
        frozen_bn=frozen_bn,
        return_intermediate=return_intermediate,
        d_model=d_model
    )
