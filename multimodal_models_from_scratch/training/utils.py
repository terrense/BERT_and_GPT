"""
训练工具函数

复用 bert-gpt-from-scratch 的训练工具函数，并添加多模态特有的工具。

包含：
- 学习率调度器工厂
- 梯度裁剪工具
- 检查点管理工具
- 日志记录工具
- 指标跟踪工具
- 训练配置数据类

需求: 17.6
"""

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

# 复用 bert-gpt-from-scratch 的训练配置
from bert_gpt_from_scratch.config import TrainingConfig, SFTConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 训练配置数据类
# =============================================================================

@dataclass
class MultimodalTrainingConfig(TrainingConfig):
    """多模态训练配置
    
    继承自 TrainingConfig，添加多模态训练特有的配置参数。
    
    Attributes:
        gradient_accumulation_steps: 梯度累积步数
        mixed_precision: 是否使用混合精度训练
        freeze_vision_encoder: 是否冻结视觉编码器
        freeze_llm: 是否冻结语言模型
        warmup_ratio: 学习率预热比例（相对于总训练步数）
        scheduler_type: 学习率调度器类型
        label_smoothing: 标签平滑系数
    """
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    freeze_vision_encoder: bool = False
    freeze_llm: bool = False
    warmup_ratio: float = 0.0
    scheduler_type: str = 'cosine'  # 'linear', 'cosine', 'constant'
    label_smoothing: float = 0.0


# =============================================================================
# 学习率调度器工厂
# =============================================================================

def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建带线性预热的线性衰减学习率调度器
    
    学习率变化：
    - 预热阶段：从 0 线性增加到初始学习率
    - 衰减阶段：从初始学习率线性衰减到 0
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        last_epoch: 上一个 epoch（用于恢复训练）
    
    Returns:
        LambdaLR 调度器
    
    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_linear_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=1000, num_training_steps=10000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / 
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建带线性预热的余弦退火学习率调度器
    
    学习率变化：
    - 预热阶段：从 0 线性增加到初始学习率
    - 衰减阶段：按余弦曲线从初始学习率衰减到 0
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期数（默认 0.5，即半个周期）
        last_epoch: 上一个 epoch（用于恢复训练）
    
    Returns:
        LambdaLR 调度器
    
    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_cosine_schedule_with_warmup(
        ...     optimizer, num_warmup_steps=1000, num_training_steps=10000
        ... )
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    创建带线性预热的常数学习率调度器
    
    学习率变化：
    - 预热阶段：从 0 线性增加到初始学习率
    - 之后保持常数
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        last_epoch: 上一个 epoch（用于恢复训练）
    
    Returns:
        LambdaLR 调度器
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    学习率调度器工厂函数
    
    根据类型创建对应的学习率调度器。
    
    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型 ('linear', 'cosine', 'constant')
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        **kwargs: 额外参数传递给具体调度器
    
    Returns:
        学习率调度器
    
    Raises:
        ValueError: 如果 scheduler_type 不支持
    
    Examples:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = create_scheduler(
        ...     optimizer, 'cosine', num_warmup_steps=1000, num_training_steps=10000
        ... )
    """
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'linear':
        return get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == 'cosine':
        return get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, **kwargs
        )
    elif scheduler_type == 'constant':
        return get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps, **kwargs
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported types: 'linear', 'cosine', 'constant'"
        )


# =============================================================================
# 梯度裁剪工具
# =============================================================================

def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> torch.Tensor:
    """
    对模型参数进行梯度裁剪
    
    Args:
        model: 模型
        max_norm: 梯度范数的最大值
        norm_type: 范数类型（默认 L2 范数）
    
    Returns:
        裁剪前的梯度总范数
    
    Examples:
        >>> total_norm = clip_grad_norm(model, max_norm=1.0)
        >>> print(f"Gradient norm: {total_norm:.4f}")
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type=norm_type
    )


def clip_grad_value(
    model: nn.Module,
    clip_value: float
) -> None:
    """
    对模型参数进行梯度值裁剪
    
    将梯度值裁剪到 [-clip_value, clip_value] 范围内。
    
    Args:
        model: 模型
        clip_value: 裁剪值
    
    Examples:
        >>> clip_grad_value(model, clip_value=1.0)
    """
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)


# =============================================================================
# 检查点管理工具
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    global_step: int,
    checkpoint_dir: str,
    model_name: str = 'model',
    extra_state: Optional[Dict[str, Any]] = None
) -> str:
    """
    保存训练检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        epoch: 当前 epoch
        global_step: 全局训练步数
        checkpoint_dir: 检查点保存目录
        model_name: 模型名称（用于文件命名）
        extra_state: 额外需要保存的状态
    
    Returns:
        检查点文件路径
    
    Examples:
        >>> path = save_checkpoint(
        ...     model, optimizer, scheduler,
        ...     epoch=5, global_step=10000,
        ...     checkpoint_dir='./checkpoints',
        ...     model_name='clip'
        ... )
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{model_name}_epoch{epoch}_step{global_step}.pt"
    )
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if extra_state is not None:
        checkpoint.update(extra_state)
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    加载训练检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        model: 模型
        optimizer: 优化器（可选）
        scheduler: 学习率调度器（可选）
        device: 加载设备
    
    Returns:
        检查点中的额外状态（epoch, global_step 等）
    
    Examples:
        >>> state = load_checkpoint(
        ...     'checkpoints/clip_epoch5_step10000.pt',
        ...     model, optimizer, scheduler
        ... )
        >>> start_epoch = state['epoch']
    """
    map_location = device if device is not None else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    # 返回额外状态
    return {
        k: v for k, v in checkpoint.items()
        if k not in ['model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict']
    }


def get_latest_checkpoint(checkpoint_dir: str, model_name: str = 'model') -> Optional[str]:
    """
    获取最新的检查点文件路径
    
    Args:
        checkpoint_dir: 检查点目录
        model_name: 模型名称
    
    Returns:
        最新检查点的路径，如果没有找到则返回 None
    
    Examples:
        >>> latest = get_latest_checkpoint('./checkpoints', 'clip')
        >>> if latest:
        ...     load_checkpoint(latest, model)
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith(model_name) and f.endswith('.pt')
    ]
    
    if not checkpoints:
        return None
    
    # 按修改时间排序，返回最新的
    checkpoints.sort(
        key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)),
        reverse=True
    )
    
    return os.path.join(checkpoint_dir, checkpoints[0])


# =============================================================================
# 日志记录工具
# =============================================================================

def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    设置日志记录
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
        log_format: 日志格式（可选）
    
    Examples:
        >>> setup_logging(log_level=logging.DEBUG, log_file='training.log')
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )


def log_metrics(
    metrics: Dict[str, float],
    step: int,
    prefix: str = 'train'
) -> None:
    """
    记录训练指标
    
    Args:
        metrics: 指标字典
        step: 当前步数
        prefix: 指标前缀
    
    Examples:
        >>> log_metrics({'loss': 0.5, 'accuracy': 0.9}, step=1000, prefix='train')
    """
    metrics_str = ', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])
    logger.info(f"[{prefix}] Step {step}: {metrics_str}")


# =============================================================================
# 指标跟踪工具
# =============================================================================

class MetricTracker:
    """
    训练指标跟踪器
    
    用于跟踪和计算训练过程中的各种指标（如 loss、accuracy 等）。
    支持计算移动平均和累积平均。
    
    Attributes:
        metrics: 指标名称到值列表的映射
        window_size: 移动平均窗口大小
    
    Examples:
        >>> tracker = MetricTracker(window_size=100)
        >>> tracker.update('loss', 0.5)
        >>> tracker.update('accuracy', 0.9)
        >>> print(tracker.get_average('loss'))
        >>> tracker.reset()
    """
    
    def __init__(self, window_size: int = 100):
        """
        初始化指标跟踪器
        
        Args:
            window_size: 移动平均窗口大小
        """
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
    
    def update(self, name: str, value: float) -> None:
        """
        更新指标值
        
        Args:
            name: 指标名称
            value: 指标值
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def update_dict(self, metrics: Dict[str, float]) -> None:
        """
        批量更新指标值
        
        Args:
            metrics: 指标字典
        """
        for name, value in metrics.items():
            self.update(name, value)
    
    def get_average(self, name: str, window: Optional[int] = None) -> float:
        """
        获取指标的平均值
        
        Args:
            name: 指标名称
            window: 窗口大小（None 表示使用默认窗口，-1 表示全部）
        
        Returns:
            平均值
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        
        values = self.metrics[name]
        if window is None:
            window = self.window_size
        
        if window == -1 or window >= len(values):
            return sum(values) / len(values)
        
        return sum(values[-window:]) / window
    
    def get_all_averages(self, window: Optional[int] = None) -> Dict[str, float]:
        """
        获取所有指标的平均值
        
        Args:
            window: 窗口大小
        
        Returns:
            指标名称到平均值的映射
        """
        return {name: self.get_average(name, window) for name in self.metrics}
    
    def get_latest(self, name: str) -> float:
        """
        获取指标的最新值
        
        Args:
            name: 指标名称
        
        Returns:
            最新值
        """
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def reset(self, name: Optional[str] = None) -> None:
        """
        重置指标
        
        Args:
            name: 指标名称（None 表示重置所有）
        """
        if name is None:
            self.metrics = {}
        elif name in self.metrics:
            self.metrics[name] = []


# =============================================================================
# 模型冻结/解冻工具
# =============================================================================

def freeze_module(module: nn.Module) -> None:
    """
    冻结模块的所有参数
    
    Args:
        module: 要冻结的模块
    
    Examples:
        >>> freeze_module(model.vision_encoder)
    """
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """
    解冻模块的所有参数
    
    Args:
        module: 要解冻的模块
    
    Examples:
        >>> unfreeze_module(model.vision_encoder)
    """
    for param in module.parameters():
        param.requires_grad = True


def get_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """
    获取模型中所有可训练的参数
    
    Args:
        model: 模型
    
    Returns:
        可训练参数列表
    
    Examples:
        >>> trainable = get_trainable_params(model)
        >>> optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    """
    return [p for p in model.parameters() if p.requires_grad]


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    统计模型参数数量
    
    Args:
        model: 模型
        trainable_only: 是否只统计可训练参数
    
    Returns:
        参数数量
    
    Examples:
        >>> total = count_parameters(model)
        >>> trainable = count_parameters(model, trainable_only=True)
        >>> print(f"Total: {total:,}, Trainable: {trainable:,}")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


# =============================================================================
# 优化器工具
# =============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adamw',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    **kwargs
) -> Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_type: 优化器类型 ('adam', 'adamw', 'sgd')
        learning_rate: 学习率
        weight_decay: 权重衰减
        betas: Adam 的 beta 参数
        eps: Adam 的 epsilon 参数
        **kwargs: 额外参数
    
    Returns:
        优化器
    
    Raises:
        ValueError: 如果 optimizer_type 不支持
    
    Examples:
        >>> optimizer = create_optimizer(model, 'adamw', learning_rate=1e-4)
    """
    params = get_trainable_params(model)
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        return torch.optim.Adam(
            params, lr=learning_rate, betas=betas, eps=eps,
            weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type == 'adamw':
        return torch.optim.AdamW(
            params, lr=learning_rate, betas=betas, eps=eps,
            weight_decay=weight_decay, **kwargs
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            params, lr=learning_rate, weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9), **kwargs
        )
    else:
        raise ValueError(
            f"Unknown optimizer type: {optimizer_type}. "
            f"Supported types: 'adam', 'adamw', 'sgd'"
        )


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    # 从 bert_gpt_from_scratch 复用的配置
    'TrainingConfig',
    'SFTConfig',
    # 多模态训练配置
    'MultimodalTrainingConfig',
    # 学习率调度器
    'get_linear_schedule_with_warmup',
    'get_cosine_schedule_with_warmup',
    'get_constant_schedule_with_warmup',
    'create_scheduler',
    # 梯度裁剪
    'clip_grad_norm',
    'clip_grad_value',
    # 检查点管理
    'save_checkpoint',
    'load_checkpoint',
    'get_latest_checkpoint',
    # 日志记录
    'setup_logging',
    'log_metrics',
    # 指标跟踪
    'MetricTracker',
    # 模型冻结/解冻
    'freeze_module',
    'unfreeze_module',
    'get_trainable_params',
    'count_parameters',
    # 优化器
    'create_optimizer',
]
