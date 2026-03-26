"""
模型配置类

包含 Transformer、BERT、GPT 的配置 dataclass，
以及训练和 SFT 配置。
"""

from dataclasses import dataclass


@dataclass
class TransformerConfig:
    """Transformer 基础配置
    
    包含 Transformer 模型的核心超参数配置。
    
    Attributes:
        vocab_size: 词表大小
        d_model: 模型维度（隐藏层大小）
        num_heads: 注意力头数量
        num_layers: Transformer 层数
        d_ff: 前馈网络中间层维度
        max_seq_len: 最大序列长度
        dropout_rate: Dropout 比率
    """
    vocab_size: int = 30000
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 512
    dropout_rate: float = 0.1


@dataclass
class BERTConfig(TransformerConfig):
    """BERT 专用配置
    
    继承自 TransformerConfig，添加 BERT 特有的配置参数。
    
    Attributes:
        num_segments: Segment Embedding 数量，用于区分句子对
    """
    num_segments: int = 2


@dataclass
class GPTConfig(TransformerConfig):
    """GPT 专用配置
    
    继承自 TransformerConfig，添加 GPT 特有的配置参数。
    
    Attributes:
        tie_weights: 是否绑定 Token Embedding 和 LM Head 权重
    """
    tie_weights: bool = True


@dataclass
class TrainingConfig:
    """训练配置
    
    包含预训练和微调的通用训练超参数。
    
    Attributes:
        batch_size: 批次大小
        learning_rate: 学习率
        num_epochs: 训练轮数
        warmup_steps: 学习率预热步数
        weight_decay: 权重衰减系数
        max_grad_norm: 梯度裁剪的最大范数
        save_steps: 保存检查点的步数间隔
        log_steps: 记录日志的步数间隔
        checkpoint_dir: 检查点保存目录
    """
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_steps: int = 10000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    save_steps: int = 1000
    log_steps: int = 100
    checkpoint_dir: str = './checkpoints'


@dataclass
class SFTConfig(TrainingConfig):
    """SFT（监督微调）配置
    
    继承自 TrainingConfig，添加 SFT 特有的配置参数。
    
    Attributes:
        warmup_ratio: 学习率预热比例（相对于总训练步数）
        freeze_layers: 冻结的层数（从底层开始）
        num_classes: 分类任务的类别数
    """
    warmup_ratio: float = 0.1
    freeze_layers: int = 0
    num_classes: int = 2
