"""
训练模块

包含：
- 预训练 Trainer（BERT/GPT）
- SFT Trainer
- 训练工具函数
"""

from .pretrain import (
    prepare_mlm_data,
    prepare_nsp_data,
    prepare_nwp_data,
    BERTPreTrainer,
    GPTPreTrainer,
)
from .sft import SFTTrainer, ClassificationHead, prepare_instruction_labels

__all__ = [
    'prepare_mlm_data',
    'prepare_nsp_data',
    'prepare_nwp_data',
    'BERTPreTrainer',
    'GPTPreTrainer',
    'SFTTrainer',
    'ClassificationHead',
    'prepare_instruction_labels',
]
