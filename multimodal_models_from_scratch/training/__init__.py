"""
Training components.

Modules:
- contrastive: Contrastive learning training
- multimodal_pretrain: Multimodal pretraining
- visual_instruction: Visual instruction fine-tuning
- detection_train: Object detection training
- utils: Training utility functions
"""

from .contrastive import (
    ContrastiveTrainer,
    ContrastiveTrainingConfig,
    info_nce_loss,
    compute_contrastive_accuracy,
)

__all__ = [
    'ContrastiveTrainer',
    'ContrastiveTrainingConfig',
    'info_nce_loss',
    'compute_contrastive_accuracy',
]
