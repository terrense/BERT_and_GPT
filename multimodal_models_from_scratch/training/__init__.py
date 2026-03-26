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

from .multimodal_pretrain import (
    MultimodalPreTrainer,
    MultimodalPreTrainingConfig,
    compute_itc_loss,
    compute_itm_loss,
    compute_itg_loss,
    sample_hard_negatives,
    compute_itc_accuracy,
    compute_itm_accuracy,
)

__all__ = [
    # Contrastive learning
    'ContrastiveTrainer',
    'ContrastiveTrainingConfig',
    'info_nce_loss',
    'compute_contrastive_accuracy',
    # Multimodal pretraining
    'MultimodalPreTrainer',
    'MultimodalPreTrainingConfig',
    'compute_itc_loss',
    'compute_itm_loss',
    'compute_itg_loss',
    'sample_hard_negatives',
    'compute_itc_accuracy',
    'compute_itm_accuracy',
]
