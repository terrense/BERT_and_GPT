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

from .visual_instruction import (
    VisualInstructionTrainer,
    VisualInstructionConfig,
    preprocess_instruction_data,
    create_response_only_labels,
    mask_instruction_tokens,
    compute_response_only_loss,
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IMAGE_TOKEN_ID,
)

from .detection_train import (
    DetectionTrainer,
    DetectionTrainingConfig,
    DetectionAugmentation,
    random_horizontal_flip,
    random_resize,
    color_jitter,
    normalize_image,
    compute_map_placeholder,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

from .utils import (
    # 从 bert_gpt_from_scratch 复用的配置
    TrainingConfig,
    SFTConfig,
    # 多模态训练配置
    MultimodalTrainingConfig,
    # 学习率调度器
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    create_scheduler,
    # 梯度裁剪
    clip_grad_norm,
    clip_grad_value,
    # 检查点管理
    save_checkpoint,
    load_checkpoint,
    get_latest_checkpoint,
    # 日志记录
    setup_logging,
    log_metrics,
    # 指标跟踪
    MetricTracker,
    # 模型冻结/解冻
    freeze_module,
    unfreeze_module,
    get_trainable_params,
    count_parameters,
    # 优化器
    create_optimizer,
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
    # Visual instruction fine-tuning
    'VisualInstructionTrainer',
    'VisualInstructionConfig',
    'preprocess_instruction_data',
    'create_response_only_labels',
    'mask_instruction_tokens',
    'compute_response_only_loss',
    'IGNORE_INDEX',
    'DEFAULT_IMAGE_TOKEN',
    'DEFAULT_IMAGE_TOKEN_ID',
    # Detection training
    'DetectionTrainer',
    'DetectionTrainingConfig',
    'DetectionAugmentation',
    'random_horizontal_flip',
    'random_resize',
    'color_jitter',
    'normalize_image',
    'compute_map_placeholder',
    'IMAGENET_MEAN',
    'IMAGENET_STD',
    # Training utilities (from utils.py)
    'TrainingConfig',
    'SFTConfig',
    'MultimodalTrainingConfig',
    'get_linear_schedule_with_warmup',
    'get_cosine_schedule_with_warmup',
    'get_constant_schedule_with_warmup',
    'create_scheduler',
    'clip_grad_norm',
    'clip_grad_value',
    'save_checkpoint',
    'load_checkpoint',
    'get_latest_checkpoint',
    'setup_logging',
    'log_metrics',
    'MetricTracker',
    'freeze_module',
    'unfreeze_module',
    'get_trainable_params',
    'count_parameters',
    'create_optimizer',
]
