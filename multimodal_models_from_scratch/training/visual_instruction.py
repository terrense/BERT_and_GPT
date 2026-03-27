"""
视觉指令微调训练模块

实现 LLaVA 等视觉对话模型的指令微调训练流程。

包含：
- VisualInstructionConfig 配置类
- 指令数据预处理
- 两阶段训练（Stage 1: 训练 Visual Projection，Stage 2: 全参数微调）
- 仅对 response 部分计算损失
- VisualInstructionTrainer 类

需求: 13.1, 13.2, 13.3, 13.4, 13.6, 13.7, 13.8
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert_gpt_from_scratch.config import TrainingConfig

logger = logging.getLogger(__name__)


# 默认的特殊 token
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_TOKEN_ID = -200
IGNORE_INDEX = -100


@dataclass
class VisualInstructionConfig(TrainingConfig):
    """视觉指令微调配置
    
    继承自 TrainingConfig，添加视觉指令微调特有的配置参数。
    
    Attributes:
        stage: 训练阶段 (1 或 2)
            - Stage 1: 仅训练 Visual Projection，冻结 LLM
            - Stage 2: 全参数微调（解冻 LLM）
        freeze_vision_encoder: 是否冻结视觉编码器
        freeze_llm: 是否冻结 LLM（Stage 1 为 True，Stage 2 为 False）
        gradient_accumulation_steps: 梯度累积步数
        image_token: 图像占位符 token
        image_token_id: 图像占位符 token ID
        system_prompt: 系统提示（可选）
    """
    stage: int = 1
    freeze_vision_encoder: bool = True
    freeze_llm: bool = True
    gradient_accumulation_steps: int = 1
    image_token: str = DEFAULT_IMAGE_TOKEN
    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID
    system_prompt: Optional[str] = None


def preprocess_instruction_data(
    conversations: List[Dict[str, str]],
    tokenizer: Any,
    image_token: str = DEFAULT_IMAGE_TOKEN,
    image_token_id: int = DEFAULT_IMAGE_TOKEN_ID,
    system_prompt: Optional[str] = None,
    max_length: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预处理指令数据
    
    将对话格式转换为模型输入，并创建 labels（仅对 response 部分计算损失）。
    
    数据格式：
    {"image": path, "conversations": [
        {"from": "human", "value": "..."},
        {"from": "gpt", "value": "..."}
    ]}
    
    Args:
        conversations: 对话列表，格式为 [{'from': 'human/gpt', 'value': str}, ...]
        tokenizer: 分词器
        image_token: 图像占位符 token
        image_token_id: 图像占位符 token ID
        system_prompt: 系统提示（可选）
        max_length: 最大序列长度（可选）
    
    Returns:
        input_ids: 输入 token ID，形状为 (1, seq_len)
        labels: 训练标签，形状为 (1, seq_len)，instruction 部分为 -100
    """
    # 构建完整的对话文本和标签掩码
    all_input_ids = []
    all_labels = []
    
    # 添加系统提示（如果有）
    if system_prompt:
        system_text = f"System: {system_prompt}\n\n"
        system_ids = tokenizer.encode(system_text)
        all_input_ids.extend(system_ids)
        # 系统提示不计算损失
        all_labels.extend([IGNORE_INDEX] * len(system_ids))
    
    # 处理每轮对话
    for turn in conversations:
        role = turn.get('from', turn.get('role', ''))
        content = turn.get('value', turn.get('content', ''))
        
        if role in ('human', 'user'):
            # 用户输入
            turn_text = f"User: {content}\n"
            turn_ids = tokenizer.encode(turn_text)
            all_input_ids.extend(turn_ids)
            # 用户输入不计算损失
            all_labels.extend([IGNORE_INDEX] * len(turn_ids))
            
        elif role in ('gpt', 'assistant'):
            # 助手回复
            turn_text = f"Assistant: {content}\n"
            turn_ids = tokenizer.encode(turn_text)
            all_input_ids.extend(turn_ids)
            # 助手回复计算损失
            all_labels.extend(turn_ids)
    
    # 截断到最大长度
    if max_length is not None and len(all_input_ids) > max_length:
        all_input_ids = all_input_ids[:max_length]
        all_labels = all_labels[:max_length]
    
    # 转换为张量
    input_ids = torch.tensor([all_input_ids], dtype=torch.long)
    labels = torch.tensor([all_labels], dtype=torch.long)
    
    return input_ids, labels


def create_response_only_labels(
    input_ids: torch.Tensor,
    response_start_positions: List[int],
    response_end_positions: List[int]
) -> torch.Tensor:
    """
    创建仅对 response 部分计算损失的 labels
    
    Args:
        input_ids: 输入 token ID，形状为 (batch, seq_len)
        response_start_positions: 每个样本中 response 开始位置列表
        response_end_positions: 每个样本中 response 结束位置列表
    
    Returns:
        labels: 训练标签，形状为 (batch, seq_len)，非 response 部分为 -100
    """
    batch_size, seq_len = input_ids.shape
    
    # 初始化 labels 为 -100（全部忽略）
    labels = torch.full_like(input_ids, fill_value=IGNORE_INDEX)
    
    # 对于每个样本，只在 response 部分设置真实标签
    for i in range(batch_size):
        if i < len(response_start_positions) and i < len(response_end_positions):
            start = response_start_positions[i]
            end = response_end_positions[i]
            if start < seq_len and end <= seq_len:
                labels[i, start:end] = input_ids[i, start:end]
    
    return labels


def mask_instruction_tokens(
    labels: torch.Tensor,
    instruction_mask: torch.Tensor
) -> torch.Tensor:
    """
    将 instruction 部分的 labels 设为 -100
    
    Args:
        labels: 原始 labels，形状为 (batch, seq_len)
        instruction_mask: 指令掩码，形状为 (batch, seq_len)，1 表示指令部分
    
    Returns:
        masked_labels: 掩码后的 labels
    """
    masked_labels = labels.clone()
    masked_labels[instruction_mask.bool()] = IGNORE_INDEX
    return masked_labels


def compute_response_only_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """
    计算仅对 response 部分的语言模型损失
    
    Args:
        logits: 模型输出 logits，形状为 (batch, seq_len, vocab_size)
        labels: 训练标签，形状为 (batch, seq_len)，非 response 部分为 ignore_index
        ignore_index: 忽略的标签索引
    
    Returns:
        loss: 语言模型损失
    """
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # 计算交叉熵损失
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index
    )
    
    return loss


class VisualInstructionTrainer:
    """
    视觉指令微调训练器
    
    用于训练 LLaVA 等视觉对话模型。
    
    特性：
    - 支持两阶段训练
        - Stage 1: 仅训练 Visual Projection，冻结 LLM
        - Stage 2: 全参数微调（解冻 LLM）
    - 仅对 response 部分计算损失
    - 支持多轮对话格式
    - 梯度累积支持
    - 训练日志记录
    
    Args:
        model: LLaVA 或 Flamingo 模型
        config: VisualInstructionConfig 训练配置
        tokenizer: 分词器（可选，用于数据预处理）
        image_processor: 图像预处理器（可选）
    
    Attributes:
        model: 训练的模型
        config: 训练配置
        optimizer: AdamW 优化器
        scheduler: 学习率调度器（可选）
        global_step: 全局训练步数
        accumulation_step: 当前梯度累积步数
        current_stage: 当前训练阶段
    
    Examples:
        >>> from multimodal_models_from_scratch.multimodal.llava import LLaVAModel
        >>> from multimodal_models_from_scratch.config import LLaVAConfig
        >>> config = LLaVAConfig()
        >>> model = LLaVAModel(config)
        >>> train_config = VisualInstructionConfig(
        ...     learning_rate=1e-4,
        ...     batch_size=32,
        ...     stage=1
        ... )
        >>> trainer = VisualInstructionTrainer(model, train_config)
        >>> # trainer.train(dataloader, num_epochs=10)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: VisualInstructionConfig,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[Any] = None
    ):
        """
        初始化视觉指令微调训练器
        
        Args:
            model: LLaVA 或 Flamingo 模型
            config: 训练配置
            tokenizer: 分词器（可选）
            image_processor: 图像预处理器（可选）
        """
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # 当前训练阶段
        self.current_stage = config.stage
        
        # 根据阶段配置冻结参数
        self._configure_stage(config.stage)
        
        # 优化器（只优化需要梯度的参数）
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器（可选）
        self.scheduler = None
        
        # 训练状态
        self.global_step = 0
        self.accumulation_step = 0
    
    def _configure_stage(self, stage: int) -> None:
        """
        根据训练阶段配置模型参数冻结
        
        Args:
            stage: 训练阶段 (1 或 2)
        """
        if stage == 1:
            # Stage 1: 仅训练 Visual Projection，冻结 Vision Encoder 和 LLM
            self._freeze_vision_encoder()
            self._freeze_llm()
            self._unfreeze_visual_projection()
            logger.info("Stage 1: Training Visual Projection only")
        elif stage == 2:
            # Stage 2: 全参数微调
            if self.config.freeze_vision_encoder:
                self._freeze_vision_encoder()
            else:
                self._unfreeze_vision_encoder()
            self._unfreeze_llm()
            self._unfreeze_visual_projection()
            logger.info("Stage 2: Full fine-tuning")
        else:
            raise ValueError(f"Invalid stage: {stage}. Must be 1 or 2.")
        
        self.current_stage = stage
    
    def _freeze_vision_encoder(self) -> None:
        """冻结视觉编码器的所有参数"""
        if hasattr(self.model, 'vision_encoder'):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = False
            logger.info("Vision encoder frozen")
    
    def _unfreeze_vision_encoder(self) -> None:
        """解冻视觉编码器的所有参数"""
        if hasattr(self.model, 'vision_encoder'):
            for param in self.model.vision_encoder.parameters():
                param.requires_grad = True
            logger.info("Vision encoder unfrozen")
    
    def _freeze_llm(self) -> None:
        """冻结 LLM 的所有参数"""
        if hasattr(self.model, 'llm'):
            for param in self.model.llm.parameters():
                param.requires_grad = False
            logger.info("LLM frozen")
    
    def _unfreeze_llm(self) -> None:
        """解冻 LLM 的所有参数"""
        if hasattr(self.model, 'llm'):
            for param in self.model.llm.parameters():
                param.requires_grad = True
            logger.info("LLM unfrozen")
    
    def _unfreeze_visual_projection(self) -> None:
        """解冻 Visual Projection 的所有参数"""
        if hasattr(self.model, 'visual_projection'):
            for param in self.model.visual_projection.parameters():
                param.requires_grad = True
            logger.info("Visual projection unfrozen")
    
    def switch_stage(self, stage: int) -> None:
        """
        切换训练阶段
        
        Args:
            stage: 目标训练阶段 (1 或 2)
        """
        if stage == self.current_stage:
            logger.info(f"Already in stage {stage}")
            return
        
        self._configure_stage(stage)
        
        # 重新创建优化器（因为可训练参数变了）
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # 重置调度器
        self.scheduler = None
        
        logger.info(f"Switched to stage {stage}")
    
    def prepare_instruction_data(
        self,
        conversations: List[Dict[str, str]],
        pixel_values: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        准备指令微调数据
        
        Args:
            conversations: 对话列表，格式为 [{'from': 'human/gpt', 'value': str}, ...]
            pixel_values: 图像张量，形状为 (batch, 3, H, W)
        
        Returns:
            Dict 包含:
            - 'input_ids': (batch, seq_len)
            - 'attention_mask': (batch, seq_len)
            - 'labels': (batch, seq_len)，instruction 部分为 -100
            - 'pixel_values': (batch, 3, H, W)（如果提供）
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for prepare_instruction_data")
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=self.tokenizer,
            image_token=self.config.image_token,
            image_token_id=self.config.image_token_id,
            system_prompt=self.config.system_prompt
        )
        
        # 创建 attention_mask
        attention_mask = torch.ones_like(input_ids)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
        if pixel_values is not None:
            result['pixel_values'] = pixel_values
        
        return result
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        单步训练（仅对 response 计算损失）
        
        执行一个 batch 的前向传播、损失计算和反向传播。
        支持梯度累积：只有在累积足够步数后才更新参数。
        
        Args:
            batch: 包含以下键的字典：
                - 'pixel_values': (batch, 3, H, W) 输入图像
                - 'input_ids': (batch, seq_len) 输入 token IDs
                - 'attention_mask': (batch, seq_len) 注意力掩码（可选）
                - 'labels': (batch, seq_len) 目标 token IDs，instruction 部分为 -100
        
        Returns:
            Dict 包含:
            - 'loss': 损失值
            - 'did_update': 是否执行了参数更新
        """
        self.model.train()
        
        pixel_values = batch.get('pixel_values')
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')
        labels = batch['labels']
        
        # 前向传播
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            image_token_index=self.config.image_token_id
        )
        
        # 获取损失
        if 'loss' in outputs:
            loss = outputs['loss']
        else:
            # 如果模型没有返回损失，手动计算
            logits = outputs['logits']
            loss = compute_response_only_loss(logits, labels)
        
        # 梯度累积：将损失除以累积步数
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        # 反向传播
        scaled_loss.backward()
        
        # 更新累积步数
        self.accumulation_step += 1
        did_update = False
        
        # 检查是否需要更新参数
        if self.accumulation_step >= self.config.gradient_accumulation_steps:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率调度器
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 重置累积步数
            self.accumulation_step = 0
            self.global_step += 1
            did_update = True
        
        return {
            'loss': loss.item(),
            'did_update': did_update
        }
    
    def train_stage1(
        self,
        dataloader: DataLoader,
        num_epochs: int
    ) -> None:
        """
        第一阶段训练：仅训练 Visual Projection
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        # 确保在 Stage 1
        if self.current_stage != 1:
            self.switch_stage(1)
        
        logger.info(f"Starting Stage 1 training for {num_epochs} epochs")
        logger.info("Training Visual Projection only (Vision Encoder and LLM frozen)")
        
        self._train_loop(dataloader, num_epochs)
    
    def train_stage2(
        self,
        dataloader: DataLoader,
        num_epochs: int
    ) -> None:
        """
        第二阶段训练：全参数微调
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        # 切换到 Stage 2
        if self.current_stage != 2:
            self.switch_stage(2)
        
        logger.info(f"Starting Stage 2 training for {num_epochs} epochs")
        logger.info("Full fine-tuning (LLM unfrozen)")
        
        self._train_loop(dataloader, num_epochs)
    
    def _train_loop(
        self,
        dataloader: DataLoader,
        num_epochs: int
    ) -> None:
        """
        训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                num_batches += 1
                
                # 日志记录（仅在参数更新时记录）
                if metrics['did_update'] and self.global_step % self.config.log_steps == 0:
                    logger.info(
                        f"Step {self.global_step}: Loss={metrics['loss']:.4f}"
                    )
                
                # 保存检查点
                if metrics['did_update'] and self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint()
            
            # Epoch 统计
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}: Avg Loss={avg_loss:.4f}"
            )
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        stage: Optional[int] = None
    ) -> None:
        """
        完整训练循环
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            stage: 训练阶段（可选，默认使用配置中的阶段）
        """
        if stage is not None and stage != self.current_stage:
            self.switch_stage(stage)
        
        if self.current_stage == 1:
            self.train_stage1(dataloader, num_epochs)
        else:
            self.train_stage2(dataloader, num_epochs)
    
    def save_checkpoint(self) -> None:
        """保存检查点"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"visual_instruction_stage{self.current_stage}_step_{self.global_step}.pt"
        )
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'current_stage': self.current_stage,
            'config': self.config,
            'model_type': self.model.__class__.__name__
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        
        if 'current_stage' in checkpoint:
            self.current_stage = checkpoint['current_stage']
            self._configure_stage(self.current_stage)
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def set_scheduler(
        self,
        total_steps: int,
        warmup_steps: Optional[int] = None
    ) -> None:
        """
        设置学习率调度器（warmup + cosine decay）
        
        Args:
            total_steps: 总训练步数
            warmup_steps: 预热步数（默认使用 config 中的值）
        """
        if warmup_steps is None:
            warmup_steps = self.config.warmup_steps
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                # 线性预热
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        logger.info(f"Scheduler set with {warmup_steps} warmup steps and {total_steps} total steps")
    
    def get_trainable_parameters(self) -> int:
        """
        获取可训练参数数量
        
        Returns:
            可训练参数数量
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_total_parameters(self) -> int:
        """
        获取总参数数量
        
        Returns:
            总参数数量
        """
        return sum(p.numel() for p in self.model.parameters())
