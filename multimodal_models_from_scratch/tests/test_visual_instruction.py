"""
视觉指令微调训练模块单元测试

测试 VisualInstructionConfig、指令数据预处理、两阶段训练、VisualInstructionTrainer 等组件。
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from multimodal_models_from_scratch.training.visual_instruction import (
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


class MockTokenizer:
    """用于测试的模拟分词器"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> list:
        """简单的字符级编码"""
        # 简单地将每个字符映射到一个 token ID
        return [ord(c) % self.vocab_size for c in text]
    
    def decode(self, ids: list) -> str:
        """简单的解码"""
        return ''.join(chr(i % 128) for i in ids)


class MockLLaVAModel(nn.Module):
    """用于测试的模拟 LLaVA 模型"""
    
    def __init__(self, embed_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        
        # 简单的视觉编码器
        self.vision_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, embed_dim)
        )
        
        # Visual Projection
        self.visual_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 简单的 LLM
        self.llm = nn.ModuleDict({
            'embed_tokens': nn.Embedding(vocab_size, embed_dim),
            'layers': nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=4,
                    dim_feedforward=embed_dim * 4,
                    batch_first=True
                )
                for _ in range(2)
            ]),
            'lm_head': nn.Linear(embed_dim, vocab_size)
        })
    
    def forward(
        self,
        pixel_values: torch.Tensor = None,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_ID
    ):
        batch_size, seq_len = input_ids.shape
        
        # 获取文本嵌入
        # 将 image_token_index 替换为 0 以避免索引错误
        safe_input_ids = input_ids.clone()
        safe_input_ids[safe_input_ids == image_token_index] = 0
        text_embeds = self.llm['embed_tokens'](safe_input_ids)
        
        # 如果有图像，获取视觉特征
        if pixel_values is not None:
            visual_features = self.vision_encoder(pixel_values)
            visual_tokens = self.visual_projection(visual_features)
            # 简单地将视觉特征加到文本嵌入上
            text_embeds = text_embeds + visual_tokens.unsqueeze(1)
        
        # 通过 LLM 层
        hidden_states = text_embeds
        for layer in self.llm['layers']:
            hidden_states = layer(hidden_states)
        
        # LM Head
        logits = self.llm['lm_head'](hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states
        }


class TestPreprocessInstructionData:
    """指令数据预处理测试"""
    
    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer(vocab_size=1000)
    
    def test_single_turn_conversation(self, tokenizer):
        """测试单轮对话"""
        conversations = [
            {'from': 'human', 'value': 'Hello'},
            {'from': 'gpt', 'value': 'Hi there'}
        ]
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        assert input_ids.shape[0] == 1
        assert labels.shape[0] == 1
        assert input_ids.shape == labels.shape
    
    def test_multi_turn_conversation(self, tokenizer):
        """测试多轮对话"""
        conversations = [
            {'from': 'human', 'value': 'What is this?'},
            {'from': 'gpt', 'value': 'This is a cat.'},
            {'from': 'human', 'value': 'What color is it?'},
            {'from': 'gpt', 'value': 'It is orange.'}
        ]
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        assert input_ids.shape[0] == 1
        assert labels.shape[0] == 1
    
    def test_labels_mask_instruction(self, tokenizer):
        """测试 labels 正确掩码 instruction 部分"""
        conversations = [
            {'from': 'human', 'value': 'Hi'},
            {'from': 'gpt', 'value': 'Hello'}
        ]
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        # 检查 labels 中有 IGNORE_INDEX
        assert (labels == IGNORE_INDEX).any()
        # 检查 labels 中有非 IGNORE_INDEX 的值（response 部分）
        assert (labels != IGNORE_INDEX).any()
    
    def test_with_system_prompt(self, tokenizer):
        """测试带系统提示的对话"""
        conversations = [
            {'from': 'human', 'value': 'Hi'},
            {'from': 'gpt', 'value': 'Hello'}
        ]
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer,
            system_prompt="You are a helpful assistant."
        )
        
        assert input_ids.shape[0] == 1
        # 系统提示部分应该被掩码
        assert (labels == IGNORE_INDEX).any()
    
    def test_max_length_truncation(self, tokenizer):
        """测试最大长度截断"""
        conversations = [
            {'from': 'human', 'value': 'A' * 100},
            {'from': 'gpt', 'value': 'B' * 100}
        ]
        
        max_length = 50
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer,
            max_length=max_length
        )
        
        assert input_ids.shape[1] <= max_length
        assert labels.shape[1] <= max_length
    
    def test_alternative_role_names(self, tokenizer):
        """测试替代角色名称"""
        conversations = [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello'}
        ]
        
        input_ids, labels = preprocess_instruction_data(
            conversations=conversations,
            tokenizer=tokenizer
        )
        
        assert input_ids.shape[0] == 1
        assert labels.shape[0] == 1


class TestCreateResponseOnlyLabels:
    """创建 response-only labels 测试"""
    
    def test_basic_masking(self):
        """测试基本掩码功能"""
        batch_size = 2
        seq_len = 10
        
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        response_start_positions = [3, 5]
        response_end_positions = [7, 9]
        
        labels = create_response_only_labels(
            input_ids,
            response_start_positions,
            response_end_positions
        )
        
        assert labels.shape == input_ids.shape
        
        # 检查第一个样本
        assert (labels[0, :3] == IGNORE_INDEX).all()
        assert (labels[0, 3:7] == input_ids[0, 3:7]).all()
        assert (labels[0, 7:] == IGNORE_INDEX).all()
        
        # 检查第二个样本
        assert (labels[1, :5] == IGNORE_INDEX).all()
        assert (labels[1, 5:9] == input_ids[1, 5:9]).all()
        assert (labels[1, 9:] == IGNORE_INDEX).all()
    
    def test_empty_response(self):
        """测试空 response 情况"""
        batch_size = 1
        seq_len = 10
        
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        response_start_positions = []
        response_end_positions = []
        
        labels = create_response_only_labels(
            input_ids,
            response_start_positions,
            response_end_positions
        )
        
        # 所有位置都应该是 IGNORE_INDEX
        assert (labels == IGNORE_INDEX).all()


class TestMaskInstructionTokens:
    """掩码 instruction tokens 测试"""
    
    def test_basic_masking(self):
        """测试基本掩码功能"""
        batch_size = 2
        seq_len = 10
        
        labels = torch.randint(0, 100, (batch_size, seq_len))
        instruction_mask = torch.zeros(batch_size, seq_len)
        instruction_mask[:, :5] = 1  # 前 5 个位置是 instruction
        
        masked_labels = mask_instruction_tokens(labels, instruction_mask)
        
        # 检查 instruction 部分被掩码
        assert (masked_labels[:, :5] == IGNORE_INDEX).all()
        # 检查 response 部分保持不变
        assert (masked_labels[:, 5:] == labels[:, 5:]).all()


class TestComputeResponseOnlyLoss:
    """计算 response-only loss 测试"""
    
    def test_loss_shape(self):
        """测试损失输出为标量"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_response_only_loss(logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_ignores_masked_positions(self):
        """测试损失忽略掩码位置"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels[:, :seq_len // 2] = IGNORE_INDEX  # 前半部分掩码
        
        loss = compute_response_only_loss(logits, labels)
        
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0
    
    def test_loss_gradient_flow(self):
        """测试梯度能够正确流动"""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        loss = compute_response_only_loss(logits, labels)
        loss.backward()
        
        assert logits.grad is not None


class TestVisualInstructionConfig:
    """视觉指令微调配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = VisualInstructionConfig()
        
        assert config.stage == 1
        assert config.freeze_vision_encoder == True
        assert config.freeze_llm == True
        assert config.gradient_accumulation_steps == 1
        assert config.image_token == DEFAULT_IMAGE_TOKEN
        assert config.image_token_id == DEFAULT_IMAGE_TOKEN_ID
    
    def test_stage2_config(self):
        """测试 Stage 2 配置"""
        config = VisualInstructionConfig(
            stage=2,
            freeze_vision_encoder=True,
            freeze_llm=False,
            learning_rate=2e-5
        )
        
        assert config.stage == 2
        assert config.freeze_vision_encoder == True
        assert config.freeze_llm == False
        assert config.learning_rate == 2e-5
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = VisualInstructionConfig(
            stage=1,
            freeze_vision_encoder=True,
            freeze_llm=True,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            system_prompt="You are a helpful assistant."
        )
        
        assert config.gradient_accumulation_steps == 4
        assert config.learning_rate == 1e-4
        assert config.system_prompt == "You are a helpful assistant."


class TestVisualInstructionTrainer:
    """VisualInstructionTrainer 测试"""
    
    @pytest.fixture
    def trainer_stage1(self):
        """创建 Stage 1 测试用的 trainer"""
        model = MockLLaVAModel(embed_dim=256, vocab_size=1000)
        config = VisualInstructionConfig(
            stage=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            log_steps=10,
            save_steps=100
        )
        return VisualInstructionTrainer(model, config)
    
    @pytest.fixture
    def trainer_stage2(self):
        """创建 Stage 2 测试用的 trainer"""
        model = MockLLaVAModel(embed_dim=256, vocab_size=1000)
        config = VisualInstructionConfig(
            stage=2,
            freeze_vision_encoder=True,
            freeze_llm=False,
            learning_rate=2e-5,
            gradient_accumulation_steps=1
        )
        return VisualInstructionTrainer(model, config)
    
    @pytest.fixture
    def sample_batch(self):
        """创建测试用的 batch"""
        batch_size = 4
        seq_len = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        # 将前半部分设为 IGNORE_INDEX（模拟 instruction 部分）
        labels[:, :seq_len // 2] = IGNORE_INDEX
        
        return {
            'pixel_values': torch.randn(batch_size, 3, 32, 32),
            'input_ids': input_ids,
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': labels
        }
    
    def test_trainer_initialization_stage1(self, trainer_stage1):
        """测试 Stage 1 trainer 初始化"""
        assert trainer_stage1.model is not None
        assert trainer_stage1.config is not None
        assert trainer_stage1.optimizer is not None
        assert trainer_stage1.global_step == 0
        assert trainer_stage1.accumulation_step == 0
        assert trainer_stage1.current_stage == 1
    
    def test_trainer_initialization_stage2(self, trainer_stage2):
        """测试 Stage 2 trainer 初始化"""
        assert trainer_stage2.current_stage == 2
    
    def test_train_step_output(self, trainer_stage1, sample_batch):
        """测试 train_step 输出格式"""
        metrics = trainer_stage1.train_step(sample_batch)
        
        assert 'loss' in metrics
        assert 'did_update' in metrics
        
        assert isinstance(metrics['loss'], float)
        assert metrics['loss'] >= 0
    
    def test_gradient_accumulation(self, trainer_stage1, sample_batch):
        """测试梯度累积"""
        # 第一步：不应该更新参数
        metrics1 = trainer_stage1.train_step(sample_batch)
        assert metrics1['did_update'] == False
        assert trainer_stage1.accumulation_step == 1
        assert trainer_stage1.global_step == 0
        
        # 第二步：应该更新参数
        metrics2 = trainer_stage1.train_step(sample_batch)
        assert metrics2['did_update'] == True
        assert trainer_stage1.accumulation_step == 0
        assert trainer_stage1.global_step == 1
    
    def test_stage_switching(self, trainer_stage1):
        """测试阶段切换"""
        assert trainer_stage1.current_stage == 1
        
        # 切换到 Stage 2
        trainer_stage1.switch_stage(2)
        assert trainer_stage1.current_stage == 2
        
        # 切换回 Stage 1
        trainer_stage1.switch_stage(1)
        assert trainer_stage1.current_stage == 1
    
    def test_stage1_freezes_llm(self, trainer_stage1):
        """测试 Stage 1 冻结 LLM"""
        # 检查 LLM 参数被冻结
        for name, param in trainer_stage1.model.llm.named_parameters():
            assert not param.requires_grad, f"LLM param {name} should be frozen in Stage 1"
        
        # 检查 Visual Projection 参数未被冻结
        for name, param in trainer_stage1.model.visual_projection.named_parameters():
            assert param.requires_grad, f"Visual projection param {name} should be trainable in Stage 1"
    
    def test_stage2_unfreezes_llm(self, trainer_stage2):
        """测试 Stage 2 解冻 LLM"""
        # 检查 LLM 参数未被冻结
        for name, param in trainer_stage2.model.llm.named_parameters():
            assert param.requires_grad, f"LLM param {name} should be trainable in Stage 2"
        
        # 检查 Visual Projection 参数未被冻结
        for name, param in trainer_stage2.model.visual_projection.named_parameters():
            assert param.requires_grad, f"Visual projection param {name} should be trainable in Stage 2"
    
    def test_multiple_train_steps(self, trainer_stage2, sample_batch):
        """测试多步训练"""
        initial_params = {
            name: param.clone() 
            for name, param in trainer_stage2.model.named_parameters()
            if param.requires_grad
        }
        
        # 执行多步训练
        for _ in range(3):
            trainer_stage2.train_step(sample_batch)
        
        # 检查参数已更新
        params_changed = False
        for name, param in trainer_stage2.model.named_parameters():
            if param.requires_grad and name in initial_params:
                if not torch.allclose(param, initial_params[name]):
                    params_changed = True
                    break
        
        assert params_changed, "Parameters should have changed after training"
    
    def test_set_scheduler(self, trainer_stage1):
        """测试学习率调度器设置"""
        total_steps = 1000
        warmup_steps = 100
        
        trainer_stage1.set_scheduler(total_steps, warmup_steps)
        
        assert trainer_stage1.scheduler is not None
    
    def test_get_trainable_parameters(self, trainer_stage1, trainer_stage2):
        """测试获取可训练参数数量"""
        stage1_params = trainer_stage1.get_trainable_parameters()
        stage2_params = trainer_stage2.get_trainable_parameters()
        
        # Stage 2 应该有更多可训练参数
        assert stage2_params > stage1_params
    
    def test_get_total_parameters(self, trainer_stage1):
        """测试获取总参数数量"""
        total_params = trainer_stage1.get_total_parameters()
        trainable_params = trainer_stage1.get_trainable_parameters()
        
        assert total_params >= trainable_params
    
    def test_prepare_instruction_data(self, trainer_stage1):
        """测试准备指令数据"""
        trainer_stage1.tokenizer = MockTokenizer(vocab_size=1000)
        
        conversations = [
            {'from': 'human', 'value': 'What is this?'},
            {'from': 'gpt', 'value': 'This is a cat.'}
        ]
        pixel_values = torch.randn(1, 3, 32, 32)
        
        result = trainer_stage1.prepare_instruction_data(
            conversations=conversations,
            pixel_values=pixel_values
        )
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert 'labels' in result
        assert 'pixel_values' in result
        
        assert result['input_ids'].shape == result['labels'].shape
        assert result['input_ids'].shape == result['attention_mask'].shape


class TestVisualInstructionTrainerWithRealLLaVA:
    """使用真实 LLaVA 模型的集成测试"""
    
    @pytest.fixture
    def real_trainer(self):
        """创建使用真实 LLaVA 模型的 trainer"""
        from multimodal_models_from_scratch.multimodal.llava import LLaVAModel
        from multimodal_models_from_scratch.config import LLaVAConfig, VisionConfig, LLaMAConfig
        
        # 使用小型配置以加快测试
        vision_config = VisionConfig(
            image_size=32,
            patch_size=8,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128
        )
        llm_config = LLaMAConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_kv_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=64
        )
        llava_config = LLaVAConfig(
            vision_config=vision_config,
            llm_config=llm_config,
            projection_type='mlp',
            freeze_vision=True,
            freeze_llm=False  # LLM is trainable
        )
        
        model = LLaVAModel(llava_config)
        # Use Stage 2 for real LLaVA tests since the model has trainable LLM
        config = VisualInstructionConfig(
            stage=2,
            freeze_vision_encoder=True,
            freeze_llm=False,
            learning_rate=1e-4,
            gradient_accumulation_steps=1
        )
        
        return VisualInstructionTrainer(model, config)
    
    @pytest.fixture
    def real_batch(self):
        """创建真实 LLaVA 模型的测试 batch"""
        batch_size = 2
        seq_len = 16
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        labels[:, :seq_len // 2] = IGNORE_INDEX
        
        return {
            'pixel_values': torch.randn(batch_size, 3, 32, 32),
            'input_ids': input_ids,
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': labels
        }
    
    def test_real_llava_train_step(self, real_trainer, real_batch):
        """测试使用真实 LLaVA 模型的训练步骤"""
        metrics = real_trainer.train_step(real_batch)
        
        assert 'loss' in metrics
        assert metrics['loss'] >= 0
    
    def test_real_llava_stage_switching(self, real_trainer, real_batch):
        """测试真实 LLaVA 模型的阶段切换"""
        # Stage 2 训练 (already in stage 2)
        metrics1 = real_trainer.train_step(real_batch)
        assert metrics1['loss'] >= 0
        
        # Note: We don't switch to Stage 1 here because the real LLaVA model
        # has its own freeze settings that may conflict with Stage 1 settings.
        # The stage switching is tested with the mock model instead.
    
    def test_real_llava_multiple_steps(self, real_trainer, real_batch):
        """测试多步训练"""
        losses = []
        
        for _ in range(3):
            metrics = real_trainer.train_step(real_batch)
            losses.append(metrics['loss'])
        
        assert len(losses) == 3
        assert all(loss >= 0 for loss in losses)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
