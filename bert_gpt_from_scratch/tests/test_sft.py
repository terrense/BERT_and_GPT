"""
SFT 单元测试
"""

import pytest
import torch
import tempfile
import os

from bert_gpt_from_scratch.config import BERTConfig, GPTConfig, SFTConfig
from bert_gpt_from_scratch.models.bert import BERTModel
from bert_gpt_from_scratch.models.gpt import GPTModel
from bert_gpt_from_scratch.tokenizer import SimpleTokenizer
from bert_gpt_from_scratch.training.sft import (
    SFTTrainer,
    ClassificationHead,
    prepare_instruction_labels
)


class TestSFTTrainer:
    """SFT Trainer 测试"""
    
    @pytest.fixture
    def bert_config(self):
        """创建小型 BERT 配置"""
        return BERTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0
        )
    
    @pytest.fixture
    def gpt_config(self):
        """创建小型 GPT 配置"""
        return GPTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0,
            tie_weights=True
        )
    
    @pytest.fixture
    def sft_config(self):
        """创建 SFT 配置"""
        return SFTConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            freeze_layers=1
        )
    
    @pytest.fixture
    def tokenizer(self):
        """创建 tokenizer"""
        return SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
    
    def test_load_pretrained(self, bert_config, sft_config, tokenizer):
        """测试加载预训练检查点"""
        model = BERTModel(bert_config)
        trainer = SFTTrainer(model, sft_config, tokenizer)
        
        # 保存一个临时检查点
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': bert_config,
            }, checkpoint_path)
            
            # 创建新模型并加载检查点
            new_model = BERTModel(bert_config)
            new_trainer = SFTTrainer(new_model, sft_config, tokenizer)
            new_trainer.load_pretrained(checkpoint_path)
            
            # 验证权重相同
            for (name1, param1), (name2, param2) in zip(
                model.named_parameters(),
                new_model.named_parameters()
            ):
                assert torch.allclose(param1, param2), f"Mismatch in {name1}"
    
    def test_freeze_layers_bert(self, bert_config, sft_config, tokenizer):
        """测试 BERT 层冻结功能"""
        model = BERTModel(bert_config)
        trainer = SFTTrainer(model, sft_config, tokenizer)
        
        # 冻结 1 层
        trainer.freeze_layers(1)
        
        # 检查 embedding 层被冻结
        assert not model.token_embedding.weight.requires_grad
        assert not model.segment_embedding.weight.requires_grad
        
        # 检查第一个 encoder 层被冻结
        for param in model.encoder_layers[0].parameters():
            assert not param.requires_grad
        
        # 检查第二个 encoder 层未被冻结
        for param in model.encoder_layers[1].parameters():
            assert param.requires_grad
    
    def test_freeze_layers_gpt(self, gpt_config, sft_config, tokenizer):
        """测试 GPT 层冻结功能"""
        model = GPTModel(gpt_config)
        trainer = SFTTrainer(model, sft_config, tokenizer)
        
        # 冻结 1 层
        trainer.freeze_layers(1)
        
        # 检查 embedding 层被冻结
        assert not model.token_embedding.weight.requires_grad
        
        # 检查第一个 decoder 层被冻结
        for param in model.decoder_layers[0].parameters():
            assert not param.requires_grad
        
        # 检查第二个 decoder 层未被冻结
        for param in model.decoder_layers[1].parameters():
            assert param.requires_grad


class TestClassificationHead:
    """分类头测试"""
    
    def test_output_shape(self):
        """测试输出形状"""
        d_model, num_classes = 64, 5
        head = ClassificationHead(d_model, num_classes)
        
        batch = 4
        cls_hidden_state = torch.randn(batch, d_model)
        
        logits = head(cls_hidden_state)
        
        assert logits.shape == (batch, num_classes)


class TestInstructionLabels:
    """指令标签测试"""
    
    def test_instruction_masked(self):
        """测试 instruction 部分被掩码"""
        batch, seq_len = 2, 20
        pad_token_id = 0
        
        input_ids = torch.randint(1, 100, (batch, seq_len))
        instruction_lengths = torch.tensor([5, 8])
        
        labels = prepare_instruction_labels(input_ids, instruction_lengths, pad_token_id)
        
        # instruction 部分应该是 -100
        assert (labels[0, :5] == -100).all()
        assert (labels[1, :8] == -100).all()
        
        # response 部分应该保留原始 token ID
        assert (labels[0, 5:] == input_ids[0, 5:]).all()
        assert (labels[1, 8:] == input_ids[1, 8:]).all()
    
    def test_padding_masked(self):
        """测试 padding 部分被掩码"""
        batch, seq_len = 2, 20
        pad_token_id = 0
        
        input_ids = torch.randint(1, 100, (batch, seq_len))
        # 添加 padding
        input_ids[0, 15:] = pad_token_id
        input_ids[1, 12:] = pad_token_id
        
        instruction_lengths = torch.tensor([3, 3])
        
        labels = prepare_instruction_labels(input_ids, instruction_lengths, pad_token_id)
        
        # padding 部分应该是 -100
        assert (labels[0, 15:] == -100).all()
        assert (labels[1, 12:] == -100).all()
