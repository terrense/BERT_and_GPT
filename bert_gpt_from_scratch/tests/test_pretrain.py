"""
预训练单元测试
"""

import pytest
import torch

from bert_gpt_from_scratch.tokenizer import SimpleTokenizer
from bert_gpt_from_scratch.training.pretrain import (
    prepare_mlm_data,
    prepare_nsp_data,
    prepare_nwp_data
)


class TestMLMDataPreprocessing:
    """MLM 数据预处理测试"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建 tokenizer"""
        return SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
    
    def test_mask_ratio(self, tokenizer):
        """测试掩码比例约为 15%"""
        # 创建较长的输入以获得稳定的统计
        batch, seq_len = 10, 100
        input_ids = torch.randint(
            tokenizer.NUM_SPECIAL_TOKENS,
            tokenizer.vocab_size,
            (batch, seq_len)
        )
        
        _, mlm_labels, mask_positions = prepare_mlm_data(
            input_ids, tokenizer, mask_prob=0.15
        )
        
        # 计算实际掩码比例
        actual_ratio = mask_positions.float().mean().item()
        
        # 允许一定误差（10% - 20%）
        assert 0.10 < actual_ratio < 0.20
    
    def test_special_tokens_not_masked(self, tokenizer):
        """测试特殊 token 不被掩码"""
        batch, seq_len = 2, 20
        input_ids = torch.randint(
            tokenizer.NUM_SPECIAL_TOKENS,
            tokenizer.vocab_size,
            (batch, seq_len)
        )
        
        # 在输入中添加特殊 token
        input_ids[0, 0] = tokenizer.cls_token_id
        input_ids[0, 10] = tokenizer.sep_token_id
        input_ids[1, 5] = tokenizer.pad_token_id
        
        _, mlm_labels, mask_positions = prepare_mlm_data(
            input_ids, tokenizer, mask_prob=0.15
        )
        
        # 特殊 token 位置不应该被掩码
        assert not mask_positions[0, 0]
        assert not mask_positions[0, 10]
        assert not mask_positions[1, 5]
    
    def test_mlm_labels_format(self, tokenizer):
        """测试 MLM 标签格式"""
        batch, seq_len = 2, 20
        input_ids = torch.randint(
            tokenizer.NUM_SPECIAL_TOKENS,
            tokenizer.vocab_size,
            (batch, seq_len)
        )
        
        _, mlm_labels, mask_positions = prepare_mlm_data(
            input_ids, tokenizer, mask_prob=0.15
        )
        
        # 非掩码位置应该是 -100
        assert (mlm_labels[~mask_positions] == -100).all()
        
        # 掩码位置应该是原始 token ID
        assert (mlm_labels[mask_positions] == input_ids[mask_positions]).all()


class TestNSPDataPreprocessing:
    """NSP 数据预处理测试"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建 tokenizer"""
        return SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
    
    def test_nsp_label_distribution(self, tokenizer):
        """测试 NSP 标签分布约为 50/50"""
        sentences = [f"sentence {i}" for i in range(100)]
        
        _, _, nsp_labels = prepare_nsp_data(sentences, tokenizer, max_length=64)
        
        # 计算正例比例
        positive_ratio = (nsp_labels == 0).float().mean().item()
        
        # 允许一定误差（30% - 70%）
        assert 0.30 < positive_ratio < 0.70
    
    def test_segment_ids_format(self, tokenizer):
        """测试 segment_ids 格式"""
        sentences = ["hello world", "this is a test"]
        
        input_ids, segment_ids, _ = prepare_nsp_data(
            sentences, tokenizer, max_length=64
        )
        
        # segment_ids 应该只包含 0 和 1
        assert set(segment_ids.unique().tolist()).issubset({0, 1})
    
    def test_output_shapes(self, tokenizer):
        """测试输出形状"""
        sentences = ["hello", "world", "test", "data"]
        max_length = 32
        
        input_ids, segment_ids, nsp_labels = prepare_nsp_data(
            sentences, tokenizer, max_length=max_length
        )
        
        # 每两个句子生成一个样本
        expected_batch = len(sentences) // 2
        
        assert input_ids.shape == (expected_batch, max_length)
        assert segment_ids.shape == (expected_batch, max_length)
        assert nsp_labels.shape == (expected_batch,)


class TestNWPDataPreprocessing:
    """NWP 数据预处理测试"""
    
    def test_sequence_shift(self):
        """测试序列偏移正确"""
        batch, seq_len = 2, 10
        input_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)
        pad_token_id = 0
        
        _, labels = prepare_nwp_data(input_ids, pad_token_id)
        
        # labels 应该是 input_ids 右移一位
        # labels[:, :-1] 应该等于 input_ids[:, 1:]
        assert (labels[:, :-1] == input_ids[:, 1:]).all()
    
    def test_padding_ignored(self):
        """测试 padding 位置被忽略"""
        batch, seq_len = 2, 10
        pad_token_id = 0
        
        input_ids = torch.randint(1, 100, (batch, seq_len))
        # 添加 padding
        input_ids[0, 7:] = pad_token_id
        input_ids[1, 5:] = pad_token_id
        
        _, labels = prepare_nwp_data(input_ids, pad_token_id)
        
        # padding 位置的 label 应该是 -100
        assert (labels[0, 6:] == -100).all()  # 7: 之后的 label
        assert (labels[1, 4:] == -100).all()  # 5: 之后的 label
