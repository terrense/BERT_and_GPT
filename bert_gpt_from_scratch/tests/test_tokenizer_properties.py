"""
Tokenizer 属性测试

使用 Hypothesis 进行属性测试，验证 round-trip 一致性。
"""

import pytest
from hypothesis import given, strategies as st, settings

from bert_gpt_from_scratch.tokenizer import SimpleTokenizer


class TestTokenizerProperties:
    """Tokenizer 属性测试"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建包含常用字符的 tokenizer"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-"
        return SimpleTokenizer.from_chars(chars)
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_roundtrip_consistency(self, text):
        """
        属性 1: Round-trip 一致性
        
        对于所有合法文本输入，decode(encode(text)) 应产生与原始文本等价的结果。
        验证需求: 12.4
        """
        tokenizer = SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
        
        # 不添加特殊 token 的 round-trip
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip failed: '{text}' -> {token_ids} -> '{decoded}'"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_roundtrip_with_special_tokens(self, text):
        """
        属性 1 变体: 带特殊 token 的 round-trip 一致性
        
        添加特殊 token 后，跳过特殊 token 解码应得到原始文本。
        """
        tokenizer = SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
        
        # 添加特殊 token 的 round-trip
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip with special tokens failed: '{text}' -> '{decoded}'"
    
    @given(st.text(alphabet="abc", min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_encode_length_property(self, text):
        """
        属性: encode 输出长度等于输入字符数（不含特殊 token 时）
        """
        tokenizer = SimpleTokenizer.from_chars("abc")
        
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(token_ids) == len(text)
    
    @given(st.text(alphabet="abc", min_size=1, max_size=50))
    @settings(max_examples=50)
    def test_encode_with_special_tokens_length(self, text):
        """
        属性: 添加特殊 token 后长度增加 2（[CLS] 和 [SEP]）
        """
        tokenizer = SimpleTokenizer.from_chars("abc")
        
        token_ids_without = tokenizer.encode(text, add_special_tokens=False)
        token_ids_with = tokenizer.encode(text, add_special_tokens=True)
        assert len(token_ids_with) == len(token_ids_without) + 2
