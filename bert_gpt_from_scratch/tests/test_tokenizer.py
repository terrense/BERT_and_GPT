"""
Tokenizer 测试

测试内容：
- Round-trip 一致性：decode(encode(text)) 应产生与原始文本等价的结果
- 特殊 token 处理
"""

import pytest
from bert_gpt_from_scratch.tokenizer import SimpleTokenizer


class TestSimpleTokenizer:
    """SimpleTokenizer 单元测试"""
    
    @pytest.fixture
    def tokenizer(self):
        """创建包含基本字符的 tokenizer"""
        vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        return SimpleTokenizer(vocab)
    
    @pytest.fixture
    def empty_tokenizer(self):
        """创建空词表的 tokenizer"""
        return SimpleTokenizer()
    
    # ==================== 特殊 token 测试 ====================
    
    def test_special_token_ids(self, tokenizer):
        """测试特殊 token ID 正确定义"""
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
        assert tokenizer.cls_token_id == 2
        assert tokenizer.sep_token_id == 3
        assert tokenizer.mask_token_id == 4
        assert tokenizer.bos_token_id == 5
        assert tokenizer.eos_token_id == 6
    
    def test_special_tokens_in_vocab(self, tokenizer):
        """测试特殊 token 在词表中"""
        vocab = tokenizer.get_vocab()
        assert "[PAD]" in vocab
        assert "[UNK]" in vocab
        assert "[CLS]" in vocab
        assert "[SEP]" in vocab
        assert "[MASK]" in vocab
        assert "[BOS]" in vocab
        assert "[EOS]" in vocab
    
    def test_vocab_size_includes_special_tokens(self, empty_tokenizer):
        """测试词表大小包含特殊 token"""
        # 空词表应该只有 7 个特殊 token
        assert empty_tokenizer.vocab_size == 7
    
    def test_vocab_size_with_custom_vocab(self, tokenizer):
        """测试自定义词表的词表大小"""
        # 7 个特殊 token + 27 个字符 (a-z + 空格)
        assert tokenizer.vocab_size == 7 + 27
    
    # ==================== encode 测试 ====================
    
    def test_encode_basic(self, tokenizer):
        """测试基本编码功能"""
        text = "hello"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        # 每个字符应该被编码为一个 token
        assert len(token_ids) == 5
    
    def test_encode_with_special_tokens(self, tokenizer):
        """测试添加特殊 token 的编码"""
        text = "hi"
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        # [CLS] + h + i + [SEP]
        assert len(token_ids) == 4
        assert token_ids[0] == tokenizer.cls_token_id
        assert token_ids[-1] == tokenizer.sep_token_id
    
    def test_encode_unknown_chars(self, tokenizer):
        """测试未知字符编码为 [UNK]"""
        text = "hello123"  # 数字不在词表中
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        # 数字应该被编码为 UNK
        assert token_ids[-3:] == [tokenizer.unk_token_id] * 3
    
    def test_encode_max_length_truncation(self, tokenizer):
        """测试 max_length 截断"""
        text = "hello world"
        token_ids = tokenizer.encode(text, add_special_tokens=True, max_length=5)
        assert len(token_ids) == 5
    
    def test_encode_padding(self, tokenizer):
        """测试 padding 填充"""
        text = "hi"
        token_ids = tokenizer.encode(
            text, add_special_tokens=True, max_length=10, padding=True
        )
        assert len(token_ids) == 10
        # 最后应该是 PAD token
        assert token_ids[-1] == tokenizer.pad_token_id
    
    def test_encode_empty_string(self, tokenizer):
        """测试空字符串编码"""
        token_ids = tokenizer.encode("", add_special_tokens=False)
        assert token_ids == []
        
        token_ids_with_special = tokenizer.encode("", add_special_tokens=True)
        assert token_ids_with_special == [tokenizer.cls_token_id, tokenizer.sep_token_id]
    
    # ==================== decode 测试 ====================
    
    def test_decode_basic(self, tokenizer):
        """测试基本解码功能"""
        text = "hello"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_decode_skip_special_tokens(self, tokenizer):
        """测试跳过特殊 token 的解码"""
        text = "hi"
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_decode_keep_special_tokens(self, tokenizer):
        """测试保留特殊 token 的解码"""
        token_ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        assert "[CLS]" in decoded
        assert "[SEP]" in decoded
    
    def test_decode_with_padding(self, tokenizer):
        """测试带 padding 的解码"""
        text = "hi"
        token_ids = tokenizer.encode(
            text, add_special_tokens=True, max_length=10, padding=True
        )
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_decode_empty_list(self, tokenizer):
        """测试空列表解码"""
        decoded = tokenizer.decode([], skip_special_tokens=True)
        assert decoded == ""
    
    # ==================== Round-trip 测试 ====================
    
    def test_roundtrip_basic(self, tokenizer):
        """测试基本 round-trip 一致性"""
        text = "hello world"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_roundtrip_with_special_tokens(self, tokenizer):
        """测试带特殊 token 的 round-trip 一致性"""
        text = "hello world"
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_roundtrip_empty_string(self, tokenizer):
        """测试空字符串的 round-trip"""
        text = ""
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    # ==================== 工厂方法测试 ====================
    
    def test_from_chars(self):
        """测试从字符集创建 tokenizer"""
        tokenizer = SimpleTokenizer.from_chars("abc")
        assert tokenizer.vocab_size == 7 + 3  # 特殊 token + abc
        
        text = "abc"
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    def test_from_text(self):
        """测试从文本创建 tokenizer"""
        text = "hello world"
        tokenizer = SimpleTokenizer.from_text(text)
        
        # 词表应该包含文本中的所有唯一字符
        unique_chars = set(text)
        assert tokenizer.vocab_size == 7 + len(unique_chars)
        
        # Round-trip 应该成功
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text
    
    # ==================== add_tokens 测试 ====================
    
    def test_add_tokens(self, tokenizer):
        """测试添加新 token"""
        original_size = tokenizer.vocab_size
        num_added = tokenizer.add_tokens(["新", "字", "符"])
        assert num_added == 3
        assert tokenizer.vocab_size == original_size + 3
    
    def test_add_existing_tokens(self, tokenizer):
        """测试添加已存在的 token"""
        original_size = tokenizer.vocab_size
        num_added = tokenizer.add_tokens(["a", "b", "c"])  # 已存在
        assert num_added == 0
        assert tokenizer.vocab_size == original_size
    
    def test_add_mixed_tokens(self, tokenizer):
        """测试添加混合 token（部分已存在）"""
        original_size = tokenizer.vocab_size
        num_added = tokenizer.add_tokens(["a", "新", "b", "字"])
        assert num_added == 2  # 只有 "新" 和 "字" 是新的
        assert tokenizer.vocab_size == original_size + 2


# ==================== 属性测试 (Property-Based Tests) ====================

from hypothesis import given, strategies as st, settings


class TestTokenizerProperties:
    """
    Tokenizer 属性测试
    
    使用 hypothesis 进行属性测试，验证 round-trip 一致性。
    **Validates: Requirements 12.4**
    """
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_roundtrip_property_without_special_tokens(self, text):
        """
        属性测试：decode(encode(text)) == text（不添加特殊 token）
        
        验证对于任意合法文本输入（仅包含词表中的字符），
        decode(encode(text)) 应产生与原始文本等价的结果。
        
        **Validates: Requirements 12.4**
        """
        vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        tokenizer = SimpleTokenizer(vocab)
        
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip failed: '{text}' -> {token_ids} -> '{decoded}'"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=100))
    @settings(max_examples=100)
    def test_roundtrip_property_with_special_tokens(self, text):
        """
        属性测试：decode(encode(text, add_special_tokens=True)) == text
        
        验证即使添加了特殊 token（[CLS] 和 [SEP]），
        在 decode 时跳过特殊 token 后仍能恢复原始文本。
        
        **Validates: Requirements 12.4**
        """
        vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        tokenizer = SimpleTokenizer(vocab)
        
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip with special tokens failed: '{text}' -> {token_ids} -> '{decoded}'"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=50))
    @settings(max_examples=100)
    def test_roundtrip_property_with_padding(self, text):
        """
        属性测试：带 padding 的 round-trip 一致性
        
        验证即使添加了 padding，decode 后仍能恢复原始文本。
        
        **Validates: Requirements 12.4**
        """
        vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}
        tokenizer = SimpleTokenizer(vocab)
        
        # 使用足够大的 max_length 确保不会截断
        max_length = len(text) + 20
        token_ids = tokenizer.encode(
            text, add_special_tokens=True, max_length=max_length, padding=True
        )
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip with padding failed: '{text}' -> {token_ids} -> '{decoded}'"
    
    @given(st.text(alphabet="你好世界中文测试", min_size=0, max_size=50))
    @settings(max_examples=50)
    def test_roundtrip_property_chinese_chars(self, text):
        """
        属性测试：中文字符的 round-trip 一致性
        
        验证对于中文字符，round-trip 属性同样成立。
        
        **Validates: Requirements 12.4**
        """
        vocab = {char: i for i, char in enumerate("你好世界中文测试")}
        tokenizer = SimpleTokenizer(vocab)
        
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip for Chinese failed: '{text}' -> {token_ids} -> '{decoded}'"
    
    @given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", min_size=0, max_size=50))
    @settings(max_examples=50)
    def test_roundtrip_property_mixed_vocab(self, text):
        """
        属性测试：混合字符（字母+数字+空格）的 round-trip 一致性
        
        **Validates: Requirements 12.4**
        """
        vocab = {char: i for i, char in enumerate("abcdefghijklmnopqrstuvwxyz0123456789 ")}
        tokenizer = SimpleTokenizer(vocab)
        
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip for mixed chars failed: '{text}' -> {token_ids} -> '{decoded}'"
