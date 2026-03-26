"""
简易 Tokenizer 实现

支持特殊 token：[PAD], [UNK], [CLS], [SEP], [MASK], [BOS], [EOS]
提供 encode 和 decode 方法。

这是一个字符级 Tokenizer，用于演示和测试目的。
"""

from typing import Dict, List, Optional


class SimpleTokenizer:
    """
    简易字符级 Tokenizer
    
    特殊 token:
    - [PAD]: 0
    - [UNK]: 1
    - [CLS]: 2
    - [SEP]: 3
    - [MASK]: 4
    - [BOS]: 5
    - [EOS]: 6
    """
    
    # 特殊 token 定义
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    
    # 特殊 token ID
    PAD_TOKEN_ID = 0
    UNK_TOKEN_ID = 1
    CLS_TOKEN_ID = 2
    SEP_TOKEN_ID = 3
    MASK_TOKEN_ID = 4
    BOS_TOKEN_ID = 5
    EOS_TOKEN_ID = 6
    
    # 特殊 token 数量
    NUM_SPECIAL_TOKENS = 7
    
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        """
        初始化 Tokenizer
        
        Args:
            vocab: 词表字典，映射 token 到 ID。如果为 None，则创建空词表。
                   特殊 token 会自动添加到词表开头。
        """
        # 初始化特殊 token 映射
        self._special_tokens = {
            self.PAD_TOKEN: self.PAD_TOKEN_ID,
            self.UNK_TOKEN: self.UNK_TOKEN_ID,
            self.CLS_TOKEN: self.CLS_TOKEN_ID,
            self.SEP_TOKEN: self.SEP_TOKEN_ID,
            self.MASK_TOKEN: self.MASK_TOKEN_ID,
            self.BOS_TOKEN: self.BOS_TOKEN_ID,
            self.EOS_TOKEN: self.EOS_TOKEN_ID,
        }
        
        # 特殊 token ID 集合（用于快速查找）
        self._special_token_ids = set(self._special_tokens.values())
        
        # 构建完整词表：特殊 token + 用户提供的词表
        self._token_to_id: Dict[str, int] = dict(self._special_tokens)
        
        if vocab is not None:
            # 将用户词表中的 token 添加到词表，ID 从 NUM_SPECIAL_TOKENS 开始
            for token, _ in vocab.items():
                if token not in self._token_to_id:
                    self._token_to_id[token] = len(self._token_to_id)
        
        # 构建反向映射：ID -> token
        self._id_to_token: Dict[int, str] = {
            id_: token for token, id_ in self._token_to_id.items()
        }
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False
    ) -> List[int]:
        """
        将文本编码为 token ID 序列
        
        Args:
            text: 输入文本
            add_special_tokens: 是否添加特殊 token（[CLS] 在开头，[SEP] 在结尾）
            max_length: 最大长度（截断）。如果为 None，则不截断。
            padding: 是否填充到 max_length。需要 max_length 不为 None。
        
        Returns:
            token_ids: token ID 列表
        """
        # 字符级分词：将文本拆分为单个字符
        tokens = list(text)
        
        # 将 token 转换为 ID
        token_ids = []
        for token in tokens:
            if token in self._token_to_id:
                token_ids.append(self._token_to_id[token])
            else:
                token_ids.append(self.UNK_TOKEN_ID)
        
        # 添加特殊 token
        if add_special_tokens:
            token_ids = [self.CLS_TOKEN_ID] + token_ids + [self.SEP_TOKEN_ID]
        
        # 截断到 max_length
        if max_length is not None:
            token_ids = token_ids[:max_length]
        
        # 填充到 max_length
        if padding and max_length is not None:
            padding_length = max_length - len(token_ids)
            if padding_length > 0:
                token_ids = token_ids + [self.PAD_TOKEN_ID] * padding_length
        
        return token_ids
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        将 token ID 序列解码为文本
        
        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            text: 解码后的文本
        """
        tokens = []
        for token_id in token_ids:
            # 跳过特殊 token
            if skip_special_tokens and token_id in self._special_token_ids:
                continue
            
            # 将 ID 转换为 token
            if token_id in self._id_to_token:
                tokens.append(self._id_to_token[token_id])
            else:
                # 未知 ID，使用 [UNK] token
                tokens.append(self.UNK_TOKEN)
        
        # 字符级 tokenizer：直接拼接
        return "".join(tokens)
    
    def add_tokens(self, tokens: List[str]) -> int:
        """
        向词表添加新 token
        
        Args:
            tokens: 要添加的 token 列表
        
        Returns:
            num_added: 实际添加的 token 数量
        """
        num_added = 0
        for token in tokens:
            if token not in self._token_to_id:
                new_id = len(self._token_to_id)
                self._token_to_id[token] = new_id
                self._id_to_token[new_id] = token
                num_added += 1
        return num_added
    
    def get_vocab(self) -> Dict[str, int]:
        """
        获取完整词表
        
        Returns:
            vocab: token 到 ID 的映射字典
        """
        return dict(self._token_to_id)
    
    @property
    def vocab_size(self) -> int:
        """词表大小"""
        return len(self._token_to_id)
    
    @property
    def pad_token_id(self) -> int:
        """[PAD] token ID"""
        return self.PAD_TOKEN_ID
    
    @property
    def unk_token_id(self) -> int:
        """[UNK] token ID"""
        return self.UNK_TOKEN_ID
    
    @property
    def cls_token_id(self) -> int:
        """[CLS] token ID"""
        return self.CLS_TOKEN_ID
    
    @property
    def sep_token_id(self) -> int:
        """[SEP] token ID"""
        return self.SEP_TOKEN_ID
    
    @property
    def mask_token_id(self) -> int:
        """[MASK] token ID"""
        return self.MASK_TOKEN_ID
    
    @property
    def bos_token_id(self) -> int:
        """[BOS] token ID"""
        return self.BOS_TOKEN_ID
    
    @property
    def eos_token_id(self) -> int:
        """[EOS] token ID"""
        return self.EOS_TOKEN_ID
    
    @classmethod
    def from_chars(cls, chars: str) -> "SimpleTokenizer":
        """
        从字符集创建 Tokenizer
        
        Args:
            chars: 包含所有字符的字符串
        
        Returns:
            tokenizer: 新创建的 Tokenizer 实例
        """
        vocab = {char: i for i, char in enumerate(chars)}
        return cls(vocab)
    
    @classmethod
    def from_text(cls, text: str) -> "SimpleTokenizer":
        """
        从文本自动构建词表并创建 Tokenizer
        
        Args:
            text: 用于构建词表的文本
        
        Returns:
            tokenizer: 新创建的 Tokenizer 实例
        """
        # 提取所有唯一字符
        unique_chars = sorted(set(text))
        vocab = {char: i for i, char in enumerate(unique_chars)}
        return cls(vocab)
