"""
核心 Transformer 组件模块

包含：
- Multi-Head Attention
- Position Encoding（正弦余弦和可学习）
- Feed Forward Network
- Encoder/Decoder Layer
"""

from .attention import scaled_dot_product_attention, MultiHeadAttention
from .position import SinusoidalPositionEncoding, LearnablePositionEmbedding
from .feedforward import FeedForwardNetwork
from .layers import EncoderLayer, DecoderLayer

__all__ = [
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    'SinusoidalPositionEncoding',
    'LearnablePositionEmbedding',
    'FeedForwardNetwork',
    'EncoderLayer',
    'DecoderLayer',
]
