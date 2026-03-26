"""
多模态模型配置类

包含视觉编码器、LLM、多模态模型的配置 dataclass。
复用 bert-gpt-from-scratch 的 TransformerConfig。
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

# 复用 bert-gpt-from-scratch 的基础配置
from bert_gpt_from_scratch.config import TransformerConfig


@dataclass
class VisionConfig:
    """视觉编码器配置
    
    包含 ViT 等视觉模型的核心超参数配置。
    
    Attributes:
        image_size: 输入图像尺寸
        patch_size: patch 大小
        in_channels: 输入通道数
        d_model: 模型维度（隐藏层大小）
        num_heads: 注意力头数量
        num_layers: Transformer 层数
        d_ff: 前馈网络中间层维度
        dropout_rate: Dropout 比率
        num_classes: 图像分类类别数
    """
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    dropout_rate: float = 0.1
    num_classes: int = 1000


@dataclass
class LLaMAConfig:
    """LLaMA 模型配置
    
    包含 LLaMA 大语言模型的核心超参数配置。
    
    Attributes:
        vocab_size: 词表大小
        d_model: 模型维度（隐藏层大小）
        num_heads: Query 注意力头数量
        num_kv_heads: Key/Value 注意力头数量（GQA）
        num_layers: Decoder 层数
        d_ff: 前馈网络中间层维度
        max_seq_len: 最大序列长度
        dropout_rate: Dropout 比率
        rope_theta: RoPE 基础频率
        tie_weights: 是否绑定 Token Embedding 和 LM Head 权重
    """
    vocab_size: int = 32000
    d_model: int = 4096
    num_heads: int = 32
    num_kv_heads: int = 8
    num_layers: int = 32
    d_ff: int = 11008
    max_seq_len: int = 4096
    dropout_rate: float = 0.0
    rope_theta: float = 10000.0
    tie_weights: bool = False


@dataclass
class QwenConfig(LLaMAConfig):
    """Qwen 模型配置
    
    继承自 LLaMAConfig，添加 Qwen 特有的配置参数。
    
    Attributes:
        use_sliding_window: 是否使用滑动窗口注意力
        sliding_window_size: 滑动窗口大小
        rope_scaling: NTK-aware 插值配置
    """
    use_sliding_window: bool = False
    sliding_window_size: int = 4096
    rope_scaling: Optional[Dict[str, Any]] = None


@dataclass
class CLIPConfig:
    """CLIP 模型配置
    
    包含 CLIP 图文对比学习模型的配置参数。
    
    Attributes:
        vision_config: 视觉编码器配置
        text_config: 文本编码器配置（复用 TransformerConfig）
        projection_dim: 投影维度
        temperature: 对比学习温度参数
    """
    vision_config: Optional[VisionConfig] = None
    text_config: Optional[TransformerConfig] = None
    projection_dim: int = 512
    temperature: float = 0.07
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.text_config is None:
            self.text_config = TransformerConfig()


@dataclass
class BLIPConfig:
    """BLIP 模型配置
    
    包含 BLIP 图文理解与生成模型的配置参数。
    
    Attributes:
        vision_config: 视觉编码器配置
        text_config: 文本编码器/解码器配置（复用 TransformerConfig）
        projection_dim: 投影维度
    """
    vision_config: Optional[VisionConfig] = None
    text_config: Optional[TransformerConfig] = None
    projection_dim: int = 256
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.text_config is None:
            self.text_config = TransformerConfig()


@dataclass
class BLIP2Config:
    """BLIP-2 模型配置
    
    包含 BLIP-2 Q-Former 架构模型的配置参数。
    
    Attributes:
        vision_config: 视觉编码器配置
        qformer_config: Q-Former 配置（复用 TransformerConfig）
        llm_config: LLM 配置
        num_query_tokens: 查询 token 数量
        projection_dim: 投影维度
    """
    vision_config: Optional[VisionConfig] = None
    qformer_config: Optional[TransformerConfig] = None
    llm_config: Optional[LLaMAConfig] = None
    num_query_tokens: int = 32
    projection_dim: int = 768
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.qformer_config is None:
            self.qformer_config = TransformerConfig()
        if self.llm_config is None:
            self.llm_config = LLaMAConfig()


@dataclass
class FlamingoConfig:
    """Flamingo 模型配置
    
    包含 Flamingo 视觉-语言交错输入模型的配置参数。
    
    Attributes:
        vision_config: 视觉编码器配置
        llm_config: LLM 配置
        num_latents: Perceiver Resampler 的 latent 向量数量
        cross_attention_freq: 每隔多少层插入交叉注意力
    """
    vision_config: Optional[VisionConfig] = None
    llm_config: Optional[LLaMAConfig] = None
    num_latents: int = 64
    cross_attention_freq: int = 4
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.llm_config is None:
            self.llm_config = LLaMAConfig()


@dataclass
class LLaVAConfig:
    """LLaVA 模型配置
    
    包含 LLaVA 视觉指令微调模型的配置参数。
    
    Attributes:
        vision_config: 视觉编码器配置
        llm_config: LLM 配置
        projection_type: 投影类型 ('linear' 或 'mlp')
        freeze_vision: 是否冻结视觉编码器
        freeze_llm: 是否冻结 LLM
    """
    vision_config: Optional[VisionConfig] = None
    llm_config: Optional[LLaMAConfig] = None
    projection_type: str = 'mlp'
    freeze_vision: bool = True
    freeze_llm: bool = False
    
    def __post_init__(self):
        if self.vision_config is None:
            self.vision_config = VisionConfig()
        if self.llm_config is None:
            self.llm_config = LLaMAConfig()


@dataclass
class DETRConfig:
    """DETR 模型配置
    
    包含 DETR 目标检测模型的配置参数。
    
    Attributes:
        num_classes: 目标类别数（COCO 为 91）
        num_queries: Object Query 数量
        d_model: 模型维度
        num_heads: 注意力头数量
        num_encoder_layers: Encoder 层数
        num_decoder_layers: Decoder 层数
        d_ff: 前馈网络中间层维度
        dropout_rate: Dropout 比率
    """
    num_classes: int = 91
    num_queries: int = 100
    d_model: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout_rate: float = 0.1
