"""
Multimodal model components.

Modules:
- clip: CLIP model
- blip: BLIP model
- blip2: BLIP-2 model
- qformer: Q-Former module
- flamingo: Flamingo model
- perceiver: Perceiver Resampler
- gated_cross_attention: Gated Cross Attention
- llava: LLaVA model
- visual_projection: Visual Projection layer
"""

from multimodal_models_from_scratch.multimodal.visual_projection import VisualProjection
from multimodal_models_from_scratch.multimodal.qformer import QFormerLayer, QFormer
from multimodal_models_from_scratch.multimodal.perceiver import PerceiverResamplerLayer, PerceiverResampler
from multimodal_models_from_scratch.multimodal.gated_cross_attention import GatedCrossAttentionLayer
from multimodal_models_from_scratch.multimodal.clip import CLIPModel, TextEncoder, contrastive_loss
from multimodal_models_from_scratch.multimodal.blip import (
    BLIPModel,
    TextEncoderWithCrossAttention,
    TextDecoderWithCrossAttention,
    CrossAttentionLayer,
    itc_loss,
    itm_loss,
)
from multimodal_models_from_scratch.multimodal.flamingo import FlamingoModel, FlamingoDecoderLayer
from multimodal_models_from_scratch.multimodal.llava import LLaVAModel, DEFAULT_IMAGE_TOKEN_ID

__all__ = [
    "VisualProjection",
    "QFormerLayer",
    "QFormer",
    "PerceiverResamplerLayer",
    "PerceiverResampler",
    "GatedCrossAttentionLayer",
    "CLIPModel",
    "TextEncoder",
    "contrastive_loss",
    "BLIPModel",
    "TextEncoderWithCrossAttention",
    "TextDecoderWithCrossAttention",
    "CrossAttentionLayer",
    "itc_loss",
    "itm_loss",
    "FlamingoModel",
    "FlamingoDecoderLayer",
    "LLaVAModel",
    "DEFAULT_IMAGE_TOKEN_ID",
]
