"""
DETR (Detection Transformer) Model.

This module implements the DETR object detection model that uses a Transformer
encoder-decoder architecture for end-to-end object detection.

Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DETRConfig
from ..vision.backbone import ResNetBackbone, PositionEmbeddingSine


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer for DETR.
    
    Uses pre-norm architecture with self-attention and feed-forward network.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        self.activation = nn.ReLU()
    
    def forward(
        self,
        src: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch, seq_len, d_model)
            pos_embed: Position embeddings (batch, seq_len, d_model)
            src_key_padding_mask: Padding mask (batch, seq_len), True for padded positions
            
        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        # Add position embedding to query and key
        q = k = src if pos_embed is None else src + pos_embed
        
        # Self-attention with residual
        src2, _ = self.self_attn(
            query=q,
            key=k,
            value=src,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        
        return src



class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer for DETR.
    
    Uses pre-norm architecture with self-attention, cross-attention, and feed-forward network.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)
        
        self.activation = nn.ReLU()
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Target sequence / object queries (batch, num_queries, d_model)
            memory: Encoder output (batch, seq_len, d_model)
            query_pos: Query position embeddings (batch, num_queries, d_model)
            memory_pos: Memory position embeddings (batch, seq_len, d_model)
            memory_key_padding_mask: Memory padding mask (batch, seq_len)
            
        Returns:
            Output tensor (batch, num_queries, d_model)
        """
        # Self-attention with query position embedding
        q = k = tgt if query_pos is None else tgt + query_pos
        
        tgt2, _ = self.self_attn(
            query=q,
            key=k,
            value=tgt
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with position embeddings
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if memory_pos is None else memory + memory_pos
        
        tgt2, _ = self.cross_attn(
            query=q,
            key=k,
            value=memory,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feed-forward with residual
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt



class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for DETR.
    
    Stack of TransformerEncoderLayer modules.
    
    Args:
        num_layers: Number of encoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        src: torch.Tensor,
        pos_embed: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch, seq_len, d_model)
            pos_embed: Position embeddings (batch, seq_len, d_model)
            src_key_padding_mask: Padding mask (batch, seq_len)
            
        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, pos_embed, src_key_padding_mask)
        
        return self.norm(output)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder for DETR.
    
    Stack of TransformerDecoderLayer modules.
    
    Args:
        num_layers: Number of decoder layers
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward network dimension
        dropout_rate: Dropout rate
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 256,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        memory_pos: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Target sequence / object queries (batch, num_queries, d_model)
            memory: Encoder output (batch, seq_len, d_model)
            query_pos: Query position embeddings (batch, num_queries, d_model)
            memory_pos: Memory position embeddings (batch, seq_len, d_model)
            memory_key_padding_mask: Memory padding mask (batch, seq_len)
            
        Returns:
            Decoder output (batch, num_queries, d_model)
        """
        output = tgt
        
        for layer in self.layers:
            output = layer(
                output, memory, query_pos, memory_pos, memory_key_padding_mask
            )
        
        return self.norm(output)



class MLP(nn.Module):
    """
    Multi-layer perceptron for classification and bounding box prediction heads.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension
        num_layers: Number of layers
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int
    ):
        super().__init__()
        
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (..., input_dim)
            
        Returns:
            Output tensor (..., output_dim)
        """
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """
    DETR (Detection Transformer) Model.
    
    End-to-end object detection model using Transformer encoder-decoder architecture.
    
    Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9
    
    Args:
        config: DETRConfig containing model hyperparameters
        backbone_name: Name of the ResNet backbone ('resnet18', 'resnet34', 'resnet50')
        frozen_bn: Whether to use frozen BatchNorm in backbone
    
    Attributes:
        backbone: ResNet backbone for feature extraction
        input_proj: Linear projection from backbone channels to d_model
        encoder: Transformer encoder
        decoder: Transformer decoder
        query_embed: Learnable object queries
        class_head: Classification head (d_model -> num_classes + 1)
        bbox_head: Bounding box regression head (d_model -> 4)
    """
    
    def __init__(
        self,
        config: DETRConfig,
        backbone_name: str = 'resnet50',
        frozen_bn: bool = True
    ):
        super().__init__()
        
        self.config = config
        self.num_classes = config.num_classes
        self.num_queries = config.num_queries
        self.d_model = config.d_model
        
        # CNN Backbone with position embedding (Requirement 3.1)
        self.backbone = ResNetBackbone(
            name=backbone_name,
            frozen_bn=frozen_bn,
            d_model=config.d_model
        )
        
        # Project backbone features to d_model dimension
        self.input_proj = nn.Conv2d(
            self.backbone.out_channels,
            config.d_model,
            kernel_size=1
        )
        
        # Transformer Encoder (Requirement 3.2)
        self.encoder = TransformerEncoder(
            num_layers=config.num_encoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout_rate=config.dropout_rate
        )
        
        # Transformer Decoder (Requirement 3.3)
        self.decoder = TransformerDecoder(
            num_layers=config.num_decoder_layers,
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            dropout_rate=config.dropout_rate
        )
        
        # Learnable Object Queries (Requirement 3.4)
        # Shape: (num_queries, d_model)
        self.query_embed = nn.Embedding(config.num_queries, config.d_model)
        
        # Classification Head (Requirement 3.5)
        # Output: num_classes + 1 (including background class)
        self.class_head = nn.Linear(config.d_model, config.num_classes + 1)
        
        # Bounding Box Regression Head (Requirement 3.6)
        # Output: 4 (cx, cy, w, h in normalized coordinates)
        self.bbox_head = MLP(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            output_dim=4,
            num_layers=3
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize query embeddings
        nn.init.normal_(self.query_embed.weight)
        
        # Initialize classification head
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0)
        
        # Initialize input projection
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0)

    
    def forward(
        self,
        pixel_values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of DETR model.
        
        Args:
            pixel_values: Input images (batch, 3, H, W)
            mask: Optional padding mask (batch, H, W), True for padded positions
            
        Returns:
            Dictionary containing:
            - 'pred_logits': Classification logits (batch, num_queries, num_classes + 1)
            - 'pred_boxes': Predicted bounding boxes (batch, num_queries, 4) in (cx, cy, w, h) format
        """
        batch_size = pixel_values.shape[0]
        
        # Extract features from backbone (Requirement 3.1)
        backbone_output = self.backbone(pixel_values, mask)
        features = backbone_output['features']  # (batch, backbone_channels, H/32, W/32)
        pos_embed = backbone_output['pos_embed']  # (batch, d_model, H/32, W/32)
        feature_mask = backbone_output.get('mask')  # (batch, H/32, W/32) or None
        
        # Project features to d_model dimension
        src = self.input_proj(features)  # (batch, d_model, H', W')
        
        # Flatten spatial dimensions for transformer
        _, _, h, w = src.shape
        src = src.flatten(2).permute(0, 2, 1)  # (batch, H'*W', d_model)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # (batch, H'*W', d_model)
        
        # Flatten mask if present
        if feature_mask is not None:
            feature_mask = feature_mask.flatten(1)  # (batch, H'*W')
        
        # Encode features with transformer encoder (Requirement 3.2)
        memory = self.encoder(
            src=src,
            pos_embed=pos_embed,
            src_key_padding_mask=feature_mask
        )  # (batch, H'*W', d_model)
        
        # Get object queries (Requirement 3.4)
        query_embed = self.query_embed.weight  # (num_queries, d_model)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, d_model)
        
        # Initialize target with zeros
        tgt = torch.zeros_like(query_embed)
        
        # Decode with transformer decoder (Requirement 3.3)
        hs = self.decoder(
            tgt=tgt,
            memory=memory,
            query_pos=query_embed,
            memory_pos=pos_embed,
            memory_key_padding_mask=feature_mask
        )  # (batch, num_queries, d_model)
        
        # Classification head (Requirement 3.5)
        pred_logits = self.class_head(hs)  # (batch, num_queries, num_classes + 1)
        
        # Bounding box regression head (Requirement 3.6)
        pred_boxes = self.bbox_head(hs).sigmoid()  # (batch, num_queries, 4)
        
        return {
            'pred_logits': pred_logits,
            'pred_boxes': pred_boxes
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True


def build_detr(
    config: Optional[DETRConfig] = None,
    backbone_name: str = 'resnet50',
    frozen_bn: bool = True
) -> DETR:
    """
    Build DETR model with default or custom configuration.
    
    Args:
        config: Optional DETRConfig. If None, uses default configuration.
        backbone_name: Name of the ResNet backbone
        frozen_bn: Whether to use frozen BatchNorm in backbone
        
    Returns:
        DETR model instance
    """
    if config is None:
        config = DETRConfig()
    
    return DETR(
        config=config,
        backbone_name=backbone_name,
        frozen_bn=frozen_bn
    )
