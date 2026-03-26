"""
Perceiver Resampler module unit tests.

Tests for PerceiverResamplerLayer and PerceiverResampler functionality.

Requirements: 8.2, 8.3
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.multimodal.perceiver import (
    PerceiverResamplerLayer,
    PerceiverResampler
)


class TestPerceiverResamplerLayer:
    """PerceiverResamplerLayer unit tests."""
    
    def test_layer_output_shape(self):
        """Test PerceiverResamplerLayer output shape is correct."""
        batch_size = 2
        num_latents = 64
        num_patches = 196
        d_model = 768
        
        layer = PerceiverResamplerLayer(d_model=d_model, num_heads=8, d_ff=3072)
        
        latents = torch.randn(batch_size, num_latents, d_model)
        visual_features = torch.randn(batch_size, num_patches, d_model)
        
        output = layer(latents, visual_features)
        
        assert output.shape == (batch_size, num_latents, d_model)
    
    def test_layer_cross_attention(self):
        """Test PerceiverResamplerLayer cross-attention (Requirement 8.3)."""
        batch_size = 2
        num_latents = 64
        num_patches = 196
        d_model = 768
        
        layer = PerceiverResamplerLayer(d_model=d_model, num_heads=8, d_ff=3072)
        
        latents = torch.randn(batch_size, num_latents, d_model)
        visual_features_1 = torch.randn(batch_size, num_patches, d_model)
        visual_features_2 = torch.randn(batch_size, num_patches, d_model)
        
        output_1 = layer(latents, visual_features_1)
        output_2 = layer(latents, visual_features_2)
        
        # Different visual features should produce different outputs
        assert not torch.allclose(output_1, output_2, atol=1e-5)
    
    def test_layer_self_attention(self):
        """Test PerceiverResamplerLayer self-attention."""
        batch_size = 2
        num_latents = 64
        d_model = 768
        
        layer = PerceiverResamplerLayer(d_model=d_model, num_heads=8, d_ff=3072)
        
        latents = torch.randn(batch_size, num_latents, d_model)
        # Use minimal visual features to focus on self-attention effect
        visual_features = torch.zeros(batch_size, 1, d_model)
        
        output = layer(latents, visual_features)
        
        # Output should be different from input due to self-attention
        assert not torch.allclose(output, latents, atol=1e-5)

    def test_layer_with_visual_attention_mask(self):
        """Test PerceiverResamplerLayer supports visual attention mask."""
        batch_size = 2
        num_latents = 64
        num_patches = 196
        d_model = 768
        
        layer = PerceiverResamplerLayer(d_model=d_model, num_heads=8, d_ff=3072)
        
        latents = torch.randn(batch_size, num_latents, d_model)
        visual_features = torch.randn(batch_size, num_patches, d_model)
        
        # Create mask that masks out half of the patches
        visual_attention_mask = torch.zeros(batch_size, num_patches)
        visual_attention_mask[:, num_patches // 2:] = 1
        
        output_with_mask = layer(
            latents, 
            visual_features, 
            visual_attention_mask=visual_attention_mask
        )
        output_without_mask = layer(latents, visual_features)
        
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
    
    def test_layer_different_configs(self):
        """Test PerceiverResamplerLayer with different configurations."""
        batch_size = 2
        num_latents = 32
        num_patches = 49
        
        configs = [
            {"d_model": 256, "num_heads": 4, "d_ff": 1024},
            {"d_model": 512, "num_heads": 8, "d_ff": 2048},
            {"d_model": 768, "num_heads": 8, "d_ff": 3072},
        ]
        
        for config in configs:
            layer = PerceiverResamplerLayer(**config)
            
            latents = torch.randn(batch_size, num_latents, config["d_model"])
            visual_features = torch.randn(batch_size, num_patches, config["d_model"])
            
            output = layer(latents, visual_features)
            
            assert output.shape == (batch_size, num_latents, config["d_model"])
    
    def test_layer_variable_num_patches(self):
        """Test PerceiverResamplerLayer handles variable number of patches."""
        batch_size = 2
        num_latents = 64
        d_model = 768
        
        layer = PerceiverResamplerLayer(d_model=d_model, num_heads=8, d_ff=3072)
        
        latents = torch.randn(batch_size, num_latents, d_model)
        
        # Test with different numbers of patches
        for num_patches in [49, 196, 576]:
            visual_features = torch.randn(batch_size, num_patches, d_model)
            output = layer(latents, visual_features)
            
            assert output.shape == (batch_size, num_latents, d_model)


class TestPerceiverResampler:
    """PerceiverResampler unit tests."""
    
    def test_resampler_output_shape(self):
        """Test PerceiverResampler output shape (Requirement 8.2)."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        visual_features = torch.randn(batch_size, num_patches, d_model)
        
        output = resampler(visual_features)
        
        assert output.shape == (batch_size, num_latents, d_model)
    
    def test_resampler_learnable_latents(self):
        """Test PerceiverResampler has learnable latent vectors (Requirement 8.3)."""
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        assert hasattr(resampler, 'latents')
        assert isinstance(resampler.latents, nn.Parameter)
        assert resampler.latents.shape == (1, num_latents, d_model)
        assert resampler.latents.requires_grad
    
    def test_resampler_default_64_latents(self):
        """Test PerceiverResampler default 64 latent vectors (Requirement 8.2)."""
        d_model = 768
        
        resampler = PerceiverResampler(d_model=d_model, num_heads=8, num_layers=6)
        
        assert resampler.num_latents == 64
        assert resampler.latents.shape == (1, 64, d_model)

    def test_resampler_fixed_output_tokens(self):
        """Test PerceiverResampler outputs fixed number of tokens (Requirement 8.2)."""
        batch_size = 4
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        # Test with different numbers of patches (variable input)
        for num_patches in [49, 196, 576]:
            visual_features = torch.randn(batch_size, num_patches, d_model)
            output = resampler(visual_features)
            
            # Output should always have fixed num_latents tokens
            assert output.shape == (batch_size, num_latents, d_model)
    
    def test_resampler_cross_attention_aggregation(self):
        """Test PerceiverResampler aggregates visual features via cross-attention (Requirement 8.3)."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        visual_features_1 = torch.randn(batch_size, num_patches, d_model)
        visual_features_2 = torch.randn(batch_size, num_patches, d_model)
        
        output_1 = resampler(visual_features_1)
        output_2 = resampler(visual_features_2)
        
        # Different visual features should produce different outputs
        assert not torch.allclose(output_1, output_2, atol=1e-5)
    
    def test_resampler_with_visual_attention_mask(self):
        """Test PerceiverResampler supports visual attention mask."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        visual_features = torch.randn(batch_size, num_patches, d_model)
        visual_attention_mask = torch.zeros(batch_size, num_patches)
        visual_attention_mask[:, num_patches // 2:] = 1
        
        output_with_mask = resampler(visual_features, visual_attention_mask=visual_attention_mask)
        output_without_mask = resampler(visual_features)
        
        assert output_with_mask.shape == (batch_size, num_latents, d_model)
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
    
    def test_resampler_get_latents(self):
        """Test PerceiverResampler get_latents method."""
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        latents = resampler.get_latents()
        
        assert latents.shape == (1, num_latents, d_model)
        assert torch.equal(latents, resampler.latents)
    
    def test_resampler_multiple_layers(self):
        """Test PerceiverResampler contains multiple layers."""
        d_model = 768
        num_layers = 6
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=64,
            num_heads=8,
            num_layers=num_layers
        )
        
        assert len(resampler.layers) == num_layers
        for layer in resampler.layers:
            assert isinstance(layer, PerceiverResamplerLayer)

    def test_resampler_final_norm(self):
        """Test PerceiverResampler has final LayerNorm."""
        d_model = 768
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=64,
            num_heads=8,
            num_layers=6
        )
        
        assert resampler.final_norm is not None
        assert isinstance(resampler.final_norm, nn.LayerNorm)
    
    def test_resampler_different_batch_sizes(self):
        """Test PerceiverResampler supports different batch sizes."""
        d_model = 768
        num_patches = 196
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        for batch_size in [1, 2, 4, 8]:
            visual_features = torch.randn(batch_size, num_patches, d_model)
            output = resampler(visual_features)
            
            assert output.shape == (batch_size, num_latents, d_model)
    
    def test_resampler_gradient_flow(self):
        """Test PerceiverResampler gradient flow is correct."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        visual_features = torch.randn(batch_size, num_patches, d_model, requires_grad=True)
        
        output = resampler(visual_features)
        loss = output.sum()
        loss.backward()
        
        # Check gradient flows to input
        assert visual_features.grad is not None
        assert not torch.all(visual_features.grad == 0)
        
        # Check gradient flows to learnable latents
        assert resampler.latents.grad is not None
        assert not torch.all(resampler.latents.grad == 0)
    
    def test_resampler_eval_mode(self):
        """Test PerceiverResampler behavior in eval mode."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6,
            dropout_rate=0.5
        )
        
        visual_features = torch.randn(batch_size, num_patches, d_model)
        
        # In train mode, outputs should vary due to dropout
        resampler.train()
        outputs_train = [resampler(visual_features) for _ in range(5)]
        
        # In eval mode, outputs should be deterministic
        resampler.eval()
        outputs_eval = [resampler(visual_features) for _ in range(5)]
        
        for i in range(1, 5):
            assert torch.allclose(outputs_eval[0], outputs_eval[i], atol=1e-6)
    
    def test_resampler_default_d_ff(self):
        """Test PerceiverResampler default d_ff is 4 * d_model."""
        d_model = 768
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=64,
            num_heads=8,
            num_layers=6
        )
        
        assert resampler.d_ff == 4 * d_model
    
    def test_resampler_custom_d_ff(self):
        """Test PerceiverResampler with custom d_ff."""
        d_model = 768
        custom_d_ff = 2048
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=64,
            num_heads=8,
            num_layers=6,
            d_ff=custom_d_ff
        )
        
        assert resampler.d_ff == custom_d_ff


class TestPerceiverResamplerIntegration:
    """PerceiverResampler integration tests."""
    
    def test_resampler_with_vit_output(self):
        """Test PerceiverResampler integration with ViT output."""
        batch_size = 2
        num_patches = 196  # 14x14 patches from 224x224 image with patch_size=16
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        # Simulate ViT output (without CLS token)
        vit_output = torch.randn(batch_size, num_patches, d_model)
        
        output = resampler(vit_output)
        
        assert output.shape == (batch_size, num_latents, d_model)
    
    def test_resampler_output_for_llm_projection(self):
        """Test PerceiverResampler output can be used for LLM projection."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_latents = 64
        llm_dim = 4096
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        # Simple linear projection to LLM dimension
        visual_projection = nn.Linear(d_model, llm_dim)
        
        visual_features = torch.randn(batch_size, num_patches, d_model)
        
        resampler_output = resampler(visual_features)
        projected_output = visual_projection(resampler_output)
        
        assert projected_output.shape == (batch_size, num_latents, llm_dim)
    
    def test_resampler_compression_ratio(self):
        """Test PerceiverResampler compresses variable tokens to fixed tokens."""
        batch_size = 2
        d_model = 768
        num_latents = 64
        
        resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
            num_heads=8,
            num_layers=6
        )
        
        # Test compression from different input sizes
        test_cases = [
            (49, 64),    # 7x7 patches -> 64 latents (expansion)
            (196, 64),   # 14x14 patches -> 64 latents (compression ~3x)
            (576, 64),   # 24x24 patches -> 64 latents (compression ~9x)
        ]
        
        for num_patches, expected_latents in test_cases:
            visual_features = torch.randn(batch_size, num_patches, d_model)
            output = resampler(visual_features)
            
            assert output.shape == (batch_size, expected_latents, d_model)
