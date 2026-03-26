"""
Gated Cross Attention module unit tests.

Tests for GatedCrossAttentionLayer functionality.

Requirements: 8.5, 8.6
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.multimodal.gated_cross_attention import (
    GatedCrossAttentionLayer
)


class TestGatedCrossAttentionLayer:
    """GatedCrossAttentionLayer unit tests."""
    
    def test_output_shape(self):
        """Test GatedCrossAttentionLayer output shape is correct."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output = layer(hidden_states, visual_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_gate_parameter_initialization(self):
        """Test gate parameter alpha is initialized to 0 (Requirement 8.6)."""
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        # alpha should be initialized to 0
        assert hasattr(layer, 'alpha')
        assert isinstance(layer.alpha, nn.Parameter)
        assert torch.allclose(layer.alpha, torch.zeros(1))
        assert layer.alpha.requires_grad
    
    def test_gate_value_is_zero_at_init(self):
        """Test tanh(alpha) = 0 at initialization (Requirement 8.6)."""
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        gate_value = layer.get_gate_value()
        
        # tanh(0) = 0
        assert torch.allclose(gate_value, torch.zeros(1), atol=1e-6)
    
    def test_output_equals_input_at_init(self):
        """Test output equals input when gate is 0 (Requirement 8.6)."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8, dropout_rate=0.0)
        layer.eval()  # Disable dropout
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output = layer(hidden_states, visual_features)
        
        # At initialization, tanh(alpha) = 0, so output should equal input
        assert torch.allclose(output, hidden_states, atol=1e-5)
    
    def test_cross_attention_effect(self):
        """Test cross-attention affects output when gate is non-zero (Requirement 8.5)."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8, dropout_rate=0.0)
        layer.eval()
        
        # Set alpha to a non-zero value
        with torch.no_grad():
            layer.alpha.fill_(1.0)  # tanh(1.0) ≈ 0.76
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output = layer(hidden_states, visual_features)
        
        # Output should be different from input when gate is non-zero
        assert not torch.allclose(output, hidden_states, atol=1e-5)
    
    def test_different_visual_features_produce_different_outputs(self):
        """Test different visual features produce different outputs (Requirement 8.5)."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8, dropout_rate=0.0)
        layer.eval()
        
        # Set alpha to a non-zero value
        with torch.no_grad():
            layer.alpha.fill_(1.0)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features_1 = torch.randn(batch_size, num_visual_tokens, d_model)
        visual_features_2 = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output_1 = layer(hidden_states, visual_features_1)
        output_2 = layer(hidden_states, visual_features_2)
        
        # Different visual features should produce different outputs
        assert not torch.allclose(output_1, output_2, atol=1e-5)
    
    def test_visual_attention_mask(self):
        """Test visual attention mask is applied correctly."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8, dropout_rate=0.0)
        layer.eval()
        
        # Set alpha to a non-zero value
        with torch.no_grad():
            layer.alpha.fill_(1.0)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        # Create mask that masks out half of the visual tokens
        visual_attention_mask = torch.zeros(batch_size, num_visual_tokens)
        visual_attention_mask[:, num_visual_tokens // 2:] = 1
        
        output_with_mask = layer(
            hidden_states, 
            visual_features, 
            visual_attention_mask=visual_attention_mask
        )
        output_without_mask = layer(hidden_states, visual_features)
        
        assert output_with_mask.shape == (batch_size, seq_len, d_model)
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
    
    def test_different_configs(self):
        """Test GatedCrossAttentionLayer with different configurations."""
        batch_size = 2
        seq_len = 64
        num_visual_tokens = 32
        
        configs = [
            {"d_model": 256, "num_heads": 4},
            {"d_model": 512, "num_heads": 8},
            {"d_model": 768, "num_heads": 12},
            {"d_model": 1024, "num_heads": 16},
        ]
        
        for config in configs:
            layer = GatedCrossAttentionLayer(**config)
            
            hidden_states = torch.randn(batch_size, seq_len, config["d_model"])
            visual_features = torch.randn(batch_size, num_visual_tokens, config["d_model"])
            
            output = layer(hidden_states, visual_features)
            
            assert output.shape == (batch_size, seq_len, config["d_model"])
    
    def test_variable_sequence_lengths(self):
        """Test GatedCrossAttentionLayer handles variable sequence lengths."""
        batch_size = 2
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        # Test with different sequence lengths
        for seq_len in [32, 64, 128, 256]:
            hidden_states = torch.randn(batch_size, seq_len, d_model)
            output = layer(hidden_states, visual_features)
            
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_variable_visual_tokens(self):
        """Test GatedCrossAttentionLayer handles variable number of visual tokens."""
        batch_size = 2
        seq_len = 128
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # Test with different numbers of visual tokens
        for num_visual_tokens in [32, 64, 128, 196]:
            visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
            output = layer(hidden_states, visual_features)
            
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_gradient_flow(self):
        """Test gradient flow through GatedCrossAttentionLayer."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        # Set alpha to non-zero so gradient flows through cross-attention
        with torch.no_grad():
            layer.alpha.fill_(1.0)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model, requires_grad=True)
        
        output = layer(hidden_states, visual_features)
        loss = output.sum()
        loss.backward()
        
        # Check gradient flows to inputs
        assert hidden_states.grad is not None
        assert not torch.all(hidden_states.grad == 0)
        
        assert visual_features.grad is not None
        assert not torch.all(visual_features.grad == 0)
        
        # Check gradient flows to gate parameter
        assert layer.alpha.grad is not None
    
    def test_gate_parameter_learnable(self):
        """Test gate parameter alpha is learnable."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        # Initial alpha value
        initial_alpha = layer.alpha.clone()
        
        # Forward and backward pass
        output = layer(hidden_states, visual_features)
        loss = output.sum()
        loss.backward()
        
        # Simulate optimizer step
        with torch.no_grad():
            layer.alpha -= 0.1 * layer.alpha.grad
        
        # Alpha should have changed
        assert not torch.allclose(layer.alpha, initial_alpha)
    
    def test_gate_value_range(self):
        """Test gate value tanh(alpha) is in range [-1, 1]."""
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        # Test with various alpha values
        test_alphas = [-10.0, -1.0, 0.0, 1.0, 10.0]
        
        for alpha_val in test_alphas:
            with torch.no_grad():
                layer.alpha.fill_(alpha_val)
            
            gate_value = layer.get_gate_value()
            
            assert gate_value >= -1.0
            assert gate_value <= 1.0
    
    def test_eval_mode_deterministic(self):
        """Test GatedCrossAttentionLayer is deterministic in eval mode."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8, dropout_rate=0.5)
        layer.eval()
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        # Multiple forward passes should produce same output in eval mode
        outputs = [layer(hidden_states, visual_features) for _ in range(5)]
        
        for i in range(1, 5):
            assert torch.allclose(outputs[0], outputs[i], atol=1e-6)
    
    def test_different_batch_sizes(self):
        """Test GatedCrossAttentionLayer supports different batch sizes."""
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        for batch_size in [1, 2, 4, 8]:
            hidden_states = torch.randn(batch_size, seq_len, d_model)
            visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
            
            output = layer(hidden_states, visual_features)
            
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_has_layer_norm(self):
        """Test GatedCrossAttentionLayer has LayerNorm for cross-attention."""
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        assert hasattr(layer, 'cross_attn_norm')
        assert isinstance(layer.cross_attn_norm, nn.LayerNorm)
    
    def test_has_cross_attention_components(self):
        """Test GatedCrossAttentionLayer has all cross-attention components."""
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        # Check Q, K, V projections
        assert hasattr(layer, 'cross_attn_q')
        assert hasattr(layer, 'cross_attn_k')
        assert hasattr(layer, 'cross_attn_v')
        assert hasattr(layer, 'cross_attn_out')
        
        # Check they are Linear layers
        assert isinstance(layer.cross_attn_q, nn.Linear)
        assert isinstance(layer.cross_attn_k, nn.Linear)
        assert isinstance(layer.cross_attn_v, nn.Linear)
        assert isinstance(layer.cross_attn_out, nn.Linear)


class TestGatedCrossAttentionIntegration:
    """GatedCrossAttentionLayer integration tests."""
    
    def test_integration_with_perceiver_output(self):
        """Test GatedCrossAttentionLayer with Perceiver Resampler output."""
        batch_size = 2
        seq_len = 256
        num_latents = 64  # Perceiver Resampler output
        d_model = 768
        
        layer = GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
        
        # Set alpha to non-zero for meaningful test
        with torch.no_grad():
            layer.alpha.fill_(1.0)
        
        # Simulate LLM hidden states
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # Simulate Perceiver Resampler output
        visual_features = torch.randn(batch_size, num_latents, d_model)
        
        output = layer(hidden_states, visual_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_insertion_in_decoder_layer(self):
        """Test GatedCrossAttentionLayer can be inserted in decoder layer (Requirement 8.5)."""
        batch_size = 2
        seq_len = 256
        num_visual_tokens = 64
        d_model = 768
        
        # Simulate a decoder layer with gated cross attention
        class MockDecoderLayerWithGatedCrossAttention(nn.Module):
            def __init__(self, d_model, num_heads):
                super().__init__()
                self.self_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
                self.gated_cross_attn = GatedCrossAttentionLayer(d_model, num_heads)
                self.ffn = nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model)
                )
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
            
            def forward(self, hidden_states, visual_features):
                # Self-attention
                residual = hidden_states
                hidden_states = self.norm1(hidden_states)
                hidden_states, _ = self.self_attn(hidden_states, hidden_states, hidden_states)
                hidden_states = residual + hidden_states
                
                # Gated cross-attention
                hidden_states = self.gated_cross_attn(hidden_states, visual_features)
                
                # FFN
                residual = hidden_states
                hidden_states = self.norm2(hidden_states)
                hidden_states = self.ffn(hidden_states)
                hidden_states = residual + hidden_states
                
                return hidden_states
        
        decoder_layer = MockDecoderLayerWithGatedCrossAttention(d_model, num_heads=8)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        output = decoder_layer(hidden_states, visual_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_multiple_gated_cross_attention_layers(self):
        """Test multiple GatedCrossAttentionLayer instances work independently."""
        batch_size = 2
        seq_len = 128
        num_visual_tokens = 64
        d_model = 768
        num_layers = 4
        
        layers = nn.ModuleList([
            GatedCrossAttentionLayer(d_model=d_model, num_heads=8)
            for _ in range(num_layers)
        ])
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        visual_features = torch.randn(batch_size, num_visual_tokens, d_model)
        
        # Each layer should have its own alpha parameter
        for i, layer in enumerate(layers):
            with torch.no_grad():
                layer.alpha.fill_(float(i) * 0.5)
        
        # Process through all layers
        x = hidden_states
        for layer in layers:
            x = layer(x, visual_features)
        
        assert x.shape == (batch_size, seq_len, d_model)
        
        # Verify each layer has different alpha
        for i, layer in enumerate(layers):
            expected_alpha = float(i) * 0.5
            assert torch.allclose(layer.alpha, torch.tensor([expected_alpha]))
