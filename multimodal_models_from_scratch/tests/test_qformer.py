"""
Q-Former module unit tests.

Tests for QFormerLayer and QFormer functionality.

Requirements: 6.2, 6.3, 6.4, 6.5
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.multimodal.qformer import QFormerLayer, QFormer


class TestQFormerLayer:
    """QFormerLayer unit tests."""
    
    def test_qformer_layer_output_shape(self):
        """Test QFormerLayer output shape is correct."""
        batch_size = 2
        num_queries = 32
        num_patches = 196
        d_model = 768
        
        layer = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072)
        
        query_embeds = torch.randn(batch_size, num_queries, d_model)
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        output = layer(query_embeds, encoder_hidden_states)
        
        assert output.shape == (batch_size, num_queries, d_model)
    
    def test_qformer_layer_self_attention(self):
        """Test QFormerLayer self-attention (Requirement 6.3)."""
        batch_size = 2
        num_queries = 32
        d_model = 768
        
        layer = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072)
        
        query_embeds = torch.randn(batch_size, num_queries, d_model)
        encoder_hidden_states = torch.zeros(batch_size, 1, d_model)
        
        output = layer(query_embeds, encoder_hidden_states)
        
        assert not torch.allclose(output, query_embeds, atol=1e-5)
    
    def test_qformer_layer_cross_attention(self):
        """Test QFormerLayer cross-attention (Requirement 6.4)."""
        batch_size = 2
        num_queries = 32
        num_patches = 196
        d_model = 768
        
        layer = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072)
        
        query_embeds = torch.randn(batch_size, num_queries, d_model)
        encoder_hidden_states_1 = torch.randn(batch_size, num_patches, d_model)
        encoder_hidden_states_2 = torch.randn(batch_size, num_patches, d_model)
        
        output_1 = layer(query_embeds, encoder_hidden_states_1)
        output_2 = layer(query_embeds, encoder_hidden_states_2)
        
        assert not torch.allclose(output_1, output_2, atol=1e-5)
    
    def test_qformer_layer_with_attention_mask(self):
        """Test QFormerLayer supports attention mask."""
        batch_size = 2
        num_queries = 32
        num_patches = 196
        d_model = 768
        
        layer = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072)
        
        query_embeds = torch.randn(batch_size, num_queries, d_model)
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        encoder_attention_mask = torch.zeros(batch_size, num_patches)
        encoder_attention_mask[:, num_patches // 2:] = 1
        
        output_with_mask = layer(
            query_embeds, 
            encoder_hidden_states, 
            encoder_attention_mask=encoder_attention_mask
        )
        output_without_mask = layer(query_embeds, encoder_hidden_states)
        
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
    
    def test_qformer_layer_pre_norm_vs_post_norm(self):
        """Test Pre-Norm and Post-Norm architectures."""
        batch_size = 2
        num_queries = 32
        num_patches = 196
        d_model = 768
        
        layer_pre_norm = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072, use_pre_norm=True)
        layer_post_norm = QFormerLayer(d_model=d_model, num_heads=12, d_ff=3072, use_pre_norm=False)
        
        query_embeds = torch.randn(batch_size, num_queries, d_model)
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        output_pre = layer_pre_norm(query_embeds, encoder_hidden_states)
        output_post = layer_post_norm(query_embeds, encoder_hidden_states)
        
        assert output_pre.shape == output_post.shape
        assert not torch.allclose(output_pre, output_post, atol=1e-5)
    
    def test_qformer_layer_different_configs(self):
        """Test QFormerLayer with different configurations."""
        batch_size = 2
        num_queries = 16
        num_patches = 49
        
        configs = [
            {"d_model": 256, "num_heads": 4, "d_ff": 1024},
            {"d_model": 512, "num_heads": 8, "d_ff": 2048},
            {"d_model": 768, "num_heads": 12, "d_ff": 3072},
        ]
        
        for config in configs:
            layer = QFormerLayer(**config)
            
            query_embeds = torch.randn(batch_size, num_queries, config["d_model"])
            encoder_hidden_states = torch.randn(batch_size, num_patches, config["d_model"])
            
            output = layer(query_embeds, encoder_hidden_states)
            
            assert output.shape == (batch_size, num_queries, config["d_model"])


class TestQFormer:
    """QFormer unit tests."""
    
    def test_qformer_output_shape(self):
        """Test QFormer output shape (Requirement 6.5)."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        output = qformer(encoder_hidden_states)
        
        assert output.shape == (batch_size, num_query_tokens, d_model)
    
    def test_qformer_learnable_query_tokens(self):
        """Test QFormer has learnable query tokens (Requirement 6.2)."""
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        assert hasattr(qformer, 'query_tokens')
        assert isinstance(qformer.query_tokens, nn.Parameter)
        assert qformer.query_tokens.shape == (1, num_query_tokens, d_model)
        assert qformer.query_tokens.requires_grad
    
    def test_qformer_default_query_tokens(self):
        """Test QFormer default 32 query tokens (Requirement 6.2)."""
        d_model = 768
        
        qformer = QFormer(d_model=d_model, num_heads=12, num_layers=6, d_ff=3072)
        
        assert qformer.num_query_tokens == 32
        assert qformer.query_tokens.shape == (1, 32, d_model)
    
    def test_qformer_fixed_output_tokens(self):
        """Test QFormer outputs fixed number of visual tokens (Requirement 6.5)."""
        batch_size = 4
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        for num_patches in [49, 196, 576]:
            encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
            output = qformer(encoder_hidden_states)
            
            assert output.shape == (batch_size, num_query_tokens, d_model)
    
    def test_qformer_with_encoder_attention_mask(self):
        """Test QFormer supports encoder attention mask."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        encoder_attention_mask = torch.zeros(batch_size, num_patches)
        encoder_attention_mask[:, num_patches // 2:] = 1
        
        output_with_mask = qformer(encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
        output_without_mask = qformer(encoder_hidden_states)
        
        assert output_with_mask.shape == (batch_size, num_query_tokens, d_model)
        assert not torch.allclose(output_with_mask, output_without_mask, atol=1e-5)
    
    def test_qformer_with_external_query_embeds(self):
        """Test QFormer supports external query embeddings."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        external_query_embeds = torch.randn(batch_size, 16, d_model)
        
        output = qformer(encoder_hidden_states, query_embeds=external_query_embeds)
        
        assert output.shape == (batch_size, 16, d_model)
    
    def test_qformer_get_query_tokens(self):
        """Test QFormer get_query_tokens method."""
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        query_tokens = qformer.get_query_tokens()
        
        assert query_tokens.shape == (1, num_query_tokens, d_model)
        assert torch.equal(query_tokens, qformer.query_tokens)
    
    def test_qformer_multiple_layers(self):
        """Test QFormer contains multiple QFormerLayers."""
        d_model = 768
        num_layers = 6
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=num_layers,
            d_ff=3072,
            num_query_tokens=32
        )
        
        assert len(qformer.layers) == num_layers
        for layer in qformer.layers:
            assert isinstance(layer, QFormerLayer)
    
    def test_qformer_final_norm(self):
        """Test QFormer final LayerNorm (Pre-Norm architecture)."""
        d_model = 768
        
        qformer_pre_norm = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=32,
            use_pre_norm=True
        )
        
        qformer_post_norm = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=32,
            use_pre_norm=False
        )
        
        assert qformer_pre_norm.final_norm is not None
        assert isinstance(qformer_pre_norm.final_norm, nn.LayerNorm)
        
        assert qformer_post_norm.final_norm is None
    
    def test_qformer_different_batch_sizes(self):
        """Test QFormer supports different batch sizes."""
        d_model = 768
        num_patches = 196
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        for batch_size in [1, 2, 4, 8]:
            encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
            output = qformer(encoder_hidden_states)
            
            assert output.shape == (batch_size, num_query_tokens, d_model)
    
    def test_qformer_gradient_flow(self):
        """Test QFormer gradient flow is correct."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model, requires_grad=True)
        
        output = qformer(encoder_hidden_states)
        loss = output.sum()
        loss.backward()
        
        assert encoder_hidden_states.grad is not None
        assert not torch.all(encoder_hidden_states.grad == 0)
        
        assert qformer.query_tokens.grad is not None
        assert not torch.all(qformer.query_tokens.grad == 0)
    
    def test_qformer_eval_mode(self):
        """Test QFormer behavior in eval mode."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens,
            dropout_rate=0.5
        )
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        qformer.train()
        outputs_train = [qformer(encoder_hidden_states) for _ in range(5)]
        
        qformer.eval()
        outputs_eval = [qformer(encoder_hidden_states) for _ in range(5)]
        
        for i in range(1, 5):
            assert torch.allclose(outputs_eval[0], outputs_eval[i], atol=1e-6)


class TestQFormerIntegration:
    """QFormer integration tests."""
    
    def test_qformer_with_vit_output(self):
        """Test QFormer integration with ViT output."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        vit_output = torch.randn(batch_size, num_patches, d_model)
        
        output = qformer(vit_output)
        
        assert output.shape == (batch_size, num_query_tokens, d_model)
    
    def test_qformer_output_for_llm_projection(self):
        """Test QFormer output can be used for LLM projection."""
        batch_size = 2
        num_patches = 196
        d_model = 768
        num_query_tokens = 32
        llm_dim = 4096
        
        qformer = QFormer(
            d_model=d_model,
            num_heads=12,
            num_layers=6,
            d_ff=3072,
            num_query_tokens=num_query_tokens
        )
        
        visual_projection = nn.Linear(d_model, llm_dim)
        
        encoder_hidden_states = torch.randn(batch_size, num_patches, d_model)
        
        qformer_output = qformer(encoder_hidden_states)
        projected_output = visual_projection(qformer_output)
        
        assert projected_output.shape == (batch_size, num_query_tokens, llm_dim)
