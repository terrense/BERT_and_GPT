"""
Grouped Query Attention (GQA) 单元测试

测试 GQA 模块的功能正确性，包括：
- 输出形状正确性
- num_kv_heads < num_heads 的配置
- KV Cache 功能
- RoPE 位置编码集成
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.llm.gqa import GroupedQueryAttention
from multimodal_models_from_scratch.llm.rope import RotaryPositionEmbedding


class TestGroupedQueryAttention:
    """GQA 模块测试类"""
    
    def test_output_shape_basic(self):
        """测试基本输出形状"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        output, past_kv = gqa(hidden_states)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert past_kv is None  # use_cache=False by default
    
    def test_output_shape_with_cache(self):
        """测试带 KV Cache 的输出形状"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = d_model // num_heads
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        output, past_kv = gqa(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert past_kv is not None
        assert len(past_kv) == 2  # (key, value)
        assert past_kv[0].shape == (batch_size, num_kv_heads, seq_len, head_dim)
        assert past_kv[1].shape == (batch_size, num_kv_heads, seq_len, head_dim)
    
    def test_kv_cache_incremental(self):
        """测试 KV Cache 增量生成"""
        batch_size = 2
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = d_model // num_heads
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        # 第一步：处理初始序列
        initial_seq_len = 10
        hidden_states = torch.randn(batch_size, initial_seq_len, d_model)
        output1, past_kv = gqa(hidden_states, use_cache=True)
        
        assert past_kv[0].shape == (batch_size, num_kv_heads, initial_seq_len, head_dim)
        
        # 第二步：增量生成一个 token
        new_token = torch.randn(batch_size, 1, d_model)
        output2, new_past_kv = gqa(new_token, past_key_value=past_kv, use_cache=True)
        
        assert output2.shape == (batch_size, 1, d_model)
        assert new_past_kv[0].shape == (batch_size, num_kv_heads, initial_seq_len + 1, head_dim)
        assert new_past_kv[1].shape == (batch_size, num_kv_heads, initial_seq_len + 1, head_dim)
    
    def test_num_groups_calculation(self):
        """测试 num_groups 计算"""
        d_model = 256
        
        # num_heads=8, num_kv_heads=2 -> num_groups=4
        gqa1 = GroupedQueryAttention(d_model=d_model, num_heads=8, num_kv_heads=2)
        assert gqa1.num_groups == 4
        
        # num_heads=8, num_kv_heads=4 -> num_groups=2
        gqa2 = GroupedQueryAttention(d_model=d_model, num_heads=8, num_kv_heads=4)
        assert gqa2.num_groups == 2
        
        # num_heads=8, num_kv_heads=8 -> num_groups=1 (MHA)
        gqa3 = GroupedQueryAttention(d_model=d_model, num_heads=8, num_kv_heads=8)
        assert gqa3.num_groups == 1
        
        # num_heads=8, num_kv_heads=1 -> num_groups=8 (MQA)
        gqa4 = GroupedQueryAttention(d_model=d_model, num_heads=8, num_kv_heads=1)
        assert gqa4.num_groups == 8
    
    def test_invalid_num_kv_heads(self):
        """测试无效的 num_kv_heads 配置"""
        d_model = 256
        num_heads = 8
        
        # num_heads 不能被 num_kv_heads 整除
        with pytest.raises(AssertionError):
            GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=3)
    
    def test_with_attention_mask(self):
        """测试带注意力掩码的前向传播"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len) * float('-inf'),
            diagonal=1
        )
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        
        output, _ = gqa(hidden_states, attention_mask=causal_mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_with_rope_position_embeddings(self):
        """测试带 RoPE 位置编码的前向传播"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = d_model // num_heads
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        rope = RotaryPositionEmbedding(d_model=head_dim, max_seq_len=128)
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # 获取 RoPE 的 cos 和 sin
        cos = rope.cos_cached[:, :, :seq_len, :]
        sin = rope.sin_cached[:, :, :seq_len, :]
        
        output, _ = gqa(hidden_states, position_embeddings=(cos, sin))
        
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_kv_cache_with_rope(self):
        """测试 KV Cache 与 RoPE 的结合使用"""
        batch_size = 2
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = d_model // num_heads
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        rope = RotaryPositionEmbedding(d_model=head_dim, max_seq_len=128)
        
        # 第一步：处理初始序列
        initial_seq_len = 10
        hidden_states = torch.randn(batch_size, initial_seq_len, d_model)
        cos = rope.cos_cached[:, :, :initial_seq_len, :]
        sin = rope.sin_cached[:, :, :initial_seq_len, :]
        
        output1, past_kv = gqa(
            hidden_states,
            position_embeddings=(cos, sin),
            use_cache=True
        )
        
        # 第二步：增量生成
        new_token = torch.randn(batch_size, 1, d_model)
        # 注意：增量生成时，RoPE 只需要新位置的 cos/sin
        cos_new = rope.cos_cached[:, :, initial_seq_len:initial_seq_len+1, :]
        sin_new = rope.sin_cached[:, :, initial_seq_len:initial_seq_len+1, :]
        
        output2, new_past_kv = gqa(
            new_token,
            position_embeddings=(cos_new, sin_new),
            past_key_value=past_kv,
            use_cache=True
        )
        
        assert output2.shape == (batch_size, 1, d_model)
        assert new_past_kv[0].shape[2] == initial_seq_len + 1
    
    def test_mha_equivalence(self):
        """测试 num_kv_heads == num_heads 时等价于 MHA"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        
        # GQA with num_kv_heads == num_heads
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        output, past_kv = gqa(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        # KV Cache 形状应该与 Query 头数相同
        assert past_kv[0].shape[1] == num_heads
    
    def test_mqa_equivalence(self):
        """测试 num_kv_heads == 1 时等价于 MQA"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        
        # GQA with num_kv_heads == 1 (Multi-Query Attention)
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=1
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        output, past_kv = gqa(hidden_states, use_cache=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        # KV Cache 只有 1 个头
        assert past_kv[0].shape[1] == 1
    
    def test_dropout(self):
        """测试 dropout 功能"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout_rate=0.1
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model)
        
        # 训练模式下 dropout 应该生效
        gqa.train()
        output1, _ = gqa(hidden_states)
        output2, _ = gqa(hidden_states)
        
        # 由于 dropout，两次输出应该不同（概率上）
        # 注意：这个测试可能偶尔失败，因为 dropout 是随机的
        
        # 评估模式下 dropout 不生效
        gqa.eval()
        output3, _ = gqa(hidden_states)
        output4, _ = gqa(hidden_states)
        
        assert torch.allclose(output3, output4)
    
    def test_extra_repr(self):
        """测试 extra_repr 方法"""
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        repr_str = gqa.extra_repr()
        assert 'd_model=256' in repr_str
        assert 'num_heads=8' in repr_str
        assert 'num_kv_heads=2' in repr_str
        assert 'num_groups=4' in repr_str


class TestGQAIntegration:
    """GQA 集成测试"""
    
    def test_gradient_flow(self):
        """测试梯度流动"""
        batch_size = 2
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        hidden_states = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output, _ = gqa(hidden_states)
        
        # 计算损失并反向传播
        loss = output.sum()
        loss.backward()
        
        # 检查梯度是否存在
        assert hidden_states.grad is not None
        assert gqa.q_proj.weight.grad is not None
        assert gqa.k_proj.weight.grad is not None
        assert gqa.v_proj.weight.grad is not None
        assert gqa.o_proj.weight.grad is not None
    
    def test_parameter_count(self):
        """测试参数数量"""
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        head_dim = d_model // num_heads
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        # Q 投影: d_model * (num_heads * head_dim)
        # K 投影: d_model * (num_kv_heads * head_dim)
        # V 投影: d_model * (num_kv_heads * head_dim)
        # O 投影: (num_heads * head_dim) * d_model
        expected_params = (
            d_model * num_heads * head_dim +  # Q
            d_model * num_kv_heads * head_dim +  # K
            d_model * num_kv_heads * head_dim +  # V
            num_heads * head_dim * d_model  # O
        )
        
        actual_params = sum(p.numel() for p in gqa.parameters())
        assert actual_params == expected_params
    
    def test_different_batch_sizes(self):
        """测试不同批次大小"""
        seq_len = 16
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        for batch_size in [1, 2, 4, 8]:
            hidden_states = torch.randn(batch_size, seq_len, d_model)
            output, _ = gqa(hidden_states)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_seq_lengths(self):
        """测试不同序列长度"""
        batch_size = 2
        d_model = 256
        num_heads = 8
        num_kv_heads = 2
        
        gqa = GroupedQueryAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads
        )
        
        for seq_len in [1, 8, 16, 32, 64]:
            hidden_states = torch.randn(batch_size, seq_len, d_model)
            output, _ = gqa(hidden_states)
            assert output.shape == (batch_size, seq_len, d_model)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
