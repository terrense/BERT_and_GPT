"""
推理引擎单元测试
"""

import pytest
import torch
import tempfile
import os

from bert_gpt_from_scratch.config import BERTConfig, GPTConfig
from bert_gpt_from_scratch.models.bert import BERTModel
from bert_gpt_from_scratch.models.gpt import GPTModel
from bert_gpt_from_scratch.tokenizer import SimpleTokenizer
from bert_gpt_from_scratch.inference.engine import InferenceEngine


class TestInferenceEngine:
    """推理引擎测试"""
    
    @pytest.fixture
    def bert_config(self):
        """创建小型 BERT 配置"""
        return BERTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0
        )
    
    @pytest.fixture
    def gpt_config(self):
        """创建小型 GPT 配置"""
        return GPTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0,
            tie_weights=True
        )
    
    @pytest.fixture
    def tokenizer(self):
        """创建 tokenizer"""
        return SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
    
    @pytest.fixture
    def bert_checkpoint(self, bert_config):
        """创建 BERT 检查点"""
        model = BERTModel(bert_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bert.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': bert_config,
            }, path)
            yield path
    
    @pytest.fixture
    def gpt_checkpoint(self, gpt_config):
        """创建 GPT 检查点"""
        model = GPTModel(gpt_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gpt.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': gpt_config,
            }, path)
            yield path
    
    def test_load_bert_model(self, bert_config, bert_checkpoint, tokenizer):
        """测试加载 BERT 模型"""
        engine = InferenceEngine(device='cpu')
        engine.load_model('bert', bert_checkpoint, bert_config, tokenizer)
        
        assert engine.model is not None
        assert engine.model_type == 'bert'
    
    def test_load_gpt_model(self, gpt_config, gpt_checkpoint, tokenizer):
        """测试加载 GPT 模型"""
        engine = InferenceEngine(device='cpu')
        engine.load_model('gpt', gpt_checkpoint, gpt_config, tokenizer)
        
        assert engine.model is not None
        assert engine.model_type == 'gpt'


class TestBERTInference:
    """BERT 推理测试"""
    
    @pytest.fixture
    def engine(self):
        """创建加载了 BERT 的推理引擎"""
        config = BERTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0
        )
        tokenizer = SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
        
        model = BERTModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bert.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
            }, path)
            
            engine = InferenceEngine(device='cpu')
            engine.load_model('bert', path, config, tokenizer)
            yield engine
    
    def test_bert_fill_mask_output_format(self, engine):
        """测试 BERT MLM 填空输出格式"""
        # 在文本中添加 [MASK]
        text = "hello [MASK] world"
        # 手动将 [MASK] 编码到输入中
        input_ids = engine.tokenizer.encode("hello ", add_special_tokens=False)
        input_ids.append(engine.tokenizer.mask_token_id)
        input_ids.extend(engine.tokenizer.encode(" world", add_special_tokens=False))
        input_ids = [engine.tokenizer.cls_token_id] + input_ids + [engine.tokenizer.sep_token_id]
        
        # 直接测试模型输出
        input_tensor = torch.tensor([input_ids])
        segment_ids = torch.zeros_like(input_tensor)
        
        with torch.no_grad():
            outputs = engine.model(input_tensor, segment_ids)
        
        # 验证输出形状
        assert outputs['mlm_logits'].shape[-1] == engine.model.config.vocab_size


class TestGPTInference:
    """GPT 推理测试"""
    
    @pytest.fixture
    def engine(self):
        """创建加载了 GPT 的推理引擎"""
        config = GPTConfig(
            vocab_size=100,
            d_model=32,
            num_heads=2,
            num_layers=2,
            d_ff=64,
            max_seq_len=32,
            dropout_rate=0.0,
            tie_weights=True
        )
        tokenizer = SimpleTokenizer.from_chars("abcdefghijklmnopqrstuvwxyz ")
        
        model = GPTModel(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "gpt.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
            }, path)
            
            engine = InferenceEngine(device='cpu')
            engine.load_model('gpt', path, config, tokenizer)
            yield engine
    
    def test_gpt_generate_greedy(self, engine):
        """测试 GPT greedy 生成"""
        prompt = "hello"
        generated = engine.gpt_generate(
            prompt,
            max_gen_len=10,
            decoding_strategy='greedy'
        )
        
        # 生成的文本应该以 prompt 开头
        assert generated.startswith(prompt)
    
    def test_gpt_generate_top_k(self, engine):
        """测试 GPT top-k 采样"""
        prompt = "hello"
        generated = engine.gpt_generate(
            prompt,
            max_gen_len=10,
            decoding_strategy='top_k',
            top_k=10
        )
        
        assert generated.startswith(prompt)
    
    def test_gpt_generate_top_p(self, engine):
        """测试 GPT top-p 采样"""
        prompt = "hello"
        generated = engine.gpt_generate(
            prompt,
            max_gen_len=10,
            decoding_strategy='top_p',
            top_p=0.9
        )
        
        assert generated.startswith(prompt)
    
    def test_gpt_generate_max_length(self, engine):
        """测试 GPT 生成最大长度限制"""
        prompt = "hi"
        max_gen_len = 5
        
        generated = engine.gpt_generate(
            prompt,
            max_gen_len=max_gen_len,
            decoding_strategy='greedy'
        )
        
        # 生成的 token 数不应超过 prompt + max_gen_len
        generated_ids = engine.tokenizer.encode(generated, add_special_tokens=False)
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        
        assert len(generated_ids) <= len(prompt_ids) + max_gen_len
    
    def test_gpt_generate_temperature(self, engine):
        """测试温度参数影响"""
        prompt = "hello"
        
        # 低温度应该更确定性
        generated_low_temp = engine.gpt_generate(
            prompt,
            max_gen_len=5,
            temperature=0.1,
            decoding_strategy='greedy'
        )
        
        # 高温度应该更随机（但 greedy 仍然是确定性的）
        generated_high_temp = engine.gpt_generate(
            prompt,
            max_gen_len=5,
            temperature=2.0,
            decoding_strategy='greedy'
        )
        
        # 两者都应该以 prompt 开头
        assert generated_low_temp.startswith(prompt)
        assert generated_high_temp.startswith(prompt)
