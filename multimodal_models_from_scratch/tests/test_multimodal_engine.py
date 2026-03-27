"""
多模态推理引擎单元测试

测试 MultimodalInferenceEngine 的各种功能：
- 模型加载
- 解码策略
- 各模型类型的推理方法
"""

import pytest
import torch
import torch.nn as nn

from multimodal_models_from_scratch.inference.multimodal_engine import (
    MultimodalInferenceEngine,
    SUPPORTED_MODEL_TYPES
)
from multimodal_models_from_scratch.config import (
    VisionConfig,
    CLIPConfig,
    BLIPConfig,
    BLIP2Config,
    FlamingoConfig,
    LLaVAConfig,
    DETRConfig,
    LLaMAConfig,
)
from bert_gpt_from_scratch.config import TransformerConfig


class SimpleTokenizer:
    """简单的测试用分词器"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
    
    def encode(self, text: str, add_special_tokens: bool = True) -> list:
        """简单编码：将每个字符映射到一个 token ID"""
        ids = [ord(c) % (self.vocab_size - 3) + 3 for c in text[:50]]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        return ids
    
    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """简单解码"""
        if skip_special_tokens:
            ids = [i for i in ids if i not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        return ''.join(chr((i - 3) % 128 + 32) for i in ids if i >= 3)
    
    def __call__(self, texts):
        """批量编码"""
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = [self.encode(t) for t in texts]
        max_len = max(len(e) for e in encoded)
        
        input_ids = []
        attention_mask = []
        for e in encoded:
            padded = e + [self.pad_token_id] * (max_len - len(e))
            mask = [1] * len(e) + [0] * (max_len - len(e))
            input_ids.append(padded)
            attention_mask.append(mask)
        
        return {
            'input_ids': torch.tensor(input_ids),
            'attention_mask': torch.tensor(attention_mask)
        }


class TestMultimodalInferenceEngine:
    """MultimodalInferenceEngine 测试类"""
    
    @pytest.fixture
    def engine(self):
        """创建推理引擎实例"""
        return MultimodalInferenceEngine(device='cpu')
    
    @pytest.fixture
    def tokenizer(self):
        """创建测试用分词器"""
        return SimpleTokenizer(vocab_size=1000)
    
    @pytest.fixture
    def small_vision_config(self):
        """创建小型视觉配置"""
        return VisionConfig(
            image_size=32,
            patch_size=8,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            num_classes=10
        )
    
    @pytest.fixture
    def small_text_config(self):
        """创建小型文本配置"""
        return TransformerConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=64
        )
    
    @pytest.fixture
    def small_llama_config(self):
        """创建小型 LLaMA 配置"""
        return LLaMAConfig(
            vocab_size=1000,
            d_model=64,
            num_heads=2,
            num_kv_heads=2,
            num_layers=2,
            d_ff=128,
            max_seq_len=64
        )
    
    def test_engine_initialization(self, engine):
        """测试引擎初始化"""
        assert engine.device is not None
        assert engine.model is None
        assert engine.model_type is None
    
    def test_supported_model_types(self):
        """测试支持的模型类型"""
        expected_types = ['vit', 'clip', 'blip', 'blip2', 'flamingo', 'llava', 'detr']
        assert SUPPORTED_MODEL_TYPES == expected_types
    
    def test_load_vit_model(self, engine, small_vision_config):
        """测试加载 ViT 模型"""
        engine.load_model('vit', None, small_vision_config)
        
        assert engine.model is not None
        assert engine.model_type == 'vit'
        assert engine.image_processor is not None
    
    def test_load_clip_model(self, engine, small_vision_config, small_text_config):
        """测试加载 CLIP 模型"""
        config = CLIPConfig(
            vision_config=small_vision_config,
            text_config=small_text_config,
            projection_dim=32
        )
        
        engine.load_model('clip', None, config)
        
        assert engine.model is not None
        assert engine.model_type == 'clip'
    
    def test_load_blip_model(self, engine, small_vision_config, small_text_config):
        """测试加载 BLIP 模型"""
        config = BLIPConfig(
            vision_config=small_vision_config,
            text_config=small_text_config,
            projection_dim=32
        )
        
        engine.load_model('blip', None, config)
        
        assert engine.model is not None
        assert engine.model_type == 'blip'
    
    def test_load_detr_model(self, engine):
        """测试加载 DETR 模型"""
        config = DETRConfig(
            num_classes=10,
            num_queries=10,
            d_model=64,
            num_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128
        )
        
        engine.load_model('detr', None, config, backbone_name='resnet18')
        
        assert engine.model is not None
        assert engine.model_type == 'detr'
    
    def test_load_unsupported_model_type(self, engine):
        """测试加载不支持的模型类型"""
        with pytest.raises(ValueError, match="Unsupported model type"):
            engine.load_model('unknown', None, None)
    
    def test_vit_classify(self, engine, small_vision_config):
        """测试 ViT 分类推理"""
        engine.load_model('vit', None, small_vision_config)
        
        # 创建测试图像
        image = torch.randn(3, 32, 32)
        
        results = engine.vit_classify(image, top_k=3)
        
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], int) and isinstance(r[1], float) for r in results)
        # 概率应该是递减的
        assert results[0][1] >= results[1][1] >= results[2][1]
    
    def test_vit_classify_wrong_model_type(self, engine, small_vision_config, small_text_config):
        """测试在非 ViT 模型上调用 vit_classify"""
        config = CLIPConfig(
            vision_config=small_vision_config,
            text_config=small_text_config
        )
        engine.load_model('clip', None, config)
        
        with pytest.raises(ValueError, match="only supported for ViT models"):
            engine.vit_classify(torch.randn(3, 32, 32))
    
    def test_detr_detect(self, engine):
        """测试 DETR 目标检测推理"""
        config = DETRConfig(
            num_classes=10,
            num_queries=10,
            d_model=64,
            num_heads=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            d_ff=128
        )
        
        engine.load_model('detr', None, config, backbone_name='resnet18')
        
        # 创建测试图像
        image = torch.randn(3, 224, 224)
        
        results = engine.detr_detect(image, threshold=0.0)  # 低阈值以确保有结果
        
        assert isinstance(results, list)
        if len(results) > 0:
            assert all('label' in r and 'score' in r and 'box' in r for r in results)
            assert all(len(r['box']) == 4 for r in results)
    
    def test_clip_zero_shot_classify(self, engine, small_vision_config, small_text_config, tokenizer):
        """测试 CLIP 零样本分类"""
        config = CLIPConfig(
            vision_config=small_vision_config,
            text_config=small_text_config,
            projection_dim=32
        )
        
        engine.load_model('clip', None, config, tokenizer=tokenizer)
        
        # 创建测试图像
        image = torch.randn(3, 32, 32)
        text_labels = ["a cat", "a dog", "a bird"]
        
        predicted_label, probabilities = engine.clip_zero_shot_classify(image, text_labels)
        
        assert predicted_label in text_labels
        assert probabilities.shape == (3,)
        assert torch.allclose(probabilities.sum(), torch.tensor(1.0), atol=1e-5)
    
    def test_clip_image_text_similarity(self, engine, small_vision_config, small_text_config, tokenizer):
        """测试 CLIP 图文相似度计算"""
        config = CLIPConfig(
            vision_config=small_vision_config,
            text_config=small_text_config,
            projection_dim=32
        )
        
        engine.load_model('clip', None, config, tokenizer=tokenizer)
        
        # 创建测试数据
        images = [torch.randn(3, 32, 32), torch.randn(3, 32, 32)]
        texts = ["a cat", "a dog", "a bird"]
        
        similarity = engine.clip_image_text_similarity(images, texts)
        
        assert similarity.shape == (2, 3)
    
    def test_top_k_sampling(self, engine):
        """测试 Top-K 采样"""
        logits = torch.randn(2, 100)
        
        sampled = engine._top_k_sampling(logits, k=10)
        
        assert sampled.shape == (2, 1)
        assert sampled.dtype == torch.long
    
    def test_top_p_sampling(self, engine):
        """测试 Top-P 采样"""
        logits = torch.randn(2, 100)
        
        sampled = engine._top_p_sampling(logits, p=0.9)
        
        assert sampled.shape == (2, 1)
        assert sampled.dtype == torch.long
    
    def test_sample_next_token_greedy(self, engine):
        """测试贪婪解码"""
        logits = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        
        next_token = engine._sample_next_token(logits, decoding_strategy='greedy')
        
        assert next_token.shape == (2, 1)
        assert next_token[0, 0] == 2  # 最大值索引
        assert next_token[1, 0] == 0  # 最大值索引
    
    def test_sample_next_token_with_temperature(self, engine):
        """测试带温度的采样"""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        # 低温度应该更确定性
        next_token_low_temp = engine._sample_next_token(
            logits, temperature=0.1, decoding_strategy='greedy'
        )
        
        assert next_token_low_temp.shape == (1, 1)
    
    def test_get_model_info_no_model(self, engine):
        """测试获取模型信息（未加载模型）"""
        info = engine.get_model_info()
        
        assert info['loaded'] is False
    
    def test_get_model_info_with_model(self, engine, small_vision_config):
        """测试获取模型信息（已加载模型）"""
        engine.load_model('vit', None, small_vision_config)
        
        info = engine.get_model_info()
        
        assert info['loaded'] is True
        assert info['model_type'] == 'vit'
        assert 'num_parameters' in info
        assert info['num_parameters'] > 0
    
    def test_to_device(self, engine, small_vision_config):
        """测试设备切换"""
        engine.load_model('vit', None, small_vision_config)
        
        engine.to('cpu')
        
        assert str(engine.device) == 'cpu'
    
    def test_eval_mode(self, engine, small_vision_config):
        """测试评估模式"""
        engine.load_model('vit', None, small_vision_config)
        
        engine.eval()
        
        assert not engine.model.training
    
    def test_train_mode(self, engine, small_vision_config):
        """测试训练模式"""
        engine.load_model('vit', None, small_vision_config)
        
        engine.train()
        
        assert engine.model.training


class TestDecodingStrategies:
    """解码策略测试类"""
    
    @pytest.fixture
    def engine(self):
        return MultimodalInferenceEngine(device='cpu')
    
    def test_greedy_decoding_deterministic(self, engine):
        """测试贪婪解码的确定性"""
        logits = torch.tensor([[1.0, 5.0, 2.0, 3.0]])
        
        result1 = engine._sample_next_token(logits, decoding_strategy='greedy')
        result2 = engine._sample_next_token(logits, decoding_strategy='greedy')
        
        assert result1.item() == result2.item() == 1  # 索引 1 有最大值 5.0
    
    def test_top_k_filters_correctly(self, engine):
        """测试 Top-K 正确过滤"""
        # 创建一个有明显差异的 logits
        logits = torch.zeros(1, 100)
        logits[0, :5] = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0])
        
        # 多次采样，应该只从前 5 个中选择
        samples = []
        for _ in range(50):
            sample = engine._top_k_sampling(logits, k=5)
            samples.append(sample.item())
        
        # 所有样本应该在 [0, 4] 范围内
        assert all(0 <= s <= 4 for s in samples)
    
    def test_top_p_filters_correctly(self, engine):
        """测试 Top-P 正确过滤"""
        # 创建一个概率分布，前几个 token 占据大部分概率
        logits = torch.zeros(1, 100)
        logits[0, 0] = 10.0  # 高概率
        logits[0, 1] = 5.0   # 中等概率
        logits[0, 2] = 2.0   # 低概率
        
        # 多次采样
        samples = []
        for _ in range(50):
            sample = engine._top_p_sampling(logits, p=0.95)
            samples.append(sample.item())
        
        # 大多数样本应该是前几个 token
        assert sum(1 for s in samples if s <= 2) > 40
    
    def test_temperature_effect(self, engine):
        """测试温度参数的效果"""
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        
        # 低温度应该更集中在高概率 token
        low_temp_samples = []
        for _ in range(100):
            sample = engine._sample_next_token(
                logits.clone(), temperature=0.1, 
                top_k=5, decoding_strategy='top_k'
            )
            low_temp_samples.append(sample.item())
        
        # 高温度应该更分散
        high_temp_samples = []
        for _ in range(100):
            sample = engine._sample_next_token(
                logits.clone(), temperature=2.0,
                top_k=5, decoding_strategy='top_k'
            )
            high_temp_samples.append(sample.item())
        
        # 低温度时，索引 4（最大值）应该出现更频繁
        low_temp_max_count = sum(1 for s in low_temp_samples if s == 4)
        high_temp_max_count = sum(1 for s in high_temp_samples if s == 4)
        
        assert low_temp_max_count >= high_temp_max_count


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
