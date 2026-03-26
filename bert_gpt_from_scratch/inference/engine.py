"""
推理引擎实现

支持：
- BERT MLM 填空推理
- BERT 文本分类推理
- GPT 自回归文本生成（greedy, top-k, top-p）
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import BERTConfig, GPTConfig, TransformerConfig
from ..models.bert import BERTModel
from ..models.gpt import GPTModel
from ..tokenizer.simple_tokenizer import SimpleTokenizer


class InferenceEngine:
    """推理引擎"""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推理引擎
        
        Args:
            device: 推理设备
        """
        self.device = torch.device(device)
        self.model: Optional[Union[BERTModel, GPTModel]] = None
        self.tokenizer: Optional[SimpleTokenizer] = None
        self.model_type: Optional[str] = None
        self.classification_head: Optional[nn.Module] = None
    
    def load_model(
        self,
        model_type: str,
        checkpoint_path: str,
        config: TransformerConfig,
        tokenizer: Optional[SimpleTokenizer] = None
    ) -> None:
        """
        加载模型检查点
        
        Args:
            model_type: 'bert' 或 'gpt'
            checkpoint_path: 检查点文件路径
            config: 模型配置
            tokenizer: Tokenizer 实例
        """
        self.model_type = model_type.lower()
        
        # 创建模型
        if self.model_type == 'bert':
            self.model = BERTModel(config)
        elif self.model_type == 'gpt':
            self.model = GPTModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载分类头（如果有）
        if 'classification_head_state_dict' in checkpoint:
            from ..training.sft import ClassificationHead
            num_classes = checkpoint['classification_head_state_dict']['classifier.weight'].shape[0]
            self.classification_head = ClassificationHead(
                config.d_model, num_classes
            )
            self.classification_head.load_state_dict(
                checkpoint['classification_head_state_dict']
            )
            self.classification_head.to(self.device)
            self.classification_head.eval()
        
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = tokenizer
    
    # ==================== BERT 推理方法 ====================
    
    def bert_fill_mask(
        self,
        text: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        MLM 填空推理
        
        Args:
            text: 包含 [MASK] 的文本
            top_k: 返回 top-k 个预测结果
        
        Returns:
            [(predicted_token, probability), ...]
        """
        if self.model_type != 'bert':
            raise ValueError("bert_fill_mask is only supported for BERT models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for inference")
        
        # 编码输入
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        segment_ids = torch.zeros_like(input_ids)
        
        # 找到 [MASK] 位置
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if len(mask_positions[1]) == 0:
            raise ValueError("No [MASK] token found in input text")
        
        mask_pos = mask_positions[1][0].item()
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids)
            mlm_logits = outputs['mlm_logits']
        
        # 获取 [MASK] 位置的 logits
        mask_logits = mlm_logits[0, mask_pos, :]
        probs = F.softmax(mask_logits, dim=-1)
        
        # 获取 top-k 预测
        top_probs, top_indices = torch.topk(probs, top_k)
        
        results = []
        vocab = self.tokenizer.get_vocab()
        id_to_token = {v: k for k, v in vocab.items()}
        
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            token = id_to_token.get(idx, "[UNK]")
            results.append((token, prob))
        
        return results
    
    def bert_classify(
        self,
        text: str,
        num_classes: int
    ) -> Tuple[int, torch.Tensor]:
        """
        文本分类推理
        
        Args:
            text: 输入文本
            num_classes: 分类类别数
        
        Returns:
            (predicted_class, class_probabilities)
        """
        if self.model_type != 'bert':
            raise ValueError("bert_classify is only supported for BERT models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for inference")
        
        if self.classification_head is None:
            raise ValueError("Classification head not loaded. Load a fine-tuned checkpoint.")
        
        # 编码输入
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], device=self.device)
        segment_ids = torch.zeros_like(input_ids)
        
        # 前向传播
        with torch.no_grad():
            outputs = self.model(input_ids, segment_ids)
            cls_hidden_state = outputs['hidden_states'][:, 0, :]
            logits = self.classification_head(cls_hidden_state)
        
        probs = F.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=-1).item()
        
        return predicted_class, probs.squeeze()
    
    # ==================== GPT 推理方法 ====================
    
    def gpt_generate(
        self,
        prompt: str,
        max_gen_len: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        decoding_strategy: str = 'greedy'
    ) -> str:
        """
        自回归文本生成
        
        Args:
            prompt: 输入提示
            max_gen_len: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P（Nucleus）采样的 P 值
            decoding_strategy: 解码策略 ('greedy', 'top_k', 'top_p')
        
        Returns:
            生成的完整文本
        """
        if self.model_type != 'gpt':
            raise ValueError("gpt_generate is only supported for GPT models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for inference")
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        generated_ids = input_ids.clone()
        
        # 自回归生成
        for _ in range(max_gen_len):
            # 截断到最大序列长度
            if generated_ids.size(1) >= self.model.config.max_seq_len:
                generated_ids = generated_ids[:, -self.model.config.max_seq_len + 1:]
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(generated_ids)
                logits = outputs['logits']
            
            # 获取最后一个位置的 logits
            next_token_logits = logits[:, -1, :]
            
            # 应用温度
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # 选择下一个 token
            if decoding_strategy == 'greedy':
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            elif decoding_strategy == 'top_k':
                next_token = self._top_k_sampling(next_token_logits, top_k or 50)
            elif decoding_strategy == 'top_p':
                next_token = self._top_p_sampling(next_token_logits, top_p or 0.9)
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
            
            # 拼接生成的 token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 检查是否生成了 EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        return generated_text
    
    def _top_k_sampling(
        self,
        logits: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """
        Top-K 采样
        
        Args:
            logits: (batch, vocab_size)
            k: K 值
        
        Returns:
            sampled_token: (batch, 1)
        """
        # 获取 top-k logits
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        
        # 计算概率
        probs = F.softmax(top_k_logits, dim=-1)
        
        # 采样
        sampled_idx = torch.multinomial(probs, num_samples=1)
        sampled_token = torch.gather(top_k_indices, -1, sampled_idx)
        
        return sampled_token
    
    def _top_p_sampling(
        self,
        logits: torch.Tensor,
        p: float
    ) -> torch.Tensor:
        """
        Top-P (Nucleus) 采样
        
        Args:
            logits: (batch, vocab_size)
            p: P 值（累积概率阈值）
        
        Returns:
            sampled_token: (batch, 1)
        """
        # 排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 找到累积概率超过 p 的位置
        sorted_indices_to_remove = cumulative_probs > p
        # 保留第一个超过阈值的 token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        # 将要移除的位置设为负无穷
        sorted_logits[sorted_indices_to_remove] = float('-inf')
        
        # 重新计算概率并采样
        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx = torch.multinomial(probs, num_samples=1)
        sampled_token = torch.gather(sorted_indices, -1, sampled_idx)
        
        return sampled_token
