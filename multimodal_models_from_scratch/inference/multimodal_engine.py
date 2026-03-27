"""
多模态推理引擎

支持多种多模态模型的推理：
- ViT: 图像分类
- CLIP: 零样本分类、图文检索
- BLIP: 图像描述生成、视觉问答
- BLIP-2: 视觉语言生成
- Flamingo: 多图像条件生成
- LLaVA: 视觉对话
- DETR: 目标检测

复用 bert-gpt-from-scratch 的解码策略（greedy, top-k, top-p）。

需求: 15.1, 15.9, 15.10, 15.11, 15.12, 17.7
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..config import (
    VisionConfig,
    CLIPConfig,
    BLIPConfig,
    BLIP2Config,
    FlamingoConfig,
    LLaVAConfig,
    DETRConfig,
)
from ..vision.vit import ViTModel
from ..vision.image_processor import ImageProcessor
from ..multimodal.clip import CLIPModel
from ..multimodal.blip import BLIPModel
from ..multimodal.blip2 import BLIP2Model
from ..multimodal.flamingo import FlamingoModel
from ..multimodal.llava import LLaVAModel
from ..detection.detr import DETR


# 支持的模型类型
SUPPORTED_MODEL_TYPES = ['vit', 'clip', 'blip', 'blip2', 'flamingo', 'llava', 'detr']


class MultimodalInferenceEngine:
    """
    多模态推理引擎
    
    支持加载和运行多种多模态模型的推理。
    复用 bert-gpt-from-scratch 的解码策略。
    
    Args:
        device: 推理设备，默认自动选择 CUDA 或 CPU
    
    Attributes:
        device: 推理设备
        model: 加载的模型
        model_type: 模型类型
        image_processor: 图像预处理器
        tokenizer: 分词器（如果需要）
    
    Examples:
        >>> engine = MultimodalInferenceEngine()
        >>> engine.load_model('vit', 'checkpoint.pt', VisionConfig())
        >>> results = engine.vit_classify(image, top_k=5)
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化推理引擎
        
        Args:
            device: 推理设备
        """
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.model_type: Optional[str] = None
        self.image_processor: Optional[ImageProcessor] = None
        self.tokenizer: Optional[Any] = None
        self.config: Optional[Any] = None
    
    def load_model(
        self,
        model_type: str,
        checkpoint_path: Optional[str],
        config: Any,
        tokenizer: Optional[Any] = None,
        image_processor: Optional[ImageProcessor] = None,
        **kwargs
    ) -> None:
        """
        加载模型检查点
        
        Args:
            model_type: 模型类型，支持 'vit', 'clip', 'blip', 'blip2', 'flamingo', 'llava', 'detr'
            checkpoint_path: 检查点文件路径，如果为 None 则使用随机初始化
            config: 模型配置对象
            tokenizer: 分词器（用于文本处理的模型）
            image_processor: 图像预处理器，如果为 None 则自动创建
            **kwargs: 额外参数（如 DETR 的 backbone_name）
        
        Raises:
            ValueError: 如果模型类型不支持
        """
        model_type = model_type.lower()
        
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {SUPPORTED_MODEL_TYPES}"
            )
        
        self.model_type = model_type
        self.config = config
        self.tokenizer = tokenizer
        
        # 创建模型
        if model_type == 'vit':
            self.model = ViTModel(config)
            image_size = config.image_size
        elif model_type == 'clip':
            self.model = CLIPModel(config)
            image_size = config.vision_config.image_size
        elif model_type == 'blip':
            self.model = BLIPModel(config)
            image_size = config.vision_config.image_size
        elif model_type == 'blip2':
            self.model = BLIP2Model(config)
            image_size = config.vision_config.image_size
        elif model_type == 'flamingo':
            self.model = FlamingoModel(config)
            image_size = config.vision_config.image_size
        elif model_type == 'llava':
            self.model = LLaVAModel(config)
            image_size = config.vision_config.image_size
        elif model_type == 'detr':
            backbone_name = kwargs.get('backbone_name', 'resnet50')
            frozen_bn = kwargs.get('frozen_bn', True)
            self.model = DETR(config, backbone_name=backbone_name, frozen_bn=frozen_bn)
            image_size = 800  # DETR 默认使用较大的图像尺寸
        
        # 加载检查点（如果提供）
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 支持不同的检查点格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # 假设检查点直接是 state_dict
                self.model.load_state_dict(checkpoint)
        
        # 移动到设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 创建图像预处理器
        if image_processor is not None:
            self.image_processor = image_processor
        else:
            self.image_processor = ImageProcessor(image_size=image_size)
    
    def _preprocess_image(
        self,
        image: Union[str, Image.Image, torch.Tensor]
    ) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image: 输入图像（文件路径、PIL Image 或 Tensor）
        
        Returns:
            pixel_values: (1, 3, H, W) 预处理后的图像张量
        """
        if self.image_processor is None:
            raise ValueError("Image processor not initialized. Call load_model first.")
        
        result = self.image_processor(image, return_tensors='pt')
        pixel_values = result['pixel_values'].to(self.device)
        
        return pixel_values
    
    # ==================== 解码策略（复用 bert-gpt-from-scratch）====================
    
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
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        decoding_strategy: str = 'greedy'
    ) -> torch.Tensor:
        """
        采样下一个 token
        
        Args:
            logits: (batch, vocab_size) 最后一个位置的 logits
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            decoding_strategy: 解码策略 ('greedy', 'top_k', 'top_p')
        
        Returns:
            next_token: (batch, 1) 采样的 token
        """
        # 应用温度
        if temperature != 1.0:
            logits = logits / temperature
        
        # 选择下一个 token
        if decoding_strategy == 'greedy':
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        elif decoding_strategy == 'top_k':
            next_token = self._top_k_sampling(logits, top_k or 50)
        elif decoding_strategy == 'top_p':
            next_token = self._top_p_sampling(logits, top_p or 0.9)
        else:
            raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")
        
        return next_token

    # ==================== ViT 推理方法 ====================
    
    @torch.no_grad()
    def vit_classify(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        ViT 图像分类推理
        
        Args:
            image: 输入图像
            top_k: 返回 top-k 个预测结果
        
        Returns:
            [(class_id, probability), ...] 预测结果列表
        
        Raises:
            ValueError: 如果模型类型不是 vit
        """
        if self.model_type != 'vit':
            raise ValueError("vit_classify is only supported for ViT models")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 前向传播
        outputs = self.model(pixel_values)
        
        if 'logits' not in outputs:
            raise ValueError("ViT model does not have a classification head")
        
        logits = outputs['logits']  # (1, num_classes)
        probs = F.softmax(logits, dim=-1)
        
        # 获取 top-k 预测
        top_probs, top_indices = torch.topk(probs[0], top_k)
        
        results = [
            (idx.item(), prob.item())
            for idx, prob in zip(top_indices, top_probs)
        ]
        
        return results
    
    # ==================== DETR 推理方法 ====================
    
    @torch.no_grad()
    def detr_detect(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        DETR 目标检测推理
        
        Args:
            image: 输入图像
            threshold: 置信度阈值
        
        Returns:
            [{'label': int, 'score': float, 'box': [x1, y1, x2, y2]}, ...]
            边界框坐标为归一化坐标 [0, 1]
        
        Raises:
            ValueError: 如果模型类型不是 detr
        """
        if self.model_type != 'detr':
            raise ValueError("detr_detect is only supported for DETR models")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 前向传播
        outputs = self.model(pixel_values)
        
        pred_logits = outputs['pred_logits']  # (1, num_queries, num_classes + 1)
        pred_boxes = outputs['pred_boxes']    # (1, num_queries, 4) - (cx, cy, w, h)
        
        # 计算概率（排除背景类）
        probs = F.softmax(pred_logits, dim=-1)[0, :, :-1]  # (num_queries, num_classes)
        
        # 获取每个查询的最大概率和对应类别
        max_probs, labels = probs.max(dim=-1)  # (num_queries,)
        
        # 过滤低置信度的预测
        keep = max_probs > threshold
        
        # 转换边界框格式：(cx, cy, w, h) -> (x1, y1, x2, y2)
        boxes = pred_boxes[0]  # (num_queries, 4)
        cx, cy, w, h = boxes.unbind(-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)
        
        # 构建结果
        results = []
        for i in range(len(keep)):
            if keep[i]:
                results.append({
                    'label': labels[i].item(),
                    'score': max_probs[i].item(),
                    'box': boxes_xyxy[i].tolist()
                })
        
        # 按置信度排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
    # ==================== CLIP 推理方法 ====================
    
    @torch.no_grad()
    def clip_zero_shot_classify(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        text_labels: List[str]
    ) -> Tuple[str, torch.Tensor]:
        """
        CLIP 零样本图像分类
        
        Args:
            image: 输入图像
            text_labels: 文本标签列表
        
        Returns:
            (predicted_label, probabilities) 预测的标签和概率分布
        
        Raises:
            ValueError: 如果模型类型不是 clip 或没有 tokenizer
        """
        if self.model_type != 'clip':
            raise ValueError("clip_zero_shot_classify is only supported for CLIP models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for CLIP zero-shot classification")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 使用模型的 zero_shot_classify 方法
        predicted_indices, probabilities = self.model.zero_shot_classify(
            pixel_values, text_labels, self.tokenizer
        )
        
        predicted_label = text_labels[predicted_indices[0].item()]
        
        return predicted_label, probabilities[0]
    
    @torch.no_grad()
    def clip_image_text_similarity(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str]
    ) -> torch.Tensor:
        """
        计算图文相似度矩阵
        
        Args:
            images: 图像列表
            texts: 文本列表
        
        Returns:
            similarity: (num_images, num_texts) 相似度矩阵
        
        Raises:
            ValueError: 如果模型类型不是 clip 或没有 tokenizer
        """
        if self.model_type != 'clip':
            raise ValueError("clip_image_text_similarity is only supported for CLIP models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for CLIP similarity computation")
        
        # 预处理所有图像
        pixel_values_list = []
        for img in images:
            pv = self._preprocess_image(img)
            pixel_values_list.append(pv)
        pixel_values = torch.cat(pixel_values_list, dim=0)  # (num_images, 3, H, W)
        
        # 编码图像
        image_embeds = self.model.encode_image(pixel_values)  # (num_images, projection_dim)
        
        # 编码文本
        tokenized = self.tokenizer(texts)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized.get('attention_mask', None)
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
        input_ids = input_ids.to(self.device)
        
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor):
                attention_mask = torch.tensor(attention_mask)
            attention_mask = attention_mask.to(self.device)
        
        text_embeds = self.model.encode_text(input_ids, attention_mask)  # (num_texts, projection_dim)
        
        # 计算相似度矩阵
        similarity = image_embeds @ text_embeds.T  # (num_images, num_texts)
        
        return similarity
    
    # ==================== BLIP 推理方法 ====================
    
    @torch.no_grad()
    def blip_caption(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102
    ) -> str:
        """
        BLIP 图像描述生成
        
        Args:
            image: 输入图像
            max_length: 最大生成长度
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
        
        Returns:
            caption: 生成的图像描述
        
        Raises:
            ValueError: 如果模型类型不是 blip 或没有 tokenizer
        """
        if self.model_type != 'blip':
            raise ValueError("blip_caption is only supported for BLIP models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BLIP caption generation")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 生成描述
        generated_ids = self.model.generate_caption(
            pixel_values,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        
        # 解码
        caption = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        return caption
    
    @torch.no_grad()
    def blip_vqa(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        question: str,
        max_length: int = 30,
        bos_token_id: int = 101,
        eos_token_id: int = 102
    ) -> str:
        """
        BLIP 视觉问答
        
        Args:
            image: 输入图像
            question: 问题文本
            max_length: 最大生成长度
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
        
        Returns:
            answer: 生成的答案
        
        Raises:
            ValueError: 如果模型类型不是 blip 或没有 tokenizer
        """
        if self.model_type != 'blip':
            raise ValueError("blip_vqa is only supported for BLIP models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BLIP VQA")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 编码问题
        question_ids = self.tokenizer.encode(question)
        if not isinstance(question_ids, torch.Tensor):
            question_ids = torch.tensor([question_ids])
        question_ids = question_ids.to(self.device)
        
        # 生成答案
        answer_ids = self.model.visual_question_answering(
            pixel_values,
            question_ids,
            max_length=max_length,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        
        # 解码
        answer = self.tokenizer.decode(
            answer_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        return answer

    # ==================== BLIP-2 推理方法 ====================
    
    @torch.no_grad()
    def blip2_generate(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        prompt: Optional[str] = None,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2
    ) -> str:
        """
        BLIP-2 视觉语言生成
        
        Args:
            image: 输入图像
            prompt: 可选的文本提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            bos_token_id: 开始 token ID
            eos_token_id: 结束 token ID
        
        Returns:
            generated_text: 生成的文本
        
        Raises:
            ValueError: 如果模型类型不是 blip2 或没有 tokenizer
        """
        if self.model_type != 'blip2':
            raise ValueError("blip2_generate is only supported for BLIP-2 models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BLIP-2 generation")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 编码提示（如果有）
        prompt_ids = None
        if prompt is not None:
            prompt_ids = self.tokenizer.encode(prompt)
            if not isinstance(prompt_ids, torch.Tensor):
                prompt_ids = torch.tensor([prompt_ids])
            prompt_ids = prompt_ids.to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            pixel_values,
            prompt_ids=prompt_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        
        # 解码
        generated_text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        return generated_text
    
    # ==================== Flamingo 推理方法 ====================
    
    @torch.no_grad()
    def flamingo_generate(
        self,
        images: Union[torch.Tensor, List[Union[str, Image.Image]]],
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None
    ) -> str:
        """
        Flamingo 多图像条件生成
        
        Args:
            images: 图像张量 (batch, num_images, 3, H, W) 或图像列表
            input_ids: 输入 token ID
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            eos_token_id: 结束 token ID
        
        Returns:
            generated_text: 生成的文本
        
        Raises:
            ValueError: 如果模型类型不是 flamingo 或没有 tokenizer
        """
        if self.model_type != 'flamingo':
            raise ValueError("flamingo_generate is only supported for Flamingo models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for Flamingo generation")
        
        # 处理图像输入
        if isinstance(images, list):
            # 预处理图像列表
            pixel_values_list = []
            for img in images:
                pv = self._preprocess_image(img)
                pixel_values_list.append(pv)
            # 堆叠为 (1, num_images, 3, H, W)
            pixel_values = torch.stack([pv.squeeze(0) for pv in pixel_values_list], dim=0)
            pixel_values = pixel_values.unsqueeze(0)
        else:
            pixel_values = images.to(self.device)
        
        # 确保 input_ids 在正确的设备上
        input_ids = input_ids.to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            input_ids=input_ids,
            images=pixel_values,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id
        )
        
        # 解码
        generated_text = self.tokenizer.decode(
            generated_ids[0].tolist(),
            skip_special_tokens=True
        )
        
        return generated_text
    
    # ==================== LLaVA 推理方法 ====================
    
    @torch.no_grad()
    def llava_chat(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        messages: List[Dict[str, str]],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.9,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        image_token_index: int = -200
    ) -> str:
        """
        LLaVA 视觉对话
        
        Args:
            image: 输入图像
            messages: 对话消息列表 [{'role': 'user/assistant', 'content': str}, ...]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            do_sample: 是否使用采样
            eos_token_id: 结束 token ID
            image_token_index: <image> token 的 ID
        
        Returns:
            response: 生成的回复
        
        Raises:
            ValueError: 如果模型类型不是 llava 或没有 tokenizer
        """
        if self.model_type != 'llava':
            raise ValueError("llava_chat is only supported for LLaVA models")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for LLaVA chat")
        
        # 预处理图像
        pixel_values = self._preprocess_image(image)
        
        # 构建对话文本
        conversation_text = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'user':
                conversation_text += f"User: {content}\n"
            elif role == 'assistant':
                conversation_text += f"Assistant: {content}\n"
        
        # 添加 Assistant 前缀以开始生成
        if not conversation_text.endswith("Assistant: "):
            conversation_text += "Assistant: "
        
        # 编码对话
        input_ids = self.tokenizer.encode(conversation_text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids])
        input_ids = input_ids.to(self.device)
        
        # 生成
        generated_ids = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=eos_token_id,
            image_token_index=image_token_index
        )
        
        # 解码生成的部分（不含输入）
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids.shape[1]:].tolist(),
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    # ==================== 通用生成方法 ====================
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_gen_len: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        decoding_strategy: str = 'greedy',
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        通用自回归文本生成
        
        复用 bert-gpt-from-scratch 的解码策略。
        
        Args:
            input_ids: 输入 token IDs (batch, seq_len)
            max_gen_len: 最大生成长度
            temperature: 温度参数
            top_k: Top-K 采样的 K 值
            top_p: Top-P 采样的 P 值
            decoding_strategy: 解码策略 ('greedy', 'top_k', 'top_p')
            eos_token_id: 结束 token ID
            pad_token_id: 填充 token ID
            **kwargs: 额外参数（如 pixel_values）
        
        Returns:
            generated_ids: 生成的 token IDs (batch, generated_len)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model first.")
        
        # 确保 input_ids 在正确的设备上
        input_ids = input_ids.to(self.device)
        batch_size = input_ids.shape[0]
        
        # 初始化生成的 IDs
        generated_ids = input_ids.clone()
        
        # 记录每个序列是否已结束
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        
        # 获取额外参数
        pixel_values = kwargs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)
        
        # 自回归生成
        past_key_values = None
        
        for _ in range(max_gen_len):
            # 准备输入
            if past_key_values is not None:
                # 只需要最后一个 token
                current_input_ids = generated_ids[:, -1:]
            else:
                current_input_ids = generated_ids
            
            # 前向传播
            if self.model_type in ['blip2', 'flamingo', 'llava']:
                # 多模态模型需要特殊处理
                if past_key_values is None and pixel_values is not None:
                    outputs = self.model(
                        input_ids=current_input_ids,
                        pixel_values=pixel_values if self.model_type == 'llava' else None,
                        images=pixel_values if self.model_type == 'flamingo' else None,
                        use_cache=True
                    )
                else:
                    outputs = self.model(
                        input_ids=current_input_ids,
                        past_key_values=past_key_values,
                        use_cache=True
                    )
            else:
                outputs = self.model(current_input_ids)
            
            # 获取 logits
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
                past_key_values = outputs.get('past_key_values')
            else:
                logits = outputs
            
            if logits is None:
                raise ValueError("Model output does not contain logits")
            
            # 获取最后一个位置的 logits
            next_token_logits = logits[:, -1, :]
            
            # 采样下一个 token
            next_token = self._sample_next_token(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                decoding_strategy=decoding_strategy
            )
            
            # 对于已结束的序列，使用 pad_token_id
            if pad_token_id is not None:
                next_token = next_token * unfinished_sequences.unsqueeze(-1) + \
                            pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            
            # 拼接生成的 token
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # 检查是否遇到 EOS
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences * (next_token.squeeze(-1) != eos_token_id).long()
                if unfinished_sequences.max() == 0:
                    break
        
        return generated_ids
    
    # ==================== 工具方法 ====================
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前加载的模型信息
        
        Returns:
            模型信息字典
        """
        if self.model is None:
            return {'loaded': False}
        
        num_params = sum(p.numel() for p in self.model.parameters())
        num_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'loaded': True,
            'model_type': self.model_type,
            'device': str(self.device),
            'num_parameters': num_params,
            'num_trainable_parameters': num_trainable,
            'config': self.config
        }
    
    def to(self, device: str) -> 'MultimodalInferenceEngine':
        """
        移动模型到指定设备
        
        Args:
            device: 目标设备
        
        Returns:
            self
        """
        self.device = torch.device(device)
        if self.model is not None:
            self.model.to(self.device)
        return self
    
    def eval(self) -> 'MultimodalInferenceEngine':
        """
        设置模型为评估模式
        
        Returns:
            self
        """
        if self.model is not None:
            self.model.eval()
        return self
    
    def train(self) -> 'MultimodalInferenceEngine':
        """
        设置模型为训练模式
        
        Returns:
            self
        """
        if self.model is not None:
            self.model.train()
        return self
