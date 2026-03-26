# 实现计划：多模态大模型从零实现

## 概述

本实现计划将多模态大模型项目分解为可执行的编码任务。项目依赖 bert-gpt-from-scratch 的核心组件，按照依赖关系排序：先实现基础组件（视觉编码、LLM 扩展），再实现多模态模型。所有代码使用 Python + PyTorch 实现。

## 前置条件

- bert-gpt-from-scratch 项目已完成，包含 Transformer 核心组件（Multi-Head Attention、Feed Forward Network、Encoder Layer、Decoder Layer、Tokenizer、Inference Engine）

## 任务列表

- [x] 1. 项目结构与配置模块
  - [x] 1.1 创建项目目录结构
    - 创建 multimodal_models_from_scratch/ 目录及子目录（vision/、llm/、multimodal/、detection/、training/、inference/、tests/）
    - 创建各目录的 __init__.py 文件
    - _需求: 17.1, 17.2, 17.3_
  - [x] 1.2 实现配置模块 config.py
    - 实现 VisionConfig、LLaMAConfig、QwenConfig 数据类
    - 实现 CLIPConfig、BLIPConfig、BLIP2Config 数据类
    - 实现 FlamingoConfig、LLaVAConfig、DETRConfig 数据类
    - 复用 bert-gpt-from-scratch 的 TransformerConfig
    - _需求: 2.5, 3.9, 4.9, 5.9, 6.9, 7.9, 8.9, 9.9, 10.9_

- [x] 2. 检查点 - 确保项目结构正确
  - 确保所有目录和配置文件创建成功，询问用户是否有问题

- [x] 3. 视觉编码组件
  - [x] 3.1 实现图像预处理模块 vision/image_processor.py
    - 实现 ImageProcessor 类，支持图像缩放、归一化
    - 支持从文件路径、PIL Image、numpy array、torch Tensor 加载图像
    - 使用 ImageNet 均值和标准差进行归一化
    - _需求: 16.1, 16.2, 16.3, 16.4, 16.5, 16.6_
  - [x] 3.2 编写 ImageProcessor 单元测试
    - 测试不同输入格式的处理
    - 测试输出形状和归一化值
    - _需求: 16.5_
  - [x] 3.3 实现 Patch Embedding 模块 vision/patch_embedding.py
    - 使用 Conv2d 实现 patch 分割和线性投影
    - 实现可学习的 [CLS] token
    - 实现可学习的位置嵌入
    - _需求: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_
  - [x] 3.4 编写 Patch Embedding 单元测试
    - 测试输出形状正确性
    - 测试 [CLS] token 和位置嵌入
    - _需求: 1.3_
  - [x] 3.5 实现 ViT 模型 vision/vit.py
    - 组合 Patch Embedding 和 Encoder Layer（复用 bert-gpt-from-scratch）
    - 实现 Layer Normalization 和分类头
    - 实现 get_image_features 方法用于多模态任务
    - _需求: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 17.1, 17.2, 17.3_
  - [x] 3.6 编写 ViT 模型单元测试
    - 测试前向传播输出形状
    - 测试分类头输出
    - _需求: 2.6_

- [x] 4. 检查点 - 确保视觉编码组件正确
  - 确保所有测试通过，询问用户是否有问题

- [x] 5. LLM 扩展组件
  - [x] 5.1 实现 RMSNorm 模块 llm/rmsnorm.py
    - 实现 Root Mean Square Layer Normalization
    - _需求: 7.1, 10.1_
  - [x] 5.2 实现 RoPE 模块 llm/rope.py
    - 实现旋转位置编码
    - 预计算 cos 和 sin 缓存
    - 支持 NTK-aware 插值扩展（用于 Qwen）
    - _需求: 7.2, 10.2_
  - [x] 5.3 实现 SwiGLU 模块 llm/swiglu.py
    - 实现 SwiGLU 激活函数的前馈网络
    - _需求: 7.3, 10.3_
  - [x] 5.4 实现 GQA 模块 llm/gqa.py
    - 实现 Grouped Query Attention
    - 支持 num_kv_heads < num_heads 的配置
    - 支持 KV Cache
    - _需求: 7.5, 10.4_
  - [x] 5.5 编写 LLM 组件单元测试
    - 测试 RMSNorm、RoPE、SwiGLU、GQA 的输出形状
    - 测试 KV Cache 功能
    - _需求: 7.8_
  - [x] 5.6 实现 LLaMA 模型 llm/llama.py
    - 实现 LLaMADecoderLayer（Pre-Norm 架构）
    - 实现 LLaMAModel，组合所有组件
    - 实现 KV Cache 支持
    - 实现 prepare_inputs_for_generation 方法
    - _需求: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 17.5_
  - [x] 5.7 编写 LLaMA 模型单元测试
    - 测试前向传播输出形状
    - 测试 KV Cache 生成
    - _需求: 7.8_
  - [x] 5.8 实现 Qwen 模型 llm/qwen.py
    - 继承 LLaMA 架构
    - 实现 NTK-aware RoPE 插值
    - 实现 Sliding Window Attention（可选）
    - _需求: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9_
  - [x] 5.9 编写 Qwen 模型单元测试
    - 测试 NTK-aware 插值
    - 测试 Sliding Window Attention
    - _需求: 10.7_

- [x] 6. 检查点 - 确保 LLM 扩展组件正确
  - 确保所有测试通过，询问用户是否有问题

- [x] 7. 多模态桥接组件
  - [x] 7.1 实现 Visual Projection 模块 multimodal/visual_projection.py
    - 实现线性投影和 MLP 投影两种模式
    - _需求: 4.3, 6.6, 9.2_
  - [x] 7.2 实现 Q-Former 模块 multimodal/qformer.py
    - 实现 QFormerLayer（自注意力 + 交叉注意力 + FFN）
    - 实现 QFormer，包含可学习的查询向量
    - _需求: 6.2, 6.3, 6.4, 6.5_
  - [x] 7.3 实现 Perceiver Resampler 模块 multimodal/perceiver.py
    - 实现可学习的 latent 向量
    - 实现交叉注意力聚合视觉特征
    - _需求: 8.2, 8.3_
  - [x] 7.4 实现 Gated Cross Attention 模块 multimodal/gated_cross_attention.py
    - 实现门控交叉注意力层
    - 实现可学习的门控参数 tanh(alpha)，初始化为 0
    - _需求: 8.5, 8.6_
  - [x] 7.5 编写多模态桥接组件单元测试
    - 测试各组件的输出形状
    - 测试门控参数初始化
    - _需求: 6.5, 8.3_

- [x] 8. 检查点 - 确保多模态桥接组件正确
  - 确保所有测试通过，询问用户是否有问题

- [x] 9. CLIP 模型
  - [x] 9.1 实现 CLIP 模型 multimodal/clip.py
    - 组合 Vision Encoder（ViT）和 Text Encoder（复用 bert-gpt-from-scratch）
    - 实现 Visual Projection 和 Text Projection
    - 实现 L2 归一化
    - 实现可学习的温度参数
    - _需求: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 17.4_
  - [x] 9.2 实现 CLIP 零样本分类
    - 实现 zero_shot_classify 方法
    - _需求: 4.8_
  - [x] 9.3 编写 CLIP 模型单元测试
    - 测试图像和文本编码
    - 测试相似度矩阵计算
    - 测试零样本分类
    - _需求: 4.7, 4.8_

- [x] 10. BLIP 模型
  - [x] 10.1 实现 BLIP 模型 multimodal/blip.py
    - 组合 Vision Encoder、Text Encoder、Text Decoder
    - 实现 Cross Attention 融合视觉特征
    - 实现 ITC、ITM、ITG 三个任务头
    - _需求: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_
  - [x] 10.2 实现 BLIP 推理方法
    - 实现 generate_caption 方法
    - 实现 visual_question_answering 方法
    - _需求: 5.7, 5.8_
  - [x] 10.3 编写 BLIP 模型单元测试
    - 测试 ITC、ITM、ITG 前向传播
    - 测试图像描述生成
    - _需求: 5.6_

- [x] 11. BLIP-2 模型
  - [x] 11.1 实现 BLIP-2 模型 multimodal/blip2.py
    - 组合冻结的 Vision Encoder、Q-Former、Visual Projection、冻结的 LLM
    - 实现两阶段训练前向传播
    - _需求: 6.1, 6.2, 6.6, 6.7, 6.8_
  - [x] 11.2 实现 BLIP-2 生成方法
    - 实现 generate 方法
    - _需求: 6.7_
  - [x] 11.3 编写 BLIP-2 模型单元测试
    - 测试两阶段前向传播
    - 测试视觉语言生成
    - _需求: 6.8_

- [x] 12. 检查点 - 确保 CLIP、BLIP、BLIP-2 模型正确
  - 确保所有测试通过，询问用户是否有问题

- [x] 13. Flamingo 模型
  - [x] 13.1 实现 Flamingo 模型 multimodal/flamingo.py
    - 组合冻结的 Vision Encoder、Perceiver Resampler、冻结的 LLM
    - 在 LLM 的 Decoder Layer 中插入 Gated Cross Attention
    - 支持多图像输入
    - _需求: 8.1, 8.4, 8.5, 8.6, 8.7, 8.8_
  - [x] 13.2 实现 Flamingo 生成方法
    - 实现 generate 方法，支持多图像条件生成
    - _需求: 8.7_
  - [x] 13.3 编写 Flamingo 模型单元测试
    - 测试多图像输入处理
    - 测试门控交叉注意力
    - _需求: 8.7_

- [x] 14. LLaVA 模型
  - [x] 14.1 实现 LLaVA 模型 multimodal/llava.py
    - 组合 Vision Encoder、Visual Projection（MLP）、LLM
    - 实现视觉 token 插入到 <image> 位置
    - 支持多轮对话格式
    - _需求: 9.1, 9.2, 9.3, 9.4, 9.5, 9.8_
  - [x] 14.2 实现 LLaVA 生成方法
    - 实现 prepare_inputs_for_generation 方法
    - 实现 generate 方法
    - _需求: 9.8_
  - [x] 14.3 编写 LLaVA 模型单元测试
    - 测试视觉 token 插入
    - 测试多轮对话处理
    - _需求: 9.8_

- [x] 15. 检查点 - 确保 Flamingo、LLaVA 模型正确
  - 确保所有测试通过，询问用户是否有问题

- [ ] 16. DETR 目标检测模型
  - [x] 16.1 实现 CNN Backbone vision/backbone.py
    - 实现简化版 ResNet 作为 backbone
    - _需求: 3.1_
  - [x] 16.2 实现匈牙利匹配 detection/hungarian.py
    - 实现 HungarianMatcher 类
    - 使用 scipy.optimize.linear_sum_assignment
    - 计算分类代价、L1 边界框代价、GIoU 代价
    - _需求: 3.7, 14.1, 14.2_
  - [x] 16.3 实现 DETR 损失函数 detection/losses.py
    - 实现 DETRLoss 类
    - 包含分类损失、L1 边界框损失、GIoU 损失
    - _需求: 3.8, 14.3, 14.4, 14.5_
  - [x] 16.4 实现 DETR 模型 detection/detr.py
    - 组合 CNN Backbone、2D 位置编码、Transformer Encoder/Decoder
    - 实现 Object Queries
    - 实现分类头和边界框回归头
    - _需求: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.9_
  - [-] 16.5 编写 DETR 模型单元测试
    - 测试匈牙利匹配
    - 测试损失计算
    - 测试前向传播
    - _需求: 3.7, 3.8_

- [ ] 17. 检查点 - 确保 DETR 模型正确
  - 确保所有测试通过，询问用户是否有问题

- [ ] 18. 训练模块
  - [~] 18.1 实现对比学习训练 training/contrastive.py
    - 实现 ContrastiveTrainer 类
    - 实现 InfoNCE Loss
    - 支持可学习的温度参数
    - 支持梯度累积
    - _需求: 11.1, 11.2, 11.3, 11.4, 11.6, 11.7, 17.6_
  - [~] 18.2 实现多模态预训练 training/multimodal_pretrain.py
    - 实现 MultimodalPreTrainer 类
    - 实现 ITC、ITM、ITG 损失
    - 实现 hard negative mining
    - 支持多任务联合训练
    - _需求: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7_
  - [~] 18.3 实现视觉指令微调训练 training/visual_instruction.py
    - 实现 VisualInstructionTrainer 类
    - 实现指令数据预处理
    - 实现两阶段训练
    - 仅对 response 部分计算损失
    - _需求: 13.1, 13.2, 13.3, 13.4, 13.6, 13.7, 13.8_
  - [~] 18.4 实现目标检测训练 training/detection_train.py
    - 实现 DetectionTrainer 类
    - 集成匈牙利匹配和 DETR 损失
    - 支持数据增强
    - _需求: 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.7_
  - [~] 18.5 实现训练工具函数 training/utils.py
    - 复用 bert-gpt-from-scratch 的训练工具函数
    - 实现训练配置数据类
    - _需求: 17.6_
  - [~] 18.6 编写训练模块单元测试
    - 测试各训练器的 train_step
    - 测试损失计算
    - _需求: 11.7, 12.7, 13.8, 14.7_

- [ ] 19. 检查点 - 确保训练模块正确
  - 确保所有测试通过，询问用户是否有问题

- [ ] 20. 推理模块
  - [~] 20.1 实现多模态推理引擎 inference/multimodal_engine.py
    - 实现 MultimodalInferenceEngine 类
    - 实现模型加载方法
    - 复用 bert-gpt-from-scratch 的解码策略
    - _需求: 15.1, 15.9, 15.10, 15.11, 15.12, 17.7_
  - [~] 20.2 实现 ViT 和 DETR 推理
    - 实现 vit_classify 方法
    - 实现 detr_detect 方法
    - _需求: 15.2, 15.3_
  - [~] 20.3 实现 CLIP 推理
    - 实现 clip_zero_shot_classify 方法
    - 实现 clip_image_text_similarity 方法
    - _需求: 15.4, 15.5_
  - [~] 20.4 实现 BLIP 推理
    - 实现 blip_caption 方法
    - 实现 blip_vqa 方法
    - _需求: 15.6, 15.7_
  - [~] 20.5 实现 LLaVA 推理
    - 实现 llava_chat 方法
    - 支持多轮对话
    - _需求: 15.8_
  - [~] 20.6 编写推理模块单元测试
    - 测试各推理方法
    - 测试解码策略
    - _需求: 15.9, 15.10, 15.11_

- [ ] 21. 最终检查点 - 确保所有模块正确集成
  - 确保所有测试通过，询问用户是否有问题

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加快 MVP 开发
- 每个任务都引用了具体的需求条款以便追溯
- 检查点任务用于阶段性验证，确保增量开发的正确性
- 本项目依赖 bert-gpt-from-scratch 的核心组件，请确保该项目已完成
- 属性测试和单元测试相辅相成，验证代码的正确性
