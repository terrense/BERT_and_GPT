# 需求文档

## 简介

本项目旨在从零实现多模态大模型和计算机视觉模型的完整代码。涵盖 CV 基础模型（ViT、DETR）、多模态对齐模型（CLIP、BLIP/BLIP-2）、以及多模态大语言模型（Flamingo、LLaMA、LLaVA、Qwen 系列）。项目复用 bert-gpt-from-scratch 中的 Transformer 核心组件，使用 Python + PyTorch 构建，不依赖 HuggingFace Transformers 等高层封装库的模型实现。

## 术语表

- **Vision_Encoder**: 视觉编码器，将图像转换为特征表示的模块
- **ViT_Model**: Vision Transformer，将图像分割为 patch 并通过 Transformer 编码的视觉模型
- **DETR_Model**: Detection Transformer，基于 Transformer 的端到端目标检测模型
- **CLIP_Model**: Contrastive Language-Image Pre-training，图文对比学习模型
- **BLIP_Model**: Bootstrapping Language-Image Pre-training，图文理解与生成模型
- **BLIP2_Model**: BLIP-2，使用 Q-Former 桥接视觉和语言模型的多模态模型
- **Flamingo_Model**: 支持视觉-语言交错输入的多模态模型
- **LLaMA_Model**: Large Language Model Meta AI，Meta 开源的大语言模型
- **LLaVA_Model**: Large Language and Vision Assistant，视觉指令微调模型
- **Qwen_Model**: 阿里通义千问系列大语言模型
- **Patch_Embedding**: 图像块嵌入，将图像分割为固定大小的 patch 并映射到嵌入空间
- **Q_Former**: Querying Transformer，BLIP-2 中用于提取视觉特征的查询模块
- **Perceiver_Resampler**: Flamingo 中用于将可变长度视觉特征压缩为固定数量 token 的模块
- **Cross_Attention**: 交叉注意力，用于在不同模态之间进行信息交互
- **Gated_Cross_Attention**: 门控交叉注意力，Flamingo 中带门控机制的交叉注意力层
- **Visual_Projection**: 视觉投影层，将视觉特征映射到语言模型的嵌入空间
- **Contrastive_Loss**: 对比损失，用于拉近匹配的图文对、推远不匹配的图文对
- **RoPE**: Rotary Position Embedding，旋转位置编码
- **RMSNorm**: Root Mean Square Layer Normalization
- **SwiGLU**: Swish-Gated Linear Unit，一种门控激活函数
- **GQA**: Grouped Query Attention，分组查询注意力
- **Object_Query**: 目标查询，DETR 中用于预测目标的可学习查询向量
- **Hungarian_Matching**: 匈牙利匹配算法，DETR 中用于将预测与真实目标匹配
- **Bipartite_Matching_Loss**: 二分图匹配损失，DETR 的训练损失

## 需求

### 需求 1：Patch Embedding 组件

**用户故事：** 作为开发者，我希望实现图像块嵌入模块，以便将图像转换为 Transformer 可处理的序列格式。

#### 验收标准

1. THE Vision_Encoder SHALL 实现 Patch_Embedding，将输入图像分割为固定大小的 patch（默认 16x16）
2. THE Patch_Embedding SHALL 使用卷积层将每个 patch 映射到 d_model 维的嵌入向量
3. WHEN 输入图像尺寸为 (batch, 3, H, W) 时，THE Patch_Embedding SHALL 输出形状为 (batch, num_patches, d_model) 的张量，其中 num_patches = (H/patch_size) * (W/patch_size)
4. THE Patch_Embedding SHALL 支持配置 patch_size、d_model、image_size 参数
5. THE Patch_Embedding SHALL 添加可学习的 [CLS] token 到序列开头
6. THE Patch_Embedding SHALL 添加可学习的位置嵌入到每个 patch 嵌入

### 需求 2：ViT 模型（Vision Transformer）

**用户故事：** 作为开发者，我希望实现完整的 ViT 模型，以便作为视觉编码器用于图像分类和多模态模型。

#### 验收标准

1. THE ViT_Model SHALL 包含 Patch_Embedding 和 N 个 Transformer Encoder Layer（复用 bert-gpt-from-scratch 的 Encoder Layer）
2. THE ViT_Model SHALL 在最后一层 Encoder 输出后应用 Layer Normalization
3. THE ViT_Model SHALL 提供分类头，将 [CLS] token 的隐藏状态映射到类别数量的 logits
4. THE ViT_Model SHALL 支持输出所有 patch 的隐藏状态，用于下游多模态任务
5. THE ViT_Model SHALL 接受 image_size、patch_size、d_model、num_heads、num_layers、d_ff、num_classes、dropout_rate 作为配置参数
6. WHEN 输入图像尺寸为 (batch, 3, H, W) 时，THE ViT_Model SHALL 输出 hidden_states 形状为 (batch, num_patches+1, d_model)

### 需求 3：DETR 模型（Detection Transformer）

**用户故事：** 作为开发者，我希望实现 DETR 目标检测模型，以便理解 Transformer 在目标检测任务中的应用。

#### 验收标准

1. THE DETR_Model SHALL 包含 CNN backbone（ResNet）用于提取图像特征
2. THE DETR_Model SHALL 包含 Transformer Encoder 处理 CNN 特征
3. THE DETR_Model SHALL 包含 Transformer Decoder，使用 Object_Query 作为查询
4. THE DETR_Model SHALL 实现 Object_Query，包含 num_queries 个可学习的查询向量（默认 100 个）
5. THE DETR_Model SHALL 提供分类头，将每个 Object_Query 的输出映射到类别 logits（包含背景类）
6. THE DETR_Model SHALL 提供边界框回归头，将每个 Object_Query 的输出映射到 4 维边界框坐标（cx, cy, w, h）
7. THE DETR_Model SHALL 实现 Hungarian_Matching，使用匈牙利算法将预测与真实目标进行最优匹配
8. THE DETR_Model SHALL 实现 Bipartite_Matching_Loss，包含分类损失、L1 边界框损失和 GIoU 损失
9. THE DETR_Model SHALL 接受 num_classes、num_queries、d_model、num_heads、num_encoder_layers、num_decoder_layers、d_ff 作为配置参数

### 需求 4：CLIP 模型（图文对比学习）

**用户故事：** 作为开发者，我希望实现 CLIP 模型，以便学习图像和文本的对齐表示。

#### 验收标准

1. THE CLIP_Model SHALL 包含 Vision_Encoder（基于 ViT_Model）用于编码图像
2. THE CLIP_Model SHALL 包含 Text_Encoder（复用 bert-gpt-from-scratch 的 Transformer Encoder）用于编码文本
3. THE CLIP_Model SHALL 实现 Visual_Projection，将视觉特征投影到共享嵌入空间
4. THE CLIP_Model SHALL 实现 Text_Projection，将文本特征投影到共享嵌入空间
5. THE CLIP_Model SHALL 对投影后的特征进行 L2 归一化
6. THE CLIP_Model SHALL 实现 Contrastive_Loss（InfoNCE Loss），使用可学习的温度参数
7. WHEN 输入 batch_size 个图文对时，THE CLIP_Model SHALL 计算 batch_size x batch_size 的相似度矩阵
8. THE CLIP_Model SHALL 支持零样本图像分类：给定图像和文本标签列表，返回最匹配的标签
9. THE CLIP_Model SHALL 接受 vision_config、text_config、projection_dim、temperature 作为配置参数

### 需求 5：BLIP 模型（图文理解与生成）

**用户故事：** 作为开发者，我希望实现 BLIP 模型，以便同时支持图文理解和图文生成任务。

#### 验收标准

1. THE BLIP_Model SHALL 包含 Vision_Encoder（基于 ViT_Model）用于编码图像
2. THE BLIP_Model SHALL 包含 Text_Encoder 用于图文匹配任务（ITC: Image-Text Contrastive）
3. THE BLIP_Model SHALL 包含 Text_Decoder 用于图文生成任务（ITG: Image-grounded Text Generation）
4. THE BLIP_Model SHALL 实现 Image-Text Matching（ITM）任务头，判断图文是否匹配
5. THE BLIP_Model SHALL 在 Text_Encoder 和 Text_Decoder 中使用 Cross_Attention 融合视觉特征
6. THE BLIP_Model SHALL 实现三个预训练损失：ITC Loss、ITM Loss、ITG Loss（Language Modeling Loss）
7. THE BLIP_Model SHALL 支持图像描述生成（Image Captioning）推理
8. THE BLIP_Model SHALL 支持视觉问答（VQA）推理
9. THE BLIP_Model SHALL 接受 vision_config、text_config、projection_dim 作为配置参数

### 需求 6：BLIP-2 模型（Q-Former 架构）

**用户故事：** 作为开发者，我希望实现 BLIP-2 模型，以便理解如何高效地桥接冻结的视觉编码器和大语言模型。

#### 验收标准

1. THE BLIP2_Model SHALL 包含冻结的 Vision_Encoder（基于 ViT_Model）
2. THE BLIP2_Model SHALL 实现 Q_Former，包含 num_query_tokens 个可学习的查询向量（默认 32 个）
3. THE Q_Former SHALL 包含自注意力层处理查询向量之间的交互
4. THE Q_Former SHALL 包含交叉注意力层，使查询向量关注视觉特征
5. THE Q_Former SHALL 输出固定数量的视觉 token，形状为 (batch, num_query_tokens, d_model)
6. THE BLIP2_Model SHALL 实现 Visual_Projection，将 Q_Former 输出投影到 LLM 的嵌入空间
7. THE BLIP2_Model SHALL 支持连接冻结的 LLM（如 LLaMA_Model）进行视觉语言生成
8. THE BLIP2_Model SHALL 实现两阶段训练：第一阶段训练 Q_Former（ITC、ITM、ITG），第二阶段训练 Visual_Projection
9. THE BLIP2_Model SHALL 接受 vision_config、qformer_config、llm_config、num_query_tokens 作为配置参数

### 需求 7：LLaMA 模型（大语言模型基座）

**用户故事：** 作为开发者，我希望实现 LLaMA 模型架构，以便作为多模态大语言模型的语言基座。

#### 验收标准

1. THE LLaMA_Model SHALL 实现 RMSNorm 替代 Layer Normalization
2. THE LLaMA_Model SHALL 实现 RoPE（Rotary Position Embedding）旋转位置编码
3. THE LLaMA_Model SHALL 实现 SwiGLU 激活函数替代 GELU
4. THE LLaMA_Model SHALL 实现 Pre-Norm 架构（在注意力和 FFN 之前应用归一化）
5. THE LLaMA_Model SHALL 支持 GQA（Grouped Query Attention），允许 num_kv_heads < num_heads
6. THE LLaMA_Model SHALL 堆叠 N 个 Decoder Layer
7. THE LLaMA_Model SHALL 提供语言模型头（LM Head），支持权重绑定
8. THE LLaMA_Model SHALL 支持 KV Cache 加速自回归生成
9. THE LLaMA_Model SHALL 接受 vocab_size、d_model、num_heads、num_kv_heads、num_layers、d_ff、max_seq_len、rope_theta 作为配置参数

### 需求 8：Flamingo 模型（视觉-语言交错输入）

**用户故事：** 作为开发者，我希望实现 Flamingo 模型，以便支持多图像和文本交错输入的多模态理解。

#### 验收标准

1. THE Flamingo_Model SHALL 包含冻结的 Vision_Encoder（基于 ViT_Model）
2. THE Flamingo_Model SHALL 实现 Perceiver_Resampler，将可变数量的视觉 token 压缩为固定数量（默认 64 个）
3. THE Perceiver_Resampler SHALL 使用可学习的 latent 向量作为查询，通过交叉注意力聚合视觉特征
4. THE Flamingo_Model SHALL 包含冻结的 LLM（如 LLaMA_Model）作为语言基座
5. THE Flamingo_Model SHALL 在 LLM 的每个 Decoder Layer 中插入 Gated_Cross_Attention 层
6. THE Gated_Cross_Attention SHALL 包含可学习的门控参数 tanh(alpha)，初始化为 0
7. THE Flamingo_Model SHALL 支持多图像输入，每个图像的视觉 token 与对应位置的文本 token 进行交叉注意力
8. THE Flamingo_Model SHALL 仅训练 Perceiver_Resampler 和 Gated_Cross_Attention 层的参数
9. THE Flamingo_Model SHALL 接受 vision_config、llm_config、num_latents、cross_attention_freq 作为配置参数

### 需求 9：LLaVA 模型（视觉指令微调）

**用户故事：** 作为开发者，我希望实现 LLaVA 模型，以便通过视觉指令微调使大语言模型具备视觉理解能力。

#### 验收标准

1. THE LLaVA_Model SHALL 包含 Vision_Encoder（基于 ViT_Model，可选冻结或微调）
2. THE LLaVA_Model SHALL 实现 Visual_Projection（MLP），将视觉特征映射到 LLM 的嵌入空间
3. THE LLaVA_Model SHALL 包含 LLM（如 LLaMA_Model）作为语言基座
4. THE LLaVA_Model SHALL 将视觉 token 直接拼接到文本 token 序列中
5. THE LLaVA_Model SHALL 支持特殊 token <image> 标记图像插入位置
6. THE LLaVA_Model SHALL 实现两阶段训练：第一阶段仅训练 Visual_Projection，第二阶段全参数微调
7. WHEN 进行指令微调时，THE LLaVA_Model SHALL 仅对 response 部分计算损失
8. THE LLaVA_Model SHALL 支持多轮对话格式的输入
9. THE LLaVA_Model SHALL 接受 vision_config、llm_config、projection_type 作为配置参数

### 需求 10：Qwen 模型（通义千问系列）

**用户故事：** 作为开发者，我希望实现 Qwen 模型架构，以便理解国产大语言模型的设计特点。

#### 验收标准

1. THE Qwen_Model SHALL 实现 RMSNorm 归一化层
2. THE Qwen_Model SHALL 实现 RoPE 旋转位置编码，支持 NTK-aware 插值扩展上下文长度
3. THE Qwen_Model SHALL 实现 SwiGLU 激活函数
4. THE Qwen_Model SHALL 支持 GQA（Grouped Query Attention）
5. THE Qwen_Model SHALL 实现 Pre-Norm 架构
6. THE Qwen_Model SHALL 支持 Sliding Window Attention（可选）
7. THE Qwen_Model SHALL 支持 KV Cache 加速自回归生成
8. THE Qwen_Model SHALL 提供语言模型头（LM Head），支持权重绑定
9. THE Qwen_Model SHALL 接受 vocab_size、d_model、num_heads、num_kv_heads、num_layers、d_ff、max_seq_len、rope_theta、use_sliding_window、sliding_window_size 作为配置参数

### 需求 11：对比学习训练

**用户故事：** 作为开发者，我希望实现对比学习训练流程，以便训练 CLIP 等图文对齐模型。

#### 验收标准

1. THE Trainer SHALL 实现 InfoNCE Loss，计算图像到文本和文本到图像的双向对比损失
2. THE Trainer SHALL 支持可学习的温度参数，初始化为 0.07
3. THE Trainer SHALL 实现 hard negative mining（可选），选择困难负样本
4. THE Trainer SHALL 支持大 batch size 训练，使用梯度累积
5. THE Trainer SHALL 支持分布式训练时跨 GPU 收集负样本
6. THE Trainer SHALL 支持配置 batch_size、learning_rate、num_epochs、warmup_steps 等训练超参数
7. THE Trainer SHALL 在训练过程中定期记录 loss 值、准确率并保存模型检查点

### 需求 12：多模态预训练

**用户故事：** 作为开发者，我希望实现多模态预训练流程，以便训练 BLIP 等多任务多模态模型。

#### 验收标准

1. THE Trainer SHALL 实现 Image-Text Contrastive（ITC）损失
2. THE Trainer SHALL 实现 Image-Text Matching（ITM）损失，使用 hard negative mining 构造负样本
3. THE Trainer SHALL 实现 Image-grounded Text Generation（ITG）损失，即 Language Modeling Loss
4. THE Trainer SHALL 支持多任务联合训练，加权组合多个损失
5. THE Trainer SHALL 支持冻结部分模型参数（如 Vision_Encoder）
6. THE Trainer SHALL 支持配置各损失的权重系数
7. THE Trainer SHALL 在训练过程中定期记录各项 loss 值并保存模型检查点

### 需求 13：视觉指令微调训练

**用户故事：** 作为开发者，我希望实现视觉指令微调训练流程，以便训练 LLaVA 等视觉对话模型。

#### 验收标准

1. THE Trainer SHALL 实现视觉指令数据预处理：解析 instruction-image-response 格式的数据
2. THE Trainer SHALL 支持多轮对话数据格式
3. WHEN 进行指令微调时，THE Trainer SHALL 仅对 response 部分计算 Language Modeling Loss
4. THE Trainer SHALL 支持两阶段训练：预训练阶段和指令微调阶段
5. THE Trainer SHALL 支持 LoRA 等参数高效微调方法（可选）
6. THE Trainer SHALL 支持加载预训练的 Vision_Encoder 和 LLM 检查点
7. THE Trainer SHALL 支持配置 batch_size、learning_rate、num_epochs、warmup_ratio 等训练超参数
8. THE Trainer SHALL 在训练过程中定期记录 loss 值并保存模型检查点

### 需求 14：目标检测训练

**用户故事：** 作为开发者，我希望实现目标检测训练流程，以便训练 DETR 模型。

#### 验收标准

1. THE Trainer SHALL 实现 Hungarian_Matching，使用 scipy.optimize.linear_sum_assignment 进行最优匹配
2. THE Trainer SHALL 计算匹配代价矩阵，包含分类代价、L1 边界框代价和 GIoU 代价
3. THE Trainer SHALL 实现分类损失：对匹配的预测计算交叉熵损失，对未匹配的预测计算背景类损失
4. THE Trainer SHALL 实现边界框损失：L1 损失 + GIoU 损失
5. THE Trainer SHALL 支持配置各损失的权重系数（默认：分类 1.0、L1 5.0、GIoU 2.0）
6. THE Trainer SHALL 支持数据增强：随机裁剪、随机缩放、水平翻转
7. THE Trainer SHALL 在训练过程中定期记录 loss 值、mAP 指标并保存模型检查点

### 需求 15：多模态推理

**用户故事：** 作为开发者，我希望实现多模态推理代码，以便使用训练好的模型进行各种视觉语言任务。

#### 验收标准

1. THE Inference_Engine SHALL 支持加载 ViT_Model、CLIP_Model、BLIP_Model、BLIP2_Model、Flamingo_Model、LLaVA_Model、Qwen_Model 的检查点文件
2. THE Inference_Engine SHALL 实现 ViT_Model 的图像分类推理
3. THE Inference_Engine SHALL 实现 DETR_Model 的目标检测推理，输出边界框和类别
4. THE Inference_Engine SHALL 实现 CLIP_Model 的零样本图像分类推理
5. THE Inference_Engine SHALL 实现 CLIP_Model 的图文检索推理
6. THE Inference_Engine SHALL 实现 BLIP_Model 的图像描述生成推理
7. THE Inference_Engine SHALL 实现 BLIP_Model 的视觉问答推理
8. THE Inference_Engine SHALL 实现 LLaVA_Model 的视觉对话推理
9. WHEN 进行文本生成时，THE Inference_Engine SHALL 支持 greedy decoding、top-k sampling、top-p sampling 三种解码策略
10. WHEN 进行文本生成时，THE Inference_Engine SHALL 支持 temperature 参数控制生成随机性
11. THE Inference_Engine SHALL 支持 KV Cache 加速自回归生成
12. THE Inference_Engine SHALL 支持配置 max_gen_len 参数限制生成文本的最大长度

### 需求 16：图像预处理模块

**用户故事：** 作为开发者，我希望有统一的图像预处理接口，以便对输入图像进行标准化处理。

#### 验收标准

1. THE Vision_Encoder SHALL 提供 Image_Processor 模块
2. THE Image_Processor SHALL 实现图像缩放到指定尺寸（默认 224x224）
3. THE Image_Processor SHALL 实现图像归一化，使用 ImageNet 均值和标准差
4. THE Image_Processor SHALL 支持从文件路径、PIL Image、numpy array、torch Tensor 加载图像
5. THE Image_Processor SHALL 输出形状为 (batch, 3, H, W) 的归一化张量
6. THE Image_Processor SHALL 支持配置 image_size、mean、std 参数

### 需求 17：组件复用与集成

**用户故事：** 作为开发者，我希望本项目能够复用 bert-gpt-from-scratch 中的 Transformer 核心组件，以便保持代码一致性并减少重复实现。

#### 验收标准

1. THE Vision_Encoder SHALL 复用 bert-gpt-from-scratch 的 Multi_Head_Attention 组件
2. THE Vision_Encoder SHALL 复用 bert-gpt-from-scratch 的 Feed_Forward_Network 组件
3. THE Vision_Encoder SHALL 复用 bert-gpt-from-scratch 的 Encoder Layer 组件
4. THE CLIP_Model 的 Text_Encoder SHALL 复用 bert-gpt-from-scratch 的 Transformer Encoder
5. THE LLaMA_Model SHALL 扩展 bert-gpt-from-scratch 的 Decoder Layer，添加 RMSNorm、RoPE、SwiGLU、GQA 支持
6. THE Trainer SHALL 复用 bert-gpt-from-scratch 的训练工具函数（学习率调度、梯度裁剪、检查点保存）
7. THE Inference_Engine SHALL 复用 bert-gpt-from-scratch 的解码策略（greedy、top-k、top-p）
