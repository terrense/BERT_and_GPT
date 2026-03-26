# 需求文档

## 简介

本项目旨在从零实现 BERT（encoder-only）和 GPT（decoder-only）两大 LLM 范式的完整代码。涵盖 Transformer 核心组件的手动实现、两种架构的预训练任务、训练流程、监督微调（SFT）以及推理代码。项目使用 Python + PyTorch 构建，不依赖 HuggingFace Transformers 等高层封装库的模型实现。

## 术语表

- **Transformer_Engine**: 本项目实现的 Transformer 核心计算引擎，包含注意力机制、前馈网络等基础组件
- **BERT_Model**: 本项目实现的 encoder-only 架构模型，用于双向语言理解
- **GPT_Model**: 本项目实现的 decoder-only 架构模型，用于自回归文本生成
- **Trainer**: 本项目的训练控制模块，负责预训练和微调的训练循环
- **Inference_Engine**: 本项目的推理模块，负责模型加载和文本生成
- **MLM**: Masked Language Model，掩码语言模型，BERT 的核心预训练任务
- **NSP**: Next Sentence Prediction，下一句预测，BERT 的辅助预训练任务
- **NWP**: Next Word Prediction，下一词预测（即 Causal LM），GPT 的预训练任务
- **SFT**: Supervised Fine-Tuning，监督微调，使用标注数据对预训练模型进行任务适配
- **Multi_Head_Attention**: 多头注意力机制，Transformer 的核心注意力计算组件
- **Position_Encoding**: 位置编码，为输入序列注入位置信息的组件
- **Feed_Forward_Network**: 前馈神经网络，Transformer 中每层的逐位置全连接网络
- **Causal_Mask**: 因果掩码，用于 decoder-only 架构中防止当前位置关注未来位置的掩码矩阵

## 需求

### 需求 1：Multi-Head Attention 组件

**用户故事：** 作为开发者，我希望从零实现 Multi-Head Attention 机制，以便理解 Transformer 的核心注意力计算原理。

#### 验收标准

1. THE Transformer_Engine SHALL 实现 Scaled Dot-Product Attention，计算公式为 softmax(QK^T / sqrt(d_k))V
2. THE Transformer_Engine SHALL 实现 Multi_Head_Attention，将输入通过线性投影分为多个注意力头并行计算后拼接
3. WHEN 提供 attention_mask 参数时，THE Multi_Head_Attention SHALL 在 softmax 之前将被掩码位置的注意力分数设为负无穷
4. WHEN 提供 Causal_Mask 时，THE Multi_Head_Attention SHALL 阻止每个位置关注其后续位置的信息
5. THE Multi_Head_Attention SHALL 接受 batch_size、seq_len、d_model、num_heads 作为配置参数
6. THE Multi_Head_Attention SHALL 输出形状为 (batch_size, seq_len, d_model) 的张量

### 需求 2：Position Encoding 组件

**用户故事：** 作为开发者，我希望实现位置编码机制，以便模型能够感知输入序列中 token 的位置信息。

#### 验收标准

1. THE Transformer_Engine SHALL 实现正弦余弦位置编码（Sinusoidal Position Encoding），用于 Transformer 基础位置表示
2. THE Transformer_Engine SHALL 实现可学习的位置嵌入（Learnable Position Embedding），用于 BERT_Model 和 GPT_Model
3. WHEN 输入序列长度不超过 max_seq_len 时，THE Position_Encoding SHALL 为每个位置生成 d_model 维的位置向量
4. THE Position_Encoding SHALL 将位置编码与 token 嵌入相加后输出

### 需求 3：Feed Forward Network 组件

**用户故事：** 作为开发者，我希望实现 Transformer 的前馈网络层，以便完成每个 Transformer 层中的非线性变换。

#### 验收标准

1. THE Feed_Forward_Network SHALL 包含两层线性变换，中间使用 GELU 激活函数
2. THE Feed_Forward_Network SHALL 接受 d_model 和 d_ff 作为配置参数，其中 d_ff 为中间层维度
3. THE Feed_Forward_Network SHALL 在两层线性变换之间应用 Dropout
4. THE Feed_Forward_Network SHALL 输出形状与输入形状一致，均为 (batch_size, seq_len, d_model)

### 需求 4：Transformer Encoder Layer

**用户故事：** 作为开发者，我希望实现完整的 Transformer Encoder 层，以便组装 BERT 模型。

#### 验收标准

1. THE Transformer_Engine SHALL 实现单个 Encoder Layer，包含 Multi_Head_Attention 子层和 Feed_Forward_Network 子层
2. THE Transformer_Engine SHALL 在每个子层后应用残差连接（Residual Connection）
3. THE Transformer_Engine SHALL 在每个子层后应用 Layer Normalization
4. THE Transformer_Engine SHALL 在每个子层的输入上应用 Dropout
5. WHEN 提供 padding_mask 时，THE Encoder Layer SHALL 将 padding 位置的注意力分数掩码

### 需求 5：Transformer Decoder Layer

**用户故事：** 作为开发者，我希望实现完整的 Transformer Decoder 层，以便组装 GPT 模型。

#### 验收标准

1. THE Transformer_Engine SHALL 实现单个 Decoder Layer，包含带 Causal_Mask 的 Multi_Head_Attention 子层和 Feed_Forward_Network 子层
2. THE Decoder Layer SHALL 自动生成并应用 Causal_Mask，确保自回归特性
3. THE Transformer_Engine SHALL 在每个子层后应用残差连接（Residual Connection）
4. THE Transformer_Engine SHALL 在每个子层后应用 Layer Normalization
5. THE Transformer_Engine SHALL 在每个子层的输入上应用 Dropout

### 需求 6：BERT 模型（Encoder-Only 架构）

**用户故事：** 作为开发者，我希望实现完整的 BERT 模型架构，以便进行双向语言理解的预训练。

#### 验收标准

1. THE BERT_Model SHALL 包含 Token Embedding、Segment Embedding 和 Learnable Position Embedding 三种嵌入层
2. THE BERT_Model SHALL 堆叠 N 个 Encoder Layer（N 为可配置参数）
3. THE BERT_Model SHALL 在嵌入层输出上应用 Layer Normalization 和 Dropout
4. THE BERT_Model SHALL 输出每个 token 位置的隐藏状态，形状为 (batch_size, seq_len, d_model)
5. THE BERT_Model SHALL 提供 MLM 预测头，将隐藏状态映射到词表大小的 logits
6. THE BERT_Model SHALL 提供 NSP 预测头，将 [CLS] token 的隐藏状态映射为二分类 logits
7. THE BERT_Model SHALL 接受 vocab_size、d_model、num_heads、num_layers、d_ff、max_seq_len、dropout_rate 作为配置参数

### 需求 7：GPT 模型（Decoder-Only 架构）

**用户故事：** 作为开发者，我希望实现完整的 GPT 模型架构，以便进行自回归文本生成的预训练。

#### 验收标准

1. THE GPT_Model SHALL 包含 Token Embedding 和 Learnable Position Embedding 两种嵌入层
2. THE GPT_Model SHALL 堆叠 N 个 Decoder Layer（N 为可配置参数）
3. THE GPT_Model SHALL 在嵌入层输出上应用 Layer Normalization 和 Dropout
4. THE GPT_Model SHALL 输出每个位置的隐藏状态，形状为 (batch_size, seq_len, d_model)
5. THE GPT_Model SHALL 提供语言模型头（LM Head），将隐藏状态映射到词表大小的 logits
6. THE GPT_Model SHALL 支持权重绑定（Weight Tying），LM Head 的权重与 Token Embedding 共享
7. THE GPT_Model SHALL 接受 vocab_size、d_model、num_heads、num_layers、d_ff、max_seq_len、dropout_rate 作为配置参数

### 需求 8：BERT 预训练（MLM + NSP）

**用户故事：** 作为开发者，我希望实现 BERT 的预训练流程，以便通过 MLM 和 NSP 任务训练 BERT 模型。

#### 验收标准

1. THE Trainer SHALL 实现 MLM 数据预处理：随机选择 15% 的 token 进行掩码处理
2. WHEN 一个 token 被选中进行掩码时，THE Trainer SHALL 以 80% 概率替换为 [MASK]、10% 概率替换为随机 token、10% 概率保持不变
3. THE Trainer SHALL 实现 NSP 数据预处理：50% 概率选择真实下一句，50% 概率选择随机句子
4. THE Trainer SHALL 计算 MLM 损失：仅对被掩码位置的 token 计算交叉熵损失
5. THE Trainer SHALL 计算 NSP 损失：对 [CLS] token 的二分类输出计算交叉熵损失
6. THE Trainer SHALL 将 MLM 损失和 NSP 损失相加作为总损失进行反向传播
7. THE Trainer SHALL 支持配置 batch_size、learning_rate、num_epochs、warmup_steps 等训练超参数
8. THE Trainer SHALL 在训练过程中定期记录 loss 值并保存模型检查点

### 需求 9：GPT 预训练（NWP / Causal LM）

**用户故事：** 作为开发者，我希望实现 GPT 的预训练流程，以便通过下一词预测任务训练 GPT 模型。

#### 验收标准

1. THE Trainer SHALL 实现 NWP 数据预处理：将输入序列右移一位作为目标序列
2. THE Trainer SHALL 计算 NWP 损失：对每个位置预测下一个 token 的交叉熵损失
3. WHEN 输入包含 padding token 时，THE Trainer SHALL 在损失计算中忽略 padding 位置
4. THE Trainer SHALL 支持配置 batch_size、learning_rate、num_epochs、warmup_steps 等训练超参数
5. THE Trainer SHALL 在训练过程中定期记录 loss 值并保存模型检查点

### 需求 10：监督微调（SFT）

**用户故事：** 作为开发者，我希望实现监督微调流程，以便将预训练模型适配到具体的下游任务。

#### 验收标准

1. THE Trainer SHALL 实现 BERT_Model 的文本分类微调：在 [CLS] 隐藏状态上添加分类头
2. THE Trainer SHALL 实现 GPT_Model 的指令微调：使用 instruction-response 格式的数据进行训练
3. WHEN 进行 SFT 时，THE Trainer SHALL 支持加载预训练模型的检查点权重
4. THE Trainer SHALL 支持冻结部分模型层（freeze layers）进行微调
5. THE Trainer SHALL 支持配置微调专用超参数：learning_rate、num_epochs、warmup_ratio
6. WHEN GPT_Model 进行指令微调时，THE Trainer SHALL 仅对 response 部分计算损失，忽略 instruction 部分

### 需求 11：推理代码

**用户故事：** 作为开发者，我希望实现推理代码，以便使用训练好的模型进行预测和文本生成。

#### 验收标准

1. THE Inference_Engine SHALL 支持加载 BERT_Model 和 GPT_Model 的检查点文件
2. THE Inference_Engine SHALL 实现 BERT_Model 的 MLM 填空推理：给定含 [MASK] 的句子，预测被掩码位置的 token
3. THE Inference_Engine SHALL 实现 BERT_Model 的文本分类推理：给定文本，输出分类结果
4. THE Inference_Engine SHALL 实现 GPT_Model 的自回归文本生成：给定 prompt，逐 token 生成后续文本
5. WHEN 进行 GPT_Model 文本生成时，THE Inference_Engine SHALL 支持 greedy decoding、top-k sampling 和 top-p（nucleus）sampling 三种解码策略
6. WHEN 进行 GPT_Model 文本生成时，THE Inference_Engine SHALL 支持 temperature 参数控制生成随机性
7. THE Inference_Engine SHALL 支持配置 max_gen_len 参数限制生成文本的最大长度
8. WHEN 生成的 token 为 EOS token 或达到 max_gen_len 时，THE Inference_Engine SHALL 停止生成

### 需求 12：Tokenizer 集成

**用户故事：** 作为开发者，我希望有一个统一的 Tokenizer 接口，以便对输入文本进行分词和编码。

#### 验收标准

1. THE Transformer_Engine SHALL 提供一个简易字符级或词级 Tokenizer，用于演示和测试
2. THE Tokenizer SHALL 实现 encode 方法：将文本字符串转换为 token ID 序列
3. THE Tokenizer SHALL 实现 decode 方法：将 token ID 序列转换为文本字符串
4. FOR ALL 合法文本输入，decode(encode(text)) SHALL 产生与原始文本等价的结果（round-trip 属性）
5. THE Tokenizer SHALL 支持特殊 token：[PAD]、[UNK]、[CLS]、[SEP]、[MASK]、[BOS]、[EOS]
