# 实现计划：BERT 和 GPT 从零实现

## 概述

本实现计划将设计文档中的技术方案转化为可执行的编码任务。采用自底向上的方式，先实现核心组件，再组装模型，最后实现训练和推理流程。所有代码使用 Python + PyTorch 实现。

## 任务列表

- [ ] 1. 项目结构与配置模块
  - [x] 1.1 创建项目目录结构
    - 创建 `bert_gpt_from_scratch/` 主目录及所有子目录
    - 创建各模块的 `__init__.py` 文件
    - _需求: 整体架构_

  - [x] 1.2 实现配置类 (config.py)
    - 实现 `TransformerConfig` 基础配置 dataclass
    - 实现 `BERTConfig` 继承配置，添加 `num_segments` 参数
    - 实现 `GPTConfig` 继承配置，添加 `tie_weights` 参数
    - 实现 `TrainingConfig` 和 `SFTConfig` 训练配置
    - _需求: 6.7, 7.7_

- [x] 2. Tokenizer 模块
  - [x] 2.1 实现 SimpleTokenizer 类
    - 实现特殊 token 定义：[PAD], [UNK], [CLS], [SEP], [MASK], [BOS], [EOS]
    - 实现 `encode` 方法：文本转 token ID 序列
    - 实现 `decode` 方法：token ID 序列转文本
    - 实现词表属性：`vocab_size`, `pad_token_id`, `mask_token_id` 等
    - _需求: 12.1, 12.2, 12.3, 12.5_

  - [x] 2.2 编写 Tokenizer 属性测试
    - **属性 1: Round-trip 一致性** - `decode(encode(text))` 应产生与原始文本等价的结果
    - **验证需求: 12.4**

- [x] 3. 检查点 - 确保 Tokenizer 测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 4. 核心注意力组件
  - [x] 4.1 实现 Scaled Dot-Product Attention
    - 实现 `scaled_dot_product_attention` 函数
    - 计算公式：`softmax(QK^T / sqrt(d_k))V`
    - 支持可选的 attention_mask 参数
    - 支持可选的 dropout 参数
    - _需求: 1.1, 1.3_

  - [x] 4.2 实现 Multi-Head Attention 类
    - 实现 Q、K、V 线性投影层
    - 实现多头并行计算和拼接
    - 实现输出投影层
    - 支持 attention_mask 和 causal_mask
    - _需求: 1.2, 1.3, 1.4, 1.5, 1.6_

  - [x] 4.3 编写 Multi-Head Attention 单元测试
    - 测试输出形状正确性
    - 测试 causal mask 阻止未来位置信息
    - 测试 padding mask 正确应用
    - _需求: 1.3, 1.4, 1.6_

- [x] 5. 位置编码组件
  - [x] 5.1 实现 SinusoidalPositionEncoding 类
    - 实现正弦余弦位置编码公式
    - 支持 `d_model` 和 `max_seq_len` 参数
    - 位置编码与输入相加
    - _需求: 2.1, 2.3, 2.4_

  - [x] 5.2 实现 LearnablePositionEmbedding 类
    - 实现可学习的位置嵌入矩阵
    - 支持 `d_model` 和 `max_seq_len` 参数
    - 位置嵌入与输入相加
    - _需求: 2.2, 2.3, 2.4_

  - [x] 5.3 编写位置编码单元测试
    - 测试输出形状正确性
    - 测试位置编码值范围
    - _需求: 2.3, 2.4_

- [x] 6. 前馈网络组件
  - [x] 6.1 实现 FeedForwardNetwork 类
    - 实现两层线性变换：`d_model -> d_ff -> d_model`
    - 使用 GELU 激活函数
    - 在两层之间应用 Dropout
    - _需求: 3.1, 3.2, 3.3, 3.4_

  - [x] 6.2 编写 FeedForwardNetwork 单元测试
    - 测试输出形状与输入形状一致
    - 测试 dropout 在训练/推理模式下的行为
    - _需求: 3.4_

- [x] 7. 检查点 - 确保核心组件测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 8. Transformer 层
  - [x] 8.1 实现 EncoderLayer 类
    - 组合 Multi-Head Attention 和 FeedForwardNetwork
    - 实现残差连接
    - 实现 Layer Normalization
    - 实现 Dropout
    - 支持 padding_mask 参数
    - _需求: 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 8.2 实现 DecoderLayer 类
    - 组合带 Causal Mask 的 Multi-Head Attention 和 FeedForwardNetwork
    - 自动生成并应用 Causal Mask
    - 实现残差连接
    - 实现 Layer Normalization
    - 实现 Dropout
    - _需求: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 8.3 编写 Transformer 层单元测试
    - 测试 EncoderLayer 输出形状
    - 测试 DecoderLayer 自动应用 causal mask
    - 测试残差连接和 LayerNorm 正确应用
    - _需求: 4.1, 5.1, 5.2_

- [x] 9. BERT 模型
  - [x] 9.1 实现 BERTModel 类
    - 实现 Token Embedding 层
    - 实现 Segment Embedding 层
    - 实现 Learnable Position Embedding 层
    - 堆叠 N 个 EncoderLayer
    - 实现嵌入层后的 LayerNorm 和 Dropout
    - _需求: 6.1, 6.2, 6.3, 6.4, 6.7_

  - [x] 9.2 实现 BERT 预测头
    - 实现 MLM Head：隐藏状态 -> 词表 logits
    - 实现 NSP Head：[CLS] 隐藏状态 -> 二分类 logits
    - _需求: 6.5, 6.6_

  - [x] 9.3 编写 BERT 模型单元测试
    - 测试前向传播输出形状
    - 测试 MLM logits 形状为 (batch, seq_len, vocab_size)
    - 测试 NSP logits 形状为 (batch, 2)
    - _需求: 6.4, 6.5, 6.6_

- [x] 10. GPT 模型
  - [x] 10.1 实现 GPTModel 类
    - 实现 Token Embedding 层
    - 实现 Learnable Position Embedding 层
    - 堆叠 N 个 DecoderLayer
    - 实现嵌入层后的 LayerNorm 和 Dropout
    - _需求: 7.1, 7.2, 7.3, 7.4, 7.7_

  - [x] 10.2 实现 GPT 语言模型头
    - 实现 LM Head：隐藏状态 -> 词表 logits
    - 实现权重绑定（Weight Tying）选项
    - _需求: 7.5, 7.6_

  - [x] 10.3 编写 GPT 模型单元测试
    - 测试前向传播输出形状
    - 测试 logits 形状为 (batch, seq_len, vocab_size)
    - 测试权重绑定正确性
    - _需求: 7.4, 7.5, 7.6_

- [x] 11. 检查点 - 确保模型测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 12. BERT 预训练
  - [x] 12.1 实现 MLM 数据预处理
    - 随机选择 15% 的 token 进行掩码
    - 80% 概率替换为 [MASK]
    - 10% 概率替换为随机 token
    - 10% 概率保持不变
    - _需求: 8.1, 8.2_

  - [x] 12.2 实现 NSP 数据预处理
    - 50% 概率选择真实下一句
    - 50% 概率选择随机句子
    - 生成 segment_ids 和 nsp_labels
    - _需求: 8.3_

  - [x] 12.3 实现 BERTPreTrainer 类
    - 实现 MLM 损失计算（仅对掩码位置）
    - 实现 NSP 损失计算
    - 实现总损失 = MLM 损失 + NSP 损失
    - 实现训练循环、日志记录和检查点保存
    - _需求: 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 12.4 编写 BERT 预训练单元测试
    - 测试 MLM 数据预处理的掩码比例
    - 测试 NSP 数据预处理的标签分布
    - 测试损失计算正确性
    - _需求: 8.1, 8.2, 8.3, 8.4, 8.5_

- [x] 13. GPT 预训练
  - [x] 13.1 实现 NWP 数据预处理
    - 将输入序列右移一位作为目标序列
    - 处理 padding token（设置 label 为 -100）
    - _需求: 9.1, 9.3_

  - [x] 13.2 实现 GPTPreTrainer 类
    - 实现 NWP 损失计算（忽略 padding 位置）
    - 实现训练循环、日志记录和检查点保存
    - _需求: 9.2, 9.3, 9.4, 9.5_

  - [x] 13.3 编写 GPT 预训练单元测试
    - 测试 NWP 数据预处理的序列偏移
    - 测试损失计算忽略 padding
    - _需求: 9.1, 9.2, 9.3_

- [x] 14. 检查点 - 确保预训练测试通过
  - 确保所有测试通过，如有问题请询问用户。

- [x] 15. 监督微调 (SFT)
  - [x] 15.1 实现 SFTTrainer 基础功能
    - 实现预训练检查点加载
    - 实现层冻结功能
    - _需求: 10.3, 10.4, 10.5_

  - [x] 15.2 实现 BERT 文本分类微调
    - 在 [CLS] 隐藏状态上添加分类头
    - 实现分类损失计算和训练循环
    - _需求: 10.1_

  - [x] 15.3 实现 GPT 指令微调
    - 支持 instruction-response 格式数据
    - 仅对 response 部分计算损失
    - _需求: 10.2, 10.6_

  - [x] 15.4 编写 SFT 单元测试
    - 测试检查点加载正确性
    - 测试层冻结功能
    - 测试指令微调的损失掩码
    - _需求: 10.3, 10.4, 10.6_

- [x] 16. 推理引擎
  - [x] 16.1 实现 InferenceEngine 基础功能
    - 实现模型检查点加载
    - 支持 BERT 和 GPT 模型类型
    - _需求: 11.1_

  - [x] 16.2 实现 BERT 推理方法
    - 实现 MLM 填空推理：预测 [MASK] 位置的 token
    - 实现文本分类推理：输出分类结果
    - _需求: 11.2, 11.3_

  - [x] 16.3 实现 GPT 文本生成
    - 实现自回归生成循环
    - 实现 greedy decoding 策略
    - 实现 top-k sampling 策略
    - 实现 top-p (nucleus) sampling 策略
    - 支持 temperature 参数
    - 支持 max_gen_len 参数
    - 实现 EOS token 停止条件
    - _需求: 11.4, 11.5, 11.6, 11.7, 11.8_

  - [x] 16.4 编写推理引擎单元测试
    - 测试 BERT MLM 填空输出格式
    - 测试 GPT 生成的停止条件
    - 测试不同解码策略的行为
    - _需求: 11.2, 11.4, 11.5, 11.8_

- [x] 17. 最终检查点 - 确保所有测试通过
  - 确保所有测试通过，如有问题请询问用户。

## 备注

- 标记 `*` 的任务为可选任务，可跳过以加快 MVP 开发
- 每个任务都引用了具体的需求条款以确保可追溯性
- 检查点任务用于阶段性验证，确保增量开发的正确性
- 属性测试验证核心正确性属性，单元测试验证具体示例和边界情况
