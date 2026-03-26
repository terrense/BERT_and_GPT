# 🚀 从零手搓 Transformer 全家桶

> 纯 PyTorch 实现，不调包，每一行代码都是自己写的（好吧，有些是 debug 了一晚上才写对的 😭）

## 🤔 这是啥？

这个项目是我学习 Transformer 架构时的练手项目。网上教程看了一堆，论文也啃了不少，但总感觉隔靴搔痒。于是决定：**不如自己从零撸一遍！**

项目包含两大部分：

### 📦 bert_gpt_from_scratch
经典 Transformer 架构实现，包括：
- 🔤 BERT（双向编码器）
- 📝 GPT（自回归解码器）
- ⚙️ 核心组件：Multi-Head Attention、Position Encoding、Feed Forward Network...
- 🎓 预训练 & SFT 训练流程
- 🎯 推理引擎（支持 greedy、sampling、top-k、top-p）

### 🖼️ multimodal_models_from_scratch  
多模态大模型实现，这部分是真的肝 😵：
- 👁️ Vision Transformer (ViT)
- 🦙 LLaMA / Qwen（带 RoPE、GQA、SwiGLU 这些花活）
- 🔗 CLIP（对比学习 yyds）
- 🎨 BLIP / BLIP-2（图文理解）
- 🦩 Flamingo（多图推理）
- 🗣️ LLaVA（视觉对话）
- 🎯 DETR（目标检测，用 Transformer 做检测是真的优雅）

## 🏗️ 项目结构

```
.
├── bert_gpt_from_scratch/          # 经典 Transformer
│   ├── core/                       # 核心组件（attention、ffn、layers）
│   ├── models/                     # BERT、GPT 模型
│   ├── tokenizer/                  # 简易分词器
│   ├── training/                   # 预训练 & SFT
│   ├── inference/                  # 推理引擎
│   └── tests/                      # 单元测试
│
├── multimodal_models_from_scratch/ # 多模态模型
│   ├── vision/                     # ViT、图像处理
│   ├── llm/                        # LLaMA、Qwen
│   ├── multimodal/                 # CLIP、BLIP、LLaVA...
│   ├── detection/                  # DETR 目标检测
│   ├── training/                   # 各种训练器
│   ├── inference/                  # 多模态推理
│   └── tests/                      # 测试用例
│
└── requirements.txt
```

## 🛠️ 快速开始

```bash
# 克隆项目
git clone https://github.com/你的用户名/encoder_decoder_only_LLM.git
cd encoder_decoder_only_LLM

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或者 .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 🧪 跑测试

```bash
# 跑全部测试
pytest

# 只跑某个模块的测试
pytest bert_gpt_from_scratch/tests/ -v
pytest multimodal_models_from_scratch/tests/ -v

# 看覆盖率（如果你装了 pytest-cov）
pytest --cov=. --cov-report=html
```

## 📚 学习路线建议

如果你也想从零学习，建议按这个顺序来：

1. **先看 `bert_gpt_from_scratch/core/`** - 理解 Attention 机制是一切的基础
2. **再看 BERT 和 GPT** - 理解 Encoder-only 和 Decoder-only 的区别
3. **然后是 ViT** - 把图像也变成 token 序列，思路很妙
4. **接着是 CLIP** - 对比学习把图文拉到同一个空间
5. **最后是 BLIP/LLaVA** - 真正的多模态理解

每个模块都有对应的测试文件，可以当作使用示例来看 👀

## 🤝 一些碎碎念

写这个项目的过程中踩了无数坑：

- Attention 的 mask 搞反了，debug 了一整天 🤦
- RoPE 的实现一开始完全理解错了，后来对着论文一行行推才搞明白
- BLIP-2 的两阶段训练逻辑绕了好久
- 匈牙利匹配算法...数学太美了但代码太丑了 😂

如果这个项目对你有帮助，欢迎 star ⭐️

有问题可以开 issue，虽然我可能回复得比较慢（社畜没办法 😅）

## 📖 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 开山之作
- [BERT](https://arxiv.org/abs/1810.04805) - 双向编码器
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) - 自回归语言模型
- [ViT](https://arxiv.org/abs/2010.11929) - 图像也能用 Transformer
- [CLIP](https://arxiv.org/abs/2103.00020) - 对比学习连接图文
- [BLIP](https://arxiv.org/abs/2201.12086) / [BLIP-2](https://arxiv.org/abs/2301.12597) - 图文预训练
- [LLaMA](https://arxiv.org/abs/2302.13971) - Meta 的开源 LLM
- [Flamingo](https://arxiv.org/abs/2204.14198) - 多图推理
- [LLaVA](https://arxiv.org/abs/2304.08485) - 视觉指令微调
- [DETR](https://arxiv.org/abs/2005.12872) - 端到端目标检测

## 📜 License

MIT License - 随便用，能帮到你就好 🎉

---

*如果你也在学习 Transformer，希望这个项目能帮你少走一些弯路～*

*Happy Coding! 🎮*
