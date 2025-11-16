# LLaVA模型LoRA微调项目

> 人工智能原理与应用课程 · 第 18 组 · 王秀程

本项目实现了基于Qwen1.5-4B-Chat和CLIP-ViT-Large的LLaVA多模态大语言模型的构建与LoRA微调，支持图像-文本对话任务。

## 📋 项目简介

本项目主要包含以下内容：

- 将Qwen语言模型与CLIP视觉编码器组合构建LLaVA架构
- 使用LoRA（Low-Rank Adaptation）技术对模型进行参数高效微调
- 在LLaVA-CC3M-Pretrain-595K数据集上进行预训练
- 提供对比演示工具，可视化微调前后的效果差异

## 🏗️ 项目结构

```
.
├── merge_llava_model.py      # 合并Qwen和CLIP模型，构建LLaVA模型
├── lora_train.py             # LoRA微调训练脚本
├── Dataset.py                # 数据集加载和处理模块
├── merge_llava.py            # 模型合并工具（备用）
├── llava_contrast_demo.py    # Gradio可视化对比演示界面
├── llava_contrast_eval.py    # 命令行对比评估脚本
├── Qwen1.5-4B-Chat/          # Qwen语言模型目录（需从HuggingFace下载）
├── clip-vit-large-patch14-336/  # CLIP视觉模型目录（需从HuggingFace下载）
├── LLaVA-CC3M-Pretrain-595K/    # 训练数据集目录（需从HuggingFace下载）
├── show_model/               # 合并后的模型保存目录
│   └── model001/             # 合并后的LLaVA模型
└── output/                   # LoRA微调后的模型输出目录
```

## 🔧 环境要求

### 硬件要求

- GPU: 推荐NVIDIA GPU（显存≥24GB，如A6000）
- 内存: ≥32GB RAM

### 软件依赖

```bash
torch>=2.0.0
transformers>=4.37.0
peft>=0.6.0
pillow
pandas
gradio
```

安装依赖：

```bash
pip install torch transformers peft pillow pandas gradio
```

## 📦 数据准备

### 1. 下载预训练模型

**Qwen1.5-4B-Chat模型**：

```bash
# 使用HuggingFace CLI下载
huggingface-cli download Qwen/Qwen1.5-4B-Chat --local-dir Qwen1.5-4B-Chat
```

**CLIP-ViT-Large模型**：

```bash
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir clip-vit-large-patch14-336
```

### 2. 下载训练数据集

**LLaVA-CC3M-Pretrain-595K数据集**：

```bash
# 从HuggingFace下载数据集
# 数据集应包含chat.json和images_dl/目录
```

## 🚀 使用流程

### 步骤1: 合并模型

首先将Qwen语言模型和CLIP视觉编码器合并为LLaVA架构：

```bash
python merge_llava_model.py
```

该脚本会：

- 加载Qwen1.5-4B-Chat语言模型
- 加载CLIP-ViT-Large视觉编码器
- 创建LLaVA配置并合并两个模型
- 将合并后的模型保存到`show_model/model001/`

### 步骤2: LoRA微调训练

使用LoRA技术对合并后的模型进行微调：

```bash
python lora_train.py
```

**训练配置**：

- LoRA rank (r): 4
- LoRA alpha: 8
- LoRA dropout: 0.05
- 目标模块: `q_proj`, `v_proj`
- 学习率: 1e-3
- 训练步数: 800
- 批次大小: 6
- 优化器: AdamW
- 学习率调度: Cosine with warmup

训练后的LoRA权重将保存在`output/`目录。

### 步骤3: 模型对比评估

#### 方式1: Gradio可视化界面

启动交互式对比演示界面：

```bash
python llava_contrast_demo.py
```

界面功能：

- **Tab1**: 按数据集索引选择样本进行对比
- **Tab2**: 上传自定义图片和问题进行对比
- 支持调整生成参数（max_new_tokens, temperature, top_p等）
- 实时显示零样本基座模型和微调后模型的输出对比

访问地址：`http://localhost:7860`

#### 方式2: 命令行评估

使用命令行脚本进行单样本对比：

```bash
python llava_contrast_eval.py
```

修改脚本中的`SAMPLE_IDX`变量选择要评估的样本索引。

## 📊 模型架构

### LLaVA模型组成

- **Vision Tower**: CLIP-ViT-Large视觉编码器（冻结参数）
- **Multi-modal Projector**: 视觉-语言投影层（可训练）
- **Language Model**: Qwen1.5-4B-Chat语言模型（LoRA微调）

### LoRA配置

- 仅在语言模型的`q_proj`和`v_proj`层应用LoRA
- 同时训练`multi_modal_projector`模块
- 视觉编码器参数完全冻结

## 📈 训练细节

### 数据集格式

数据集应包含：

- `chat.json`: 包含对话数据的JSON文件
- `images_dl/`: 图像文件目录

每个样本格式：

```json
{
  "image": "image_path.jpg",
  "conversations": [
    {"role": "human", "value": "问题文本"},
    {"role": "gpt", "value": "答案文本"}
  ]
}
```

### 训练策略

- 使用梯度检查点以节省显存
- 梯度裁剪（max_grad_norm=0.3）
- 混合精度训练（bf16）
- 每200步保存一次检查点

## 🔍 评估指标

对比评估脚本会计算：

- **Exact Match (EM)**: 生成文本与标准答案的严格匹配率
- 参数统计：各模块的参数量和可训练参数量

## 📝 注意事项

1. **模型路径**: 确保所有模型和数据集路径正确配置
2. **显存管理**: 如果显存不足，可以减小`per_device_train_batch_size`或启用梯度累积
3. **数据集**: 数据集和权重文件夹为空，使用前需从HuggingFace自行下载
4. **设备配置**: 默认使用`cuda:0`，可根据实际情况修改

## 🛠️ 主要文件说明

- `merge_llava_model.py`: 模型合并脚本，将Qwen和CLIP组合成LLaVA
- `lora_train.py`: LoRA微调训练主脚本
- `Dataset.py`: 实现`LlavaDataset`数据集类和`TrainLlavaModelCollator`数据整理器
- `llava_contrast_demo.py`: Gradio可视化对比界面
- `llava_contrast_eval.py`: 命令行对比评估脚本

## 📚 参考资料

- [LLaVA论文](https://arxiv.org/abs/2304.08485)
- [Qwen1.5模型](https://huggingface.co/Qwen/Qwen1.5-4B-Chat)
- [CLIP模型](https://huggingface.co/openai/clip-vit-large-patch14-336)
- [PEFT LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)

## 👥 作者

王秀程

## 📄 许可证

本项目仅用于课程学习和研究目的。

---

**提示**: 首次运行前请确保已下载所有必需的模型和数据集文件。
