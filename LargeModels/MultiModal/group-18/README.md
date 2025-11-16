# 多模态大模型课程作业 - 第18组

本项目是《人工智能原理与应用》课程的多模态大模型实验项目，包含多个子项目，涵盖了当前主流的多模态大模型技术，包括BLIP2、LLaVA、LAVIS等框架的应用与实践。

## 📁 项目结构

本项目包含以下5个子项目：

### 1. BLIP2-ZeroShot-QLoRA
**项目简介：** 基于BLIP2模型的零样本学习与QLoRA（Quantized Low-Rank Adaptation）微调实验。

**主要功能：**
- BLIP2模型的零样本推理
- 使用QLoRA技术对BLIP2进行高效微调
- VQAv2数据集的基准测试与微调评估
- 支持图像描述生成和视觉问答任务

**关键文件：**
- `finetune_vqav2_lora.py` - VQAv2数据集的LoRA微调脚本
- `evaluate_vqav2_finetuned.py` - 微调后模型评估
- `create_vqav2_baseline.py` - 创建VQAv2基准测试
- `inference_demo.py` - 推理演示脚本
- `src/blip2_inference.py` - BLIP2推理核心模块

### 2. LAVIS-main
**项目简介：** LAVIS（Language-Vision Intelligence System）多模态框架的应用与实践。

**主要功能：**
- 图像描述生成（Image Captioning）
- 视觉问答（Visual Question Answering, VQA）
- 图像文本匹配（Image-Text Matching）
- 零样本图像分类（Zero-shot Classification）
- 多模态特征提取
- 文本定位（Text Localization）
- 多模态搜索

**关键文件：**
- `app/main.py` - 主应用入口
- `app/vqa.py` - 视觉问答模块
- `app/caption.py` - 图像描述生成模块
- `app/multimodal_search.py` - 多模态搜索模块
- `evaluate.py` - 模型评估脚本
- `train.py` - 模型训练脚本

### 3. LLaVA_OpenSet_Experiment
**项目简介：** LLaVA（Large Language and Vision Assistant）模型的开放集（Open Set）实验。

**主要功能：**
- LLaVA模型的开放集评估
- 已知类别与未知类别的分类实验
- COCO数据集的开放集实验
- 模型在开放域场景下的性能评估

**关键文件：**
- `open_set_eval_simple.py` - 开放集评估脚本
- `predict.py` - 预测脚本
- `run_llava_batch.py` - 批量运行LLaVA
- `make_known_from_coco.py` - 从COCO数据集生成已知类别

### 4. Llava-Reproduction-main
**项目简介：** LLaVA模型的完整复现项目。

**主要功能：**
- LLaVA模型的完整实现
- 模型训练与推理
- 支持多种视觉-语言任务

**关键文件：**
- `predict.py` - 模型预测脚本
- `llava/` - LLaVA模型核心实现
- `scripts/` - 训练和评估脚本

### 5. Qwen-CLIP-LLaVA-LoRA
**项目简介：** 结合Qwen、CLIP和LLaVA模型的LoRA微调实验。

**主要功能：**
- 使用Qwen语言模型、CLIP视觉编码器和LLaVA架构
- LoRA（Low-Rank Adaptation）高效微调
- 多模态对比学习实验
- 支持自定义数据集的训练

**关键文件：**
- `lora_train.py` - LoRA训练脚本
- `llava_contrast_demo.py` - 对比学习演示
- `llava_contrast_eval.py` - 对比学习评估
- `merge_llava_model.py` - 模型合并脚本
- `Dataset.py` - 数据集处理模块

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.12+
- CUDA（推荐，用于GPU加速）
- 足够的存储空间（用于下载预训练模型）

### 安装依赖

每个子项目都有独立的依赖文件，请根据具体需求安装：

```bash
# BLIP2项目
cd BLIP2-ZeroShot-QLoRA
pip install -r requirements.txt

# LAVIS项目
cd LAVIS-main
pip install -r requirements.txt
pip install -e .

# LLaVA项目
cd Llava-Reproduction-main
pip install -e .

# Qwen-CLIP-LLaVA项目
cd Qwen-CLIP-LLaVA-LoRA
pip install -r requirements.txt
```

## 📊 实验内容

### 实验1：BLIP2零样本与微调
- **目标：** 评估BLIP2在零样本场景下的性能，并通过QLoRA进行高效微调
- **数据集：** VQAv2
- **评估指标：** 准确率、BLEU分数等

### 实验2：LAVIS多模态应用
- **目标：** 探索LAVIS框架在多种多模态任务中的应用
- **任务：** VQA、图像描述、图像分类、多模态搜索等
- **特点：** 统一的框架支持多种任务

### 实验3：LLaVA开放集实验
- **目标：** 评估LLaVA在开放集场景下的泛化能力
- **数据集：** COCO 2017
- **评估：** 已知类别vs未知类别的分类性能

### 实验4：LLaVA模型复现
- **目标：** 完整复现LLaVA模型，理解其架构与训练流程
- **内容：** 模型实现、训练、推理全流程

### 实验5：Qwen-CLIP-LLaVA LoRA微调
- **目标：** 结合多个先进模型，使用LoRA进行高效微调
- **特点：** 多模型融合、参数高效微调

## 🔧 使用说明

### BLIP2项目
```bash
cd BLIP2-ZeroShot-QLoRA
# 零样本推理
python blip2_cli.py --image <image_path> --question "What is in the image?"

# 微调模型
python finetune_vqav2_lora.py

# 评估模型
python evaluate_vqav2_finetuned.py
```

### LAVIS项目
```bash
cd LAVIS-main
# 运行Web应用
python -m app.main

# 或使用Streamlit
streamlit run app/main.py
```

### LLaVA开放集实验
```bash
cd LLaVA_OpenSet_Experiment
# 运行开放集评估
python open_set_eval_simple.py
```

### Qwen-CLIP-LLaVA项目
```bash
cd Qwen-CLIP-LLaVA-LoRA
# 训练LoRA模型
python lora_train.py

# 评估模型
python llava_contrast_eval.py
```

## 📈 实验结果

各子项目的详细实验结果请参考各子项目目录下的输出文件或报告。

## 📝 技术要点

1. **参数高效微调（PEFT）：** 使用LoRA和QLoRA技术，在保持模型性能的同时大幅减少训练参数
2. **多模态融合：** 探索视觉编码器与语言模型的融合方式
3. **零样本学习：** 评估预训练模型在未见过任务上的泛化能力
4. **开放集评估：** 测试模型在开放域场景下的鲁棒性
5. **对比学习：** 使用对比学习提升多模态表示质量

## 👥 小组成员

第18组

## 📄 许可证

本项目仅用于课程学习与研究目的。

## 🙏 致谢

感谢以下开源项目：
- [BLIP2](https://github.com/salesforce/BLIP)
- [LAVIS](https://github.com/salesforce/LAVIS)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [Qwen](https://github.com/QwenLM/Qwen)

## 📚 参考资料

- BLIP2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
- LLaVA: Visual Instruction Tuning
- LAVIS: A One-stop Library for Language-Vision Intelligence
- LoRA: Low-Rank Adaptation of Large Language Models

---

**注意：** 本项目需要下载大量预训练模型，请确保有足够的网络带宽和存储空间。部分模型可能需要特定的硬件配置（如GPU显存要求）。

