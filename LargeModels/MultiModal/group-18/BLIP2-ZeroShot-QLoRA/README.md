# 基于 BLIP-2 的零样本推理与 QLoRA 微调

> 人工智能原理与应用课程 · 第 18 组 · 王秀程

本项目复现并扩展了 BLIP-2 模型在视觉问答任务上的零样本能力，并通过 QLoRA 方法在 VQAv2 数据集上进行轻量级微调。除了标准的训练与评估脚本，仓库还提供了便捷的对比分析工具与 Gradio 可视化 Demo，便于课堂展示与实验复盘。

## 核心特性

- **零样本基线**：调用 `Salesforce/blip2-opt-2.7b` 在 COCO/VQAv2 样本上直接做 Caption 与 VQA 推理，构建可对比的初始表现。
- **QLoRA 微调**：使用 4bit 量化 + LoRA 插入，显著降低显存与计算成本，1000 步即可得到可提交的微调模型。
- **一致的评估流程**：基线与微调结果共享相同的提示词、解码与官方 VQA 评分标准，指标可直接对比。
- **多样的可视化方式**：脚本自动生成对比图片、HTML 报告与文本摘要，辅助课堂讲解。
- **交互式 Demo**：基于 Gradio 的网页端演示，支持上传任意图片并提问问题，实时展示 Caption 与答案。

## 项目结构

```
基于BLIP2的零样本推理与QLoRA微调/
├── src/                     # 核心模块：数据集封装、推理函数
├── data/                    # 数据目录（需放置 COCO/VQAv2 原始文件）
│   ├── images/README.md     # 指引如何解压 COCO 图像
│   └── vqa/README.md        # 指引如何放置 VQAv2 标注
├── baseline_images/         # 课堂演示用的 10 张图片
├── outputs/                 # 推理、评估与 LoRA 适配器的输出目录
├── finetune_vqav2_lora.py   # LoRA 微调主脚本
├── create_vqav2_baseline.py # VQAv2 验证集零样本推理
├── evaluate_vqav2_finetuned.py
├── compare_baseline_finetuned.py
├── inference_demo.py        # Gradio Demo（附带比较可视化）
├── blip2_cli.py             # 与 demo 功能类似的命令行入口
├── mini_infer.py            # 单张图片的快速体验脚本
├── requirements.txt
└── README.md
```

## 环境准备

- Python ≥ 3.10
- 推荐使用具备至少 24 GB 显存的 NVIDIA GPU（QLoRA 支持 4bit 量化，理论上 16 GB 级别也可运行但训练步数需酌情调整）
- 安装 PyTorch（建议与本地 CUDA 版本匹配，例如 `pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121`）
- 安装项目依赖：

```bash
pip install -r requirements.txt
```

如需使用 4bit 量化，请确保 `bitsandbytes` 能够正确加载 GPU（若出现报错，可尝试改用 8bit 或 CPU 演示模式）。

## 数据集准备

1. **COCO 2014 图像**  
   
   - 下载 `train2014.zip` 与 `val2014.zip`（可在官方 Kaggle/Mirror 获取），解压至 `data/images/train2014/` 与 `data/images/val2014/`。  
   - 详见 `data/images/README.md`。

2. **VQAv2 标注文件**  
   
   - 解压问题与标注的四个 JSON 文件到 `data/vqa/`，名称需保持默认格式。  
   - 详见 `data/vqa/README.md`。

3. **课程演示迷你集（可选）**  
   
   - 仓库自带 `data/tiny.jsonl` 与 `data/picture_*.jpg`，便于在无完整 COCO 的环境下做快速测试。

## 快速上手

1. **零样本体验（单图）**
   
   ```bash
   python mini_infer.py
   ```
   
   输出将显示 `baseline_images/pic_01.jpg` 的描述与回答。

2. **构建课堂基线（10 张示例图）**
   
   ```bash
   python create_baseline.py
   ```
   
   结果保存在 `outputs/baseline.jsonl`，便于与微调模型做并行对比。

3. **VQAv2 验证集零样本推理**  
   （需准备完整数据集，默认抽样 1000 个样本，可在脚本中修改 `MAX_SAMPLES`）
   
   ```bash
   python create_vqav2_baseline.py
   ```

4. **QLoRA 微调（训练 LoRA 适配器）**
   
   ```bash
   python finetune_vqav2_lora.py
   ```
   
   - 关键参数可在脚本顶部修改，如 `TRAIN_STEPS`、`PER_DEVICE_BATCH`、`USE_4BIT` 等。
   - 训练完成后，LoRA 权重与处理器将保存在 `outputs/lora_adapter/`。

5. **在验证集上评估微调模型**
   
   ```bash
   python evaluate_vqav2_finetuned.py
   ```
   
   脚本会重新跑与基线相同的一批样本，输出 `outputs/vqav2_finetuned.jsonl` 并打印精确匹配率与官方 VQA 指标变化。

6. **生成可视化对比材料**
   
   ```bash
   python compare_baseline_finetuned.py
   ```
   
   将在 `assets/comparisons/` 下生成对比图片、HTML 报告与文本摘要，适合放入课程汇报。

7. **启动 Gradio Demo**
   
   ```bash
   python inference_demo.py
   ```
   
   浏览器访问本地地址后，可上传图片并提问问题，实时查看 Caption 与回答。脚本末尾还包含基线与微调结果的自动对比与可视化。

## 结果示例

`outputs/baseline.jsonl` 中记录了零样本推理的原始输出，可作为课堂展示或优化对比的素材之一：

```1:10:outputs/baseline.jsonl
{"image_path": "baseline_images/pic_01.jpg", "caption": "a pizza with pineapple, onions and cilantro", "question": "What is in the image?", "answer": "A Hawaiian pizza with pineapple, bacon, and cilantro"}
{"image_path": "baseline_images/pic_02.jpg", "caption": "a toy mario standing on a wooden floor", "question": "What is in the image?", "answer": "A mario figurine"}
{"image_path": "baseline_images/pic_03.jpg", "caption": "the blue mosque in istanbul, turkey", "question": "What is in the image?", "answer": "Blue Mosque, Istanbul, Turkey"}
{"image_path": "baseline_images/pic_04.jpg", "caption": "kuala lumpur petronas twin towers at night", "question": "What is in the image?", "answer": "The Petronas Twin Towers in Kuala Lumpur, Malaysia"}
{"image_path": "baseline_images/pic_05.jpg", "caption": "a deer stands in a field with other deer", "question": "What is in the image?", "answer": "A buck deer in a field"}
{"image_path": "baseline_images/pic_06.jpg", "caption": "a dart board with a target on it", "question": "What is in the image?", "answer": "A dart board with a target on it"}
{"image_path": "baseline_images/pic_07.jpg", "caption": "a desert with sand dunes and rocks", "question": "What is in the image?", "answer": "The desert"}
{"image_path": "baseline_images/pic_08.jpg", "caption": "two people in a boat on a lake in the mountains", "question": "What is in the image?", "answer": "The lake of the dolomites, italy"}
{"image_path": "baseline_images/pic_09.jpg", "caption": "a red fox stands on a snowy hillside", "question": "What is in the image?", "answer": "A red fox"}
{"image_path": "baseline_images/pic_10.jpg", "caption": "a variety of meat and vegetables on a wooden table", "question": "What is in the image?", "answer": "A barbecue"}
```

运行 `evaluate_vqav2_finetuned.py` 后，终端会输出零样本基线与微调模型在验证集上的精确匹配率与 VQA 官方准确率，并随机打印若干案例，以此展示 QLoRA 带来的性能提升。

## 课程信息

- 课程：人工智能原理与应用
- 小组：第 18 组

欢迎在课堂或答辩时基于此 README 进行讲解，如需复现实验请提前准备 GPU 环境并下载完整数据集。

## 致谢

- [BLIP-2: Bootstrapping Language-Image Pre-training](https://arxiv.org/abs/2301.12597)
- Salesforce 官方的 `blip2-opt-2.7b` 模型与 Hugging Face Transformers 社区
- `peft`、`bitsandbytes` 等开源项目对低资源微调的支持
