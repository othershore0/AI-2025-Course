#!/usr/bin/env python3
"""
用于 VQAv2 数据集的 LoRA 微调脚本。
该脚本基于 VQAv2 数据集对 BLIP-2 模型进行训练。
"""
import os
import sys
import json
import math
import torch
from dataclasses import dataclass
from typing import Dict, List, Any

# 将项目根目录加入 Python 路径，便于导入 src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ==== 配置 ====
MODEL_ID = "Salesforce/blip2-opt-2.7b"

# VQAv2 数据集配置
VQAV2_QUESTIONS_PATH = "data/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
VQAV2_ANNOTATIONS_PATH = "data/vqa/v2_mscoco_train2014_annotations.json"
VQAV2_IMAGES_DIR = "data/images"
VQAV2_SPLIT = "train"                    # 使用训练集进行训练

OUTPUT_DIR = "outputs/lora_adapter"      # LoRA 适配器保存目录
USE_4BIT = True                          # 启用 4bit（QLoRA），降低显存占用
MAX_NEW_TOKENS = 32                      # 推理时的最大生成长度
MAX_SEQ_LEN = 128                        # 文本最大序列长度（prompt + target）
TRAIN_STEPS = 1000                       # 训练步数（可根据资源调整）
LR = 5e-5                                # 学习率
WARMUP_RATIO = 0.03
GRAD_ACCUM = 4                           # 梯度累积步数
PER_DEVICE_BATCH = 1                      # 单卡 batch size（T4 建议为 1-2）
SEED = 42

# =======================

os.makedirs(OUTPUT_DIR, exist_ok=True)

from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig, TrainingArguments, Trainer, set_seed
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

device = "cuda" if torch.cuda.is_available() else "cpu"

# 量化配置（用于 QLoRA）
quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # A100 可用 bfloat16；T4 上 float16 更稳
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)

# 加载基础模型
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)

# 冻结除语言解码器外的大部分参数（视觉编码器/Q-Former 保持冻结）
for name, p in model.named_parameters():
    p.requires_grad = False

# 为 QLoRA 训练做准备（仅 k-bit 量化模型需要）
if USE_4BIT:
    model = prepare_model_for_kbit_training(model)

# 在 OPT 的注意力与 MLP 层插入 LoRA 适配器
# 常见目标模块：q_proj, k_proj, v_proj, out_proj, fc1, fc2
lora_cfg = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
)
model = get_peft_model(model, lora_cfg)

# 打印可训练参数与总参数数量
trainable, total = 0, 0
for _, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()
print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M")

# 导入 VQAv2 数据集类
from src.vqav2_dataset import VQAv2Dataset

# 加载 VQAv2 数据集
print(f"Loading VQAv2 dataset from {VQAV2_QUESTIONS_PATH}")
dataset = VQAv2Dataset(
    questions_json_path=VQAV2_QUESTIONS_PATH,
    annotations_json_path=VQAV2_ANNOTATIONS_PATH,
    images_dir=VQAV2_IMAGES_DIR,
    processor=processor,
    max_len=MAX_SEQ_LEN,
    split=VQAV2_SPLIT,
    use_multiple_choice_answer=False,
)

print(f"Dataset size: {len(dataset)}")
if len(dataset) > 0:
    print(f"Sample keys: {dataset[0].keys()}")

@dataclass
class Collator:
    """数据整理器：将样本打包并堆叠为批次张量。"""
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        keys = features[0].keys()
        for k in keys:
            batch[k] = torch.stack([f[k] for f in features])
        return batch

collate_fn = Collator()

set_seed(SEED)

# 基于设定的总步数估算训练轮数
steps_per_epoch = math.ceil(len(dataset) / (PER_DEVICE_BATCH * GRAD_ACCUM))
num_train_epochs = max(1, math.ceil(TRAIN_STEPS / steps_per_epoch))

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=num_train_epochs,
    max_steps=TRAIN_STEPS,                # 使用基于步数的训练
    warmup_ratio=WARMUP_RATIO,
    logging_steps=10,
    save_steps=1000000,                   # 演示规模下跳过中间检查点
    fp16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

trainer.train()

# 保存 LoRA 适配器与处理器
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to:", OUTPUT_DIR)

