#!/usr/bin/env python3
"""
在 VQAv2 验证集上评估微调后的模型，并与基线结果对比，所用指标与基线脚本保持一致。
"""
import os
import sys
import json
import torch
import re

# 将项目根目录加入路径，便于导入 src
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig

# 关键路径配置
BASELINE_PATH = "outputs/vqav2_baseline.jsonl"      # 零样本推理的结果文件
LORA_DIR = "outputs/lora_adapter"                   # 微调得到的 LoRA 适配器
FINETUNED_PATH = "outputs/vqav2_finetuned.jsonl"    # 本次评估的输出
MODEL_ID = "Salesforce/blip2-opt-2.7b"
USE_4BIT = True

# 与基线相同的 VQA 规范化逻辑
ARTS = {"a", "an", "the"}
PUNC = r"""!?.,;:'"`-()[]{}"""
NUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", 
       "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

def norm(s):
    """对答案做大小写、标点等清洗，方便比较。"""
    s = s.lower().strip()
    s = ''.join(ch for ch in s if ch not in PUNC)
    s = ' '.join(NUM.get(w, w) for w in s.split() if w not in ARTS)
    return s

def vqa_score(pred, answers10):
    """按照官方规则计算 VQA 分数：min(命中次数 / 3, 1)。"""
    p = norm(pred)
    g = [norm(a["answer"]) for a in answers10]
    return min(sum(p == x for x in g) / 3.0, 1.0)

# 先载入基线推理结果
print("Loading baseline results...")
baseline_records = []
if not os.path.exists(BASELINE_PATH):
    print(f"Error: Baseline file not found: {BASELINE_PATH}")
    print("Please run 'python create_vqav2_baseline.py' first.")
    sys.exit(1)

with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            baseline_records.append(json.loads(line))

print(f"Loaded {len(baseline_records)} baseline records")

# 为评分加载标注数据
print("Loading annotations...")
from src.vqav2_dataset import load_vqav2_annotations
annotations_dict = load_vqav2_annotations("data/vqa/v2_mscoco_val2014_annotations.json")

# 初始化处理器
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 加载基础模型与 LoRA 适配器
print("Loading fine-tuned model...")
if not os.path.exists(LORA_DIR):
    print(f"Error: LoRA adapter not found: {LORA_DIR}")
    print("Please run 'python finetune_vqav2_lora.py' first.")
    sys.exit(1)

quant_config = None
if USE_4BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

base_model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
model.eval()
print("Model loaded successfully")

def process_vqa(image, question, processor, model):
    """与基线一致的提示词和解码流程，用于评估。"""
    prompt = f"Question: {question}\nAnswer with one or two words:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            num_beams=1,
        )
    
    # 仅保留新生成的 token
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]
    pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    pred = pred.replace("Answer:", "").split("\n")[0].strip()
    return pred

# 在与基线相同的样本上重新推理
os.makedirs("outputs", exist_ok=True)
print(f"\nRunning fine-tuned model inference on {len(baseline_records)} samples...")

total_vqa_score = 0.0
exact_matches = 0

with open(FINETUNED_PATH, "w", encoding="utf-8") as f:
    for i, rec in enumerate(baseline_records, 1):
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
            question = rec["question"]
            question_id = rec["question_id"]
            
            # 执行推理
            predicted_answer = process_vqa(img, question, processor, model)
            
            # 取出评分所需的参考答案
            if question_id in annotations_dict:
                gt_answers = annotations_dict[question_id].get("answers", [])
                ground_truth_main = rec["ground_truth_answer"]
                
                # 更新 VQA 分数
                vqa_s = vqa_score(predicted_answer, gt_answers)
                total_vqa_score += vqa_s
                
                # 统计精确匹配率
                if norm(predicted_answer) == norm(ground_truth_main):
                    exact_matches += 1
            else:
                vqa_s = 0.0
            
            new_rec = {
                "question_id": question_id,
                "image_id": rec["image_id"],
                "image_path": rec["image_path"],
                "question": question,
                "ground_truth_answer": rec["ground_truth_answer"],
                "predicted_answer": predicted_answer,
                "vqa_score": vqa_s,
            }
            f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(baseline_records)} (VQA Acc: {total_vqa_score/i*100:.2f}%, EM: {exact_matches/i*100:.2f}%)")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

avg_vqa_score = total_vqa_score / len(baseline_records) * 100 if len(baseline_records) > 0 else 0
exact_match_rate = exact_matches / len(baseline_records) * 100 if len(baseline_records) > 0 else 0

print(f"\n✓ 已保存微调模型的推理结果到 {FINETUNED_PATH}")

# 对比微调前后的表现
print("\n" + "="*70)
print("对比：基线 vs 微调模型表现")
print("="*70)

# 从基线结果中统计指标
baseline_vqa_total = sum(r.get("vqa_score", 0) for r in baseline_records)
baseline_em = sum(1 for r in baseline_records if norm(r.get("predicted_answer", "")) == norm(r.get("ground_truth_answer", "")))

baseline_vqa_acc = baseline_vqa_total / len(baseline_records) * 100 if len(baseline_records) > 0 else 0
baseline_em_rate = baseline_em / len(baseline_records) * 100 if len(baseline_records) > 0 else 0

print(f"\n零样本基线：")
print(f"  精确匹配: {baseline_em}/{len(baseline_records)} ({baseline_em_rate:.2f}%)")
print(f"  VQA 准确率: {baseline_vqa_acc:.2f}%")

print(f"\n微调模型：")
print(f"  精确匹配: {exact_matches}/{len(baseline_records)} ({exact_match_rate:.2f}%)")
print(f"  VQA 准确率: {avg_vqa_score:.2f}%")

print(f"\n提升幅度：")
print(f"  精确匹配: {exact_match_rate - baseline_em_rate:+.2f}% ({exact_match_rate/baseline_em_rate*100-100:+.1f}% 相对提升)")
print(f"  VQA 准确率: {avg_vqa_score - baseline_vqa_acc:+.2f}% ({avg_vqa_score/baseline_vqa_acc*100-100:+.1f}% 相对提升)")

# 随机展示若干样例
print("\n" + "="*70)
print("随机采样的对比案例")
print("="*70)

import random
samples = random.sample(baseline_records, k=min(5, len(baseline_records)))

finetuned_dict = {}
with open(FINETUNED_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            finetuned_dict[r["question_id"]] = r

for baseline_rec in samples:
    question_id = baseline_rec["question_id"]
    if question_id not in finetuned_dict:
        continue
    
    ft_rec = finetuned_dict[question_id]
    
    print("\n" + "-"*70)
    print(f"图像: {os.path.basename(baseline_rec['image_path'])}")
    print(f"问题: {baseline_rec['question']}")
    print(f"参考答案: {baseline_rec['ground_truth_answer']}")
    print(f"\n基线预测: '{baseline_rec['predicted_answer']}'")
    print(f"微调预测: '{ft_rec['predicted_answer']}'")
    if baseline_rec.get('vqa_score') is not None:
        print(f"基线 VQA 分数: {baseline_rec['vqa_score']:.3f}")
    print(f"微调 VQA 分数: {ft_rec['vqa_score']:.3f}")

print("\n" + "="*70)
