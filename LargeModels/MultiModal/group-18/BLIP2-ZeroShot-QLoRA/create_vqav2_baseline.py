#!/usr/bin/env python3
"""
在 VQAv2 验证集上运行零样本推理，生成 baseline.jsonl（已按照官方评估规范调整）。
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
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from src.vqav2_dataset import load_vqav2_questions, load_vqav2_annotations, get_vqav2_image_path, get_top_answer

# 运行配置
VQAV2_QUESTIONS_PATH = "data/vqa/v2_OpenEnded_mscoco_val2014_questions.json"
VQAV2_ANNOTATIONS_PATH = "data/vqa/v2_mscoco_val2014_annotations.json"
VQAV2_IMAGES_DIR = "data/images"
BASELINE_OUTPUT = "outputs/vqav2_baseline.jsonl"
MAX_SAMPLES = 1000  # 若想处理全量，可改为 None

# 模型设置
MODEL_ID = "Salesforce/blip2-opt-2.7b"
USE_4BIT = False  # 显存有限时可切换为 True

# VQA 规范化与评分相关的工具函数
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

print("Loading model...")
quant_args = {}
if USE_4BIT:
    quant_args = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    quant_args = dict(load_in_8bit=True)

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    **quant_args
)
model.eval()

# 读取 VQAv2 验证集的题目与标注
print("Loading VQAv2 validation dataset...")
questions_dict = load_vqav2_questions(VQAV2_QUESTIONS_PATH)
annotations_dict = load_vqav2_annotations(VQAV2_ANNOTATIONS_PATH)

# 构造样本列表，确保题目和标注对齐
items = []
for question_id, question_data in questions_dict.items():
    if question_id not in annotations_dict:
        continue
    
    annotation = annotations_dict[question_id]
    image_id = question_data["image_id"]
    
    # 查找对应图像路径
    image_path = get_vqav2_image_path(image_id, VQAV2_IMAGES_DIR, split="val")
    if image_path is None:
        continue
    
    items.append({
        "question_id": question_id,
        "image_id": image_id,
        "question": question_data["question"],
        "annotation": annotation,  # 保留完整标注，后续评分需要
        "image_path": image_path,
    })

print(f"Total samples in validation set: {len(items)}")

# 如需抽样，这里进行裁剪
if MAX_SAMPLES is not None:
    items = items[:MAX_SAMPLES]
    print(f"Limited to {len(items)} samples for baseline evaluation")

os.makedirs("outputs", exist_ok=True)

def process_vqa(image, question, processor, model):
    """用统一的提示词和解码策略执行一次 VQA 推理。"""
    # 使用简洁的 prompt，引导模型输出短答案
    prompt = f"Question: {question}\nAnswer with one or two words:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,  # 限制生成长度，避免冗余
            do_sample=False,     # 采用贪心解码
            num_beams=1,         # 不启用束搜索
        )
    
    # 仅保留新增的 token，剔除 prompt
    input_length = inputs["input_ids"].shape[1]
    new_tokens = generated_ids[:, input_length:]
    
    # 解码生成的 token，得到预测答案
    pred = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    
    # 清理多余前缀，只保留首行内容
    pred = pred.replace("Answer:", "").split("\n")[0].strip()
    
    return pred

print(f"\nRunning zero-shot inference on {len(items)} samples...")
with open(BASELINE_OUTPUT, "w", encoding="utf-8") as f:
    total_vqa_score = 0.0
    exact_matches = 0
    
    for i, item in enumerate(items, 1):
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            question = item["question"]
            annotation = item["annotation"]
            question_id = item["question_id"]
            
            # 执行推理
            predicted_answer = process_vqa(img, question, processor, model)
            
            # 每个问题有 10 个参考答案
            gt_answers = annotation.get("answers", [])
            ground_truth_main = get_top_answer(gt_answers)
            
            # 计算指标并累积
            vqa_s = vqa_score(predicted_answer, gt_answers)
            total_vqa_score += vqa_s
            
            # 记录精确匹配
            if norm(predicted_answer) == norm(ground_truth_main):
                exact_matches += 1
            
            record = {
                "question_id": question_id,
                "image_id": item["image_id"],
                "image_path": item["image_path"],
                "question": question,
                "ground_truth_answer": ground_truth_main,
                "predicted_answer": predicted_answer,
                "vqa_score": vqa_s,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            if i % 100 == 0:
                print(f"Progress: {i}/{len(items)} (VQA Acc: {total_vqa_score/i*100:.2f}%, EM: {exact_matches/i*100:.2f}%)")
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue

avg_vqa_score = total_vqa_score / len(items) * 100 if len(items) > 0 else 0
exact_match_rate = exact_matches / len(items) * 100 if len(items) > 0 else 0

print(f"\n✓ 已保存零样本基线结果至 {BASELINE_OUTPUT}")
print(f"处理样本数量: {len(items)}")

print("\n" + "="*70)
print("零样本基线表现（采用修正指标）")
print("="*70)
print(f"  样本总数: {len(items)}")
print(f"  精确匹配（标准化）: {exact_matches}/{len(items)} ({exact_match_rate:.2f}%)")
print(f"  VQA 准确率: {total_vqa_score}/{len(items)} = {avg_vqa_score:.2f}%")
print("="*70)

