#!/usr/bin/env python3
"""
用于可视化对比基线模型与微调模型的回答效果，生成并展示同一图像/问题下的差异。
"""
import os
import sys
import json
import random
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# 将项目根目录加入 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 主要配置
BASELINE_PATH = "outputs/vqav2_baseline.jsonl"
FINETUNED_PATH = "outputs/vqav2_finetuned.jsonl"
OUTPUT_DIR = "assets/comparisons"
NUM_SAMPLES = 100  # 需要展示的样本数，可按需增减

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取结果文件
print("正在载入基线结果...")
baseline_dict = {}
if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                baseline_dict[rec["question_id"]] = rec
print(f"已读取 {len(baseline_dict)} 条基线记录")
else:
    print(f"警告: 未找到基线文件 {BASELINE_PATH}")
    sys.exit(1)

print("正在载入微调结果...")
finetuned_dict = {}
if os.path.exists(FINETUNED_PATH):
    with open(FINETUNED_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                finetuned_dict[rec["question_id"]] = rec
    print(f"已读取 {len(finetuned_dict)} 条微调记录")
else:
    print(f"警告: 未找到微调结果 {FINETUNED_PATH}")
    print("请先运行 'python evaluate_vqav2_finetuned.py'。")
    sys.exit(1)

# 找到同时存在的 question_id
common_ids = set(baseline_dict.keys()) & set(finetuned_dict.keys())
print(f"\n共有 {len(common_ids)} 个问题同时存在于两份结果中")

if len(common_ids) == 0:
    print("错误: 两份结果没有重叠的问题条目。")
    sys.exit(1)

# 抽样
selected_ids = random.sample(list(common_ids), k=min(NUM_SAMPLES, len(common_ids)))

def create_comparison_image(baseline_rec, finetuned_rec, output_path):
    """生成左右并排的对比图。"""
    try:
        # 载入图像
        img_path = baseline_rec["image_path"]
        img = Image.open(img_path).convert("RGB")
        
        # 图像过大时按比例缩放
        max_size = 800
        if max(img.size) > max_size:
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # 计算画布尺寸与文字区域
        img_width, img_height = img.size
        
        # 预估文本区域的高度
        line_height = 30
        padding = 20
        text_area_height = 8 * line_height + padding * 2  # 预留足够空间
        
        # 创建画布
        canvas_height = img_height + text_area_height
        canvas = Image.new("RGB", (img_width * 2 + 40, canvas_height), color="white")
        draw = ImageDraw.Draw(canvas)
        
        # 优先使用系统字体，缺失时回退到默认字体
        try:
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            font_text = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font_title = ImageFont.load_default()
            font_text = ImageFont.load_default()
        
        # 将两张图像分别贴到画布左右
        canvas.paste(img, (0, 0))
        canvas.paste(img, (img_width + 40, 0))
        
        # 标题标签
        draw.text((img_width // 2, 10), "BASELINE", fill="blue", font=font_title, anchor="mm")
        draw.text((img_width + 40 + img_width // 2, 10), "FINE-TUNED", fill="green", font=font_title, anchor="mm")
        
        # 问题文本
        question = baseline_rec["question"]
        # 若问题过长，则截断显示
        max_chars = 50
        if len(question) > max_chars:
            question = question[:max_chars] + "..."
        
        y_start = img_height + padding
        draw.text((10, y_start), f"Question: {question}", fill="black", font=font_text)
        y_start += line_height
        
        # 真实答案
        gt = baseline_rec["ground_truth_answer"]
        draw.text((10, y_start), f"Ground Truth: {gt}", fill="black", font=font_text)
        y_start += line_height * 1.5
        
        # 基线模型答案
        baseline_ans = baseline_rec.get("predicted_answer", "")
        draw.text((10, y_start), f"Baseline Answer:", fill="blue", font=font_text)
        draw.text((10, y_start + line_height), f"  {baseline_ans}", fill="black", font=font_text)
        y_start += line_height * 2
        
        # 微调模型答案
        finetuned_ans = finetuned_rec.get("predicted_answer", "")
        draw.text((img_width + 50, img_height + padding), f"Question: {question}", fill="black", font=font_text)
        draw.text((img_width + 50, img_height + padding + line_height), 
                 f"Ground Truth: {gt}", fill="black", font=font_text)
        draw.text((img_width + 50, img_height + padding + line_height * 2.5), 
                 f"Fine-tuned Answer:", fill="green", font=font_text)
        draw.text((img_width + 50, img_height + padding + line_height * 3.5), 
                 f"  {finetuned_ans}", fill="black", font=font_text)
        
        # 若包含分数，则一并展示
        y_start += line_height
        if "vqa_score" in baseline_rec:
            draw.text((10, y_start), f"VQA Score: {baseline_rec['vqa_score']:.3f}", 
                     fill="blue", font=font_text)
        if "vqa_score" in finetuned_rec:
            draw.text((img_width + 50, img_height + padding + line_height * 4.5), 
                     f"VQA Score: {finetuned_rec['vqa_score']:.3f}", 
                     fill="green", font=font_text)
        
        # 保存图像
        canvas.save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"生成对比图出错: {e}")
        return False

def create_html_report(selected_samples, output_path):
    """生成包含所有样例对比的 HTML 报告。"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Baseline vs Fine-tuned Comparison</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .comparison-container {
            background: white;
            margin: 20px 0;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-row {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .image-box {
            flex: 1;
            text-align: center;
        }
        .image-box img {
            max-width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .baseline-box {
            border-color: #4a90e2;
        }
        .finetuned-box {
            border-color: #50c878;
        }
        .info-box {
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-top: 10px;
        }
        .label {
            font-weight: bold;
            color: #666;
        }
        .baseline-label {
            color: #4a90e2;
        }
        .finetuned-label {
            color: #50c878;
        }
        .answer {
            font-size: 18px;
            margin: 10px 0;
            padding: 10px;
            border-radius: 4px;
        }
        .baseline-answer {
            background: #e8f4f8;
            border-left: 4px solid #4a90e2;
        }
        .finetuned-answer {
            background: #e8f8f0;
            border-left: 4px solid #50c878;
        }
        .ground-truth {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .score {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
        }
        .baseline-score {
            background: #4a90e2;
            color: white;
        }
        .finetuned-score {
            background: #50c878;
            color: white;
        }
    </style>
</head>
<body>
    <h1>基线模型与微调模型对比</h1>
    <p style="text-align: center; color: #666;">VQAv2 验证集上的质性对比</p>
"""
    
    for i, qid in enumerate(selected_samples, 1):
        baseline_rec = baseline_dict[qid]
        finetuned_rec = finetuned_dict[qid]
        
        img_path = baseline_rec["image_path"]
        question = baseline_rec["question"]
        gt = baseline_rec["ground_truth_answer"]
        baseline_ans = baseline_rec.get("predicted_answer", "")
        finetuned_ans = finetuned_rec.get("predicted_answer", "")
        baseline_score = baseline_rec.get("vqa_score", 0)
        finetuned_score = finetuned_rec.get("vqa_score", 0)
        
        # Convert relative path to absolute for HTML
        if not os.path.isabs(img_path):
            img_path = os.path.join(project_root, img_path)
        
        html_content += f"""
    <div class="comparison-container">
        <h2>示例 {i}（Question ID: {qid}）</h2>
        <div class="info-box">
            <div class="label">问题：</div>
            <p style="font-size: 18px; margin: 10px 0;">{question}</p>
        </div>
        
        <div class="ground-truth">
            <div class="label">参考答案：</div>
            <p style="font-size: 20px; margin: 5px 0; font-weight: bold;">{gt}</p>
        </div>
        
        <div class="image-row">
            <div class="image-box baseline-box">
                <h3 class="baseline-label">基线（零样本）</h3>
                <img src="{img_path}" alt="Image">
                <div class="answer baseline-answer">
                    <div class="label baseline-label">答案：</div>
                    <p>{baseline_ans or '(empty)'}</p>
                    <span class="score baseline-score">VQA: {baseline_score:.3f}</span>
                </div>
            </div>
            
            <div class="image-box finetuned-box">
                <h3 class="finetuned-label">微调模型</h3>
                <img src="{img_path}" alt="Image">
                <div class="answer finetuned-answer">
                    <div class="label finetuned-label">答案：</div>
                    <p>{finetuned_ans or '(empty)'}</p>
                    <span class="score finetuned-score">VQA: {finetuned_score:.3f}</span>
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

# 开始生成各类对比材料
print(f"\n正在为 {len(selected_ids)} 个样本生成质性对比...")

# 逐个输出图片
created_images = []
for i, qid in enumerate(selected_ids, 1):
    baseline_rec = baseline_dict[qid]
    finetuned_rec = finetuned_dict[qid]
    
    output_image = os.path.join(OUTPUT_DIR, f"comparison_{i:03d}_qid_{qid}.jpg")
    if create_comparison_image(baseline_rec, finetuned_rec, output_image):
        created_images.append(output_image)
        print(f"  ✓ 已生成: {output_image}")

# 生成 HTML 报告
html_output = os.path.join(OUTPUT_DIR, "comparison_report.html")
create_html_report(selected_ids, html_output)
print(f"\n✓ 已生成 HTML 报告: {html_output}")

# 输出文本摘要
summary_output = os.path.join(OUTPUT_DIR, "comparison_summary.txt")
with open(summary_output, "w", encoding="utf-8") as f:
    f.write("基线 vs 微调 模型对比摘要\n")
    f.write("=" * 70 + "\n\n")
    
    for i, qid in enumerate(selected_ids, 1):
        baseline_rec = baseline_dict[qid]
        finetuned_rec = finetuned_dict[qid]
        
        f.write(f"\n示例 {i} (Question ID: {qid})\n")
        f.write("-" * 70 + "\n")
        f.write(f"图像: {baseline_rec['image_path']}\n")
        f.write(f"问题: {baseline_rec['question']}\n")
        f.write(f"参考答案: {baseline_rec['ground_truth_answer']}\n\n")
        f.write(f"基线回答: {baseline_rec.get('predicted_answer', '') or '(empty)'}\n")
        if 'vqa_score' in baseline_rec:
            f.write(f"基线 VQA 分数: {baseline_rec['vqa_score']:.3f}\n")
            f.write(f"  说明: VQA 分数基于与 10 个标注答案的匹配，而不仅是主答案。\n")
        f.write(f"\n微调回答: {finetuned_rec.get('predicted_answer', '') or '(empty)'}\n")
        if 'vqa_score' in finetuned_rec:
            f.write(f"微调 VQA 分数: {finetuned_rec['vqa_score']:.3f}\n")
            f.write(f"  说明: VQA 分数基于与 10 个标注答案的匹配，而不仅是主答案。\n")
        f.write("\n")

print(f"✓ 已生成摘要: {summary_output}")

print(f"\n" + "="*70)
print(f"质性对比已完成!")
print(f"="*70)
print(f"共生成 {len(created_images)} 张对比图片")
print(f"HTML 报告: {html_output}")
print(f"摘要文件: {summary_output}")
print(f"\n💡 可在浏览器中打开 {html_output} 查看对比详情。")

