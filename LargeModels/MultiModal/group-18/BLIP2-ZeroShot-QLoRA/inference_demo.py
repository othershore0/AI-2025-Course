import os
import sys
import base64

# 将项目根目录加入 Python 路径，方便导入 src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# LoRA 适配器解压目标目录
target_dir = "outputs/lora_adapter"

# 若目录不存在则先创建
os.makedirs(target_dir, exist_ok=True)

# 可在此解压 lora_adapter.zip

# 如果需要，可以在此安装依赖（与步骤 3 相同，重复安装也没有问题）

# 创建 assets/ 目录，用于保存截图或演示图片
os.makedirs("assets", exist_ok=True)

# 读取 xiaohui.jpg，转成 base64 嵌入页面
xiaohui_img_path = os.path.join(project_root, "xiaohui.jpg")
xiaohui_base64 = ""
if os.path.exists(xiaohui_img_path):
    with open(xiaohui_img_path, "rb") as img_file:
        xiaohui_base64 = base64.b64encode(img_file.read()).decode('utf-8')

import gradio as gr
from PIL import Image
import numpy as np
import glob

# 引入推理函数
from src.blip2_inference import generate_caption, answer

def _to_pil(img):
    """把 Gradio 返回的 NumPy 数组或 PIL 对象统一转成 RGB PIL.Image。"""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, np.ndarray):
        # gr.Image(type="numpy") 会返回 NumPy 数组
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise ValueError("Unsupported image type")

def infer(img, question):
    if img is None:
        return "请先上传一张图片。", ""
    try:
        pil = _to_pil(img)
    except Exception as e:
        return f"图片处理错误: {e}", ""
    q = (question or "").strip()
    if not q:
        q = "What is in the image?"
    # 调用预设的推理函数
    cap = generate_caption(pil)
    ans = answer(pil, q)
    return cap, ans

# 读取指定示例图片：shili1.jpg、shili2.jpg、shili3.jpg
def gather_examples():
    files = []
    example_names = ["shili1.jpg", "shili2.jpg", "shili3.jpg"]
    for name in example_names:
        file_path = os.path.join(project_root, name)
        if os.path.exists(file_path):
            files.append(file_path)
    # Gradio Examples 要与输入组件一一对应: [ [图片, 问题], ... ]
    return [[f, "What is in the image?"] for f in files]

EXAMPLES = gather_examples()

with gr.Blocks(title="人工智能原理与应用第18组作业演示") as demo:
    # 使用 base64 的 xiaohui.jpg 做页眉
    header_html = f"""
        <div style="text-align: center; padding: 20px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-bottom: 15px;">
                <img src="data:image/jpeg;base64,{xiaohui_base64}" alt="小灰" style="width: 60px; height: 60px; border-radius: 50%; object-fit: cover; border: 2px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <h1 style="color: #1f77b4; margin: 0;">人工智能原理与应用第18组作业演示</h1>
            </div>
            <p style="font-size: 16px; color: #666; margin: 10px 0;">
                基于 BLIP-2 的视觉问答与图像描述系统
            </p>
            <p style="font-size: 14px; color: #888;">
                上传图片并输入问题，系统将生成图像描述和视觉问答答案
            </p>
            <hr style="margin: 20px 0; border: none; border-top: 2px solid #e0e0e0;">
            <p style="font-size: 12px; color: #999;">
                <strong>模型:</strong> <code>Salesforce/blip2-opt-2.7b</code>
            </p>
        </div>
    """
    gr.Markdown(header_html)
    with gr.Row():
        with gr.Column(scale=1):
            image = gr.Image(label="上传图片", type="numpy")
            question = gr.Textbox(label="输入问题", value="What is in the image?", placeholder="例如：图片中有什么？")
            with gr.Row():
                run_btn = gr.Button("生成结果", variant="primary")
                clear_btn = gr.Button("清空", variant="secondary")
            if EXAMPLES:
                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[image, question],
                    label="示例图片"
                )
        with gr.Column(scale=1):
            caption_out = gr.Textbox(label="图像描述 (Caption)", lines=3)
            answer_out = gr.Textbox(label="视觉问答答案 (Answer)", lines=3)

    # 绑定交互逻辑
    run_btn.click(infer, inputs=[image, question], outputs=[caption_out, answer_out])
    clear_btn.click(lambda: (None, "What is in the image?", "", ""),
                    inputs=None, outputs=[image, question, caption_out, answer_out])

# 启用队列提升并发；share=True 可生成公网链接
demo.queue(max_size=8).launch(share=True, debug=True)

# ==== 基线与微调对比 ====
import os, json, random
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

# 路径与模型设定
BASELINE_PATH = "outputs/baseline.jsonl"          # 步骤 9 生成的基线结果
LORA_DIR = "outputs/lora_adapter"                 # 步骤 11 得到的 LoRA 适配器
FINETUNED_PATH = "outputs/finetuned.jsonl"        # 本步骤输出
MODEL_ID = "Salesforce/blip2-opt-2.7b"            # 与前面保持一致
USE_4BIT = True                                   # 显存紧张时启用

# 1) 加载处理器
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 2) 加载基础模型并挂载 LoRA
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

# 3) 定义推理函数（必要时去掉提示词回显）
def generate_caption(image: Image.Image, max_new_tokens=30):
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

def answer(image: Image.Image, question: str, max_new_tokens=30):
    prompt = f"Question: {question} Answer:"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    return text

# 4) 读取基线结果，用同样的问题重新推理
os.makedirs("outputs", exist_ok=True)
records = []
with open(BASELINE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

with open(FINETUNED_PATH, "w", encoding="utf-8") as f:
    for i, rec in enumerate(records, 1):
        img = Image.open(rec["image_path"]).convert("RGB")
        cap = generate_caption(img)
        ans = answer(img, rec["question"])
        new_rec = {
            "image_path": rec["image_path"],
            "caption": cap,
            "question": rec["question"],
            "answer": ans,
        }
        f.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        print(f"[{i}/{len(records)}] {os.path.basename(rec['image_path'])} 完成")

print(f"\n已将微调模型结果写入 {FINETUNED_PATH}")

# 随机选取 3 个样本做对比
with open(BASELINE_PATH, "r", encoding="utf-8") as f1, \
     open(FINETUNED_PATH, "r", encoding="utf-8") as f2:
    base_lines = [json.loads(l) for l in f1 if l.strip()]
    fine_lines = [json.loads(l) for l in f2 if l.strip()]

pairs = list(zip(base_lines, fine_lines))
samples = random.sample(pairs, k=min(3, len(pairs)))

for b, ft in samples:
    print("="*50)
    print("图像:", b["image_path"])
    print("基线 Caption:", b["caption"])
    print("微调 Caption:", ft["caption"])
    print("基线 Answer:", b["answer"])
    print("微调 Answer:", ft["answer"])

import matplotlib.pyplot as plt

for b, ft in samples:
    img = Image.open(b["image_path"]).convert("RGB")
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(os.path.basename(b["image_path"]))
    plt.figtext(0.5, -0.05, f"基线 Caption: {b['caption']}\n微调 Caption: {ft['caption']}", ha="center", fontsize=10)
    plt.figtext(0.5, -0.15, f"基线 Answer: {b['answer']}\n微调 Answer: {ft['answer']}", ha="center", fontsize=10)
    plt.show()

import matplotlib.pyplot as plt
from PIL import Image
import json, random, os

# 重新载入结果，方便可视化
with open("outputs/baseline.jsonl", "r", encoding="utf-8") as f:
    baseline = [json.loads(l) for l in f if l.strip()]
with open("outputs/finetuned.jsonl", "r", encoding="utf-8") as f:
    finetuned = [json.loads(l) for l in f if l.strip()]

# 随机抽取若干样本绘图
samples = random.sample(list(zip(baseline, finetuned)), k=min(3, len(baseline)))

# 构建可视化画布
fig, axes = plt.subplots(len(samples), 1, figsize=(8, 6*len(samples)))
if len(samples) == 1:
    axes = [axes]

for ax, (b, ft) in zip(axes, samples):
    img = Image.open(b["image_path"]).convert("RGB")
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(os.path.basename(b["image_path"]), fontsize=14, weight="bold")
    ax.text(0.5, -0.15,
            f"基线 Caption: {b['caption']}\n微调 Caption: {ft['caption']}\n"
            f"基线 Answer: {b['answer']}\n微调 Answer: {ft['answer']}",
            transform=ax.transAxes, fontsize=10, va="top", ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

plt.tight_layout()
os.makedirs("assets", exist_ok=True)
plt.savefig("assets/teaser.jpg", bbox_inches="tight")
print("可视化已保存到 assets/teaser.jpg")
