# gradio_contraste.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Tuple, Dict, Any
from PIL import Image
import torch
import gradio as gr

from transformers import LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from myDataset import LlavaDataset

# === 顶部封面图（把本地文件转成 base64，便于 share=True 场景展示） ===
import base64, mimetypes
COVER_PATH = "image.jpg"  # 你要显示的图片文件
def _inline_src(p):
    mime = mimetypes.guess_type(p)[0] or "image/jpeg"
    with open(p, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{b64}"
COVER_SRC = _inline_src(COVER_PATH)


# ========= 可配置路径 =========
BASE_PATH    = "show_model/model001"       # 零样本模型目录/仓库名
ADAPTER_PATH = "output"                    # LoRA/adapter 目录（或具体 checkpoint，如 output/checkpoint-800）
ADAPTER_NAME = "peft_v1"                   # 训练时的 adapter_name
DATA_DIR     = "LLaVA-CC3M-Pretrain-595K"  # 数据集根目录
DEVICE       = "cuda:0"
DTYPE        = torch.bfloat16              # A6000 也可用 float16

# ========= 生成参数默认值 =========
DEFAULT_MAX_NEW_TOKENS = 32
DEFAULT_DO_SAMPLE      = False
DEFAULT_TEMPERATURE    = 0.7
DEFAULT_TOP_P          = 0.9

# ========= 工具函数 =========
def normalize_text(s: str):
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", s)
    s = " ".join(s.split())
    return s

# ========= 全局状态（懒加载） =========
STATE: Dict[str, Any] = dict(
    processor=None,
    base_model=None,
    ft_model=None,
    dataset=None,
)

def _ensure_loaded():
    """首次调用时加载 Processor / Dataset / 模型；重复调用走缓存。"""
    if STATE["processor"] is None:
        processor = LlavaProcessor.from_pretrained(BASE_PATH)
        # 兜底/兼容老字段
        processor.patch_size = getattr(processor, "patch_size", 14)
        processor.vision_feature_select_strategy = getattr(processor, "vision_feature_select_strategy", "default")
        tok = processor.tokenizer
        if getattr(tok, "pad_token_id", None) is None:
            tok.pad_token = tok.eos_token or "</s>"
        STATE["processor"] = processor

    if STATE["dataset"] is None:
        STATE["dataset"] = LlavaDataset(dataset_dir=DATA_DIR)

    if STATE["base_model"] is None:
        base_model = LlavaForConditionalGeneration.from_pretrained(
            BASE_PATH, device_map=DEVICE, torch_dtype=DTYPE
        ).eval()
        if hasattr(base_model.config, "pad_token_id") and base_model.config.pad_token_id is None:
            base_model.config.pad_token_id = STATE["processor"].tokenizer.pad_token_id
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = True
        STATE["base_model"] = base_model

    if STATE["ft_model"] is None:
        ft_model = LlavaForConditionalGeneration.from_pretrained(
            BASE_PATH, device_map=DEVICE, torch_dtype=DTYPE
        )
        ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH, adapter_name=ADAPTER_NAME)
        ft_model.eval()
        if hasattr(ft_model.config, "pad_token_id") and ft_model.config.pad_token_id is None:
            ft_model.config.pad_token_id = STATE["processor"].tokenizer.pad_token_id
        if hasattr(ft_model.config, "use_cache"):
            ft_model.config.use_cache = True
        STATE["ft_model"] = ft_model

def _prepare_inputs(q_text: str, image_path: str):
    """组装对话模板，补 <image>，编码到 DEVICE。返回 (inputs, input_len, prompt, image_pil)。"""
    if "<image>" not in q_text:
        q_text = "<image>\n" + q_text

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": q_text},
    ]
    processor = STATE["processor"]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, text=prompt, return_tensors="pt")
    for k in inputs.keys():
        inputs[k] = inputs[k].to(DEVICE)
    input_len = inputs["input_ids"].shape[1]
    return inputs, input_len, prompt, image

@torch.inference_mode()
def _generate(model, inputs, input_len, max_new_tokens, do_sample, temperature, top_p):
    gen_kwargs = dict(
        max_new_tokens=int(max_new_tokens),
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[:, input_len:]
    text = STATE["processor"].batch_decode(
        new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return text

def run_single_by_index(
    sample_index: int,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
):
    """以数据集索引进行一次对比推理，返回 UI 需要的字段。"""
    _ensure_loaded()
    ds = STATE["dataset"]

    # 防越界
    n = len(ds)
    idx = max(0, min(int(sample_index), n - 1))
    q_text, gt_text, image_path = ds[idx]

    inputs, input_len, _, image = _prepare_inputs(q_text, image_path)

    base_text = _generate(
        STATE["base_model"], inputs, input_len, max_new_tokens, do_sample, temperature, top_p
    )
    ft_text = _generate(
        STATE["ft_model"], inputs, input_len, max_new_tokens, do_sample, temperature, top_p
    )

    # 计算 EM（若 GT 存在）
    em_base = em_ft = ""
    if isinstance(gt_text, str) and len(gt_text.strip()) > 0:
        em_base = 1.0 if normalize_text(base_text) == normalize_text(gt_text) else 0.0
        em_ft   = 1.0 if normalize_text(ft_text)  == normalize_text(gt_text)  else 0.0

    # UI 显示
    q_clean = q_text.replace("<image>\n", "")
    info_md = f"**样本索引**: {idx}  \n**图片路径**: `{image_path}`"
    qa_md = (
        f"**问题 (Q)**: {q_clean}\n\n"
        f"**[未微调 零样本] A**:\n\n{base_text}\n\n"
        f"**[已微调 (LoRA + projector)] A**:\n\n{ft_text}\n"
    )
    if em_base != "" and em_ft != "":
        qa_md += f"\n> 严格匹配 EM：零样本 = **{em_base}**，微调 = **{em_ft}**\n"
        qa_md += f"> GT: {gt_text}\n"

    return image, info_md, qa_md

def run_single_by_image(
    image: Image.Image,
    question: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    do_sample: bool = DEFAULT_DO_SAMPLE,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
):
    """上传任意图片 + 自定义问题的对比推理（无 GT 评估）。"""
    _ensure_loaded()

    q_text = question or "Describe the image."
    if "<image>" not in q_text:
        q_text = "<image>\n" + q_text

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": q_text},
    ]
    processor = STATE["processor"]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    for k in inputs.keys():
        inputs[k] = inputs[k].to(DEVICE)
    input_len = inputs["input_ids"].shape[1]

    base_text = _generate(
        STATE["base_model"], inputs, input_len, max_new_tokens, do_sample, temperature, top_p
    )
    ft_text = _generate(
        STATE["ft_model"], inputs, input_len, max_new_tokens, do_sample, temperature, top_p
    )

    q_clean = q_text.replace("<image>\n", "")
    info_md = f"**自定义图片对比**"
    qa_md = (
        f"**问题 (Q)**: {q_clean}\n\n"
        f"**[未微调 零样本] A**:\n\n{base_text}\n\n"
        f"**[已微调 (LoRA + projector)] A**:\n\n{ft_text}\n"
    )
    return image, info_md, qa_md

# ========= Gradio UI =========
with gr.Blocks(title="LLaVA 对比可视化 (零样本 vs 微调)") as demo:
    # === 顶部大标题（两行，内嵌图片） ===
    gr.HTML(
        f"""
<div style="display:flex;align-items:center;gap:16px;margin:8px 0 16px 0;">
  <img src="{COVER_SRC}" alt="cover" style="height:72px;border-radius:12px;object-fit:cover;">
  <div>
    <div style="font-size:28px;font-weight:700;line-height:1.25;margin-top:4px;">
      人工智能原理与应用第18组作业演示
    </div>
  </div>
</div>
"""
    )


    gr.Markdown("## LLaVA 对比可视化 (零样本 vs 微调)\n"
                "- **左侧 Tab1：按数据集索引选择样本**；**Tab2：上传任意图片 + 自定义问题**。\n"
                "- 模型在首次推理时加载，之后复用，避免重复初始化。\n"
                "- 默认关闭采样（等价温度=0），也可打开采样并调节温度/Top-p。")

    with gr.Tabs():
        with gr.Tab("按样本索引对比"):
            with gr.Row():
                with gr.Column(scale=1):
                    idx_in = gr.Number(value=1600, label="样本索引（int）")
                    max_new = gr.Slider(8, 128, value=DEFAULT_MAX_NEW_TOKENS, step=1, label="max_new_tokens")
                    do_sample = gr.Checkbox(value=DEFAULT_DO_SAMPLE, label="do_sample (开启随机采样)")
                    temp = gr.Slider(0.1, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=DEFAULT_TOP_P, step=0.05, label="top_p")
                    btn_run = gr.Button("运行对比", variant="primary")
                with gr.Column(scale=2):
                    out_img = gr.Image(label="样本图片", show_download_button=False)
                    out_info = gr.Markdown()
                    out_qa = gr.Markdown()

            btn_run.click(
                fn=run_single_by_index,
                inputs=[idx_in, max_new, do_sample, temp, top_p],
                outputs=[out_img, out_info, out_qa]
            )

        with gr.Tab("自定义图片 + 问题"):
            with gr.Row():
                with gr.Column(scale=1):
                    up_img = gr.Image(type="pil", label="上传图片")
                    q_in = gr.Textbox(value="Create a compact narrative representing the image presented.",
                                      label="问题（自动补 <image> ）")
                    max_new2 = gr.Slider(8, 128, value=DEFAULT_MAX_NEW_TOKENS, step=1, label="max_new_tokens")
                    do_sample2 = gr.Checkbox(value=DEFAULT_DO_SAMPLE, label="do_sample (开启随机采样)")
                    temp2 = gr.Slider(0.1, 1.5, value=DEFAULT_TEMPERATURE, step=0.05, label="temperature")
                    top_p2 = gr.Slider(0.1, 1.0, value=DEFAULT_TOP_P, step=0.05, label="top_p")
                    btn_run2 = gr.Button("运行对比", variant="primary")
                with gr.Column(scale=2):
                    out_img2 = gr.Image(label="图片", show_download_button=False)
                    out_info2 = gr.Markdown()
                    out_qa2 = gr.Markdown()

            btn_run2.click(
                fn=run_single_by_image,
                inputs=[up_img, q_in, max_new2, do_sample2, temp2, top_p2],
                outputs=[out_img2, out_info2, out_qa2]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)

