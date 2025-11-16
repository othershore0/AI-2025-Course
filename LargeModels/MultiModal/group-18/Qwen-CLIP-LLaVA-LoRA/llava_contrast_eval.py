# contraste.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PIL import Image
import torch
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from peft import PeftModel
from myDataset import LlavaDataset

# ========= 可配置路径 =========
BASE_PATH    = "show_model/model001"       # 基座模型目录/仓库名
ADAPTER_PATH = "output"                    # LoRA/adapter 目录（或具体 checkpoint，如 output/checkpoint-800）
ADAPTER_NAME = "peft_v1"                   # 训练时的 adapter_name
DATA_DIR     = "LLaVA-CC3M-Pretrain-595K"  # 数据集根目录
SAMPLE_IDX   = 1500                        # 要对比的样本索引
DEVICE       = "cuda:0"
DTYPE        = torch.bfloat16              # A6000 推理也可改 torch.float16

# ========= 工具函数 =========
def count_parameters(module):
    return sum(p.numel() for p in module.parameters()) if module is not None else 0

def count_parameters_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad) if module is not None else 0

def count_parameters_loraAB(module):
    if module is None:
        return 0
    s = 0
    for name, param in module.named_parameters():
        parts = name.split(".")
        if "lora_A" in parts or "lora_B" in parts:
            s += param.numel()
    return s

def normalize_text(s: str):
    import re
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", s)  # 粗略保留中英数字
    s = " ".join(s.split())
    return s

# ========= 加载数据 =========
dataset = LlavaDataset(dataset_dir=DATA_DIR)
q_text, gt_text, image_path = dataset[SAMPLE_IDX]

# 确保问句包含 <image>（很多 LLaVA 必须）
if "<image>" not in q_text:
    q_text = "<image>\n" + q_text

# ========= 处理器（修复告警） =========
processor = LlavaProcessor.from_pretrained(BASE_PATH)

# 修补 future-deprecation 的两个属性（ViT-L/14 常用 14；若视觉塔是 ViT-B/16 改成 16）
processor.patch_size = getattr(processor, "patch_size", 14)
processor.vision_feature_select_strategy = getattr(processor, "vision_feature_select_strategy", "default")

# pad_token 兜底
tok = processor.tokenizer
if getattr(tok, "pad_token_id", None) is None:
    tok.pad_token = tok.eos_token or "</s>"

# ========= 构建对话模板 & 编码输入 =========
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": q_text},
]
prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, text=prompt, return_tensors="pt")
for k in inputs.keys():
    inputs[k] = inputs[k].to(DEVICE)
input_len = inputs["input_ids"].shape[1]  # 用于裁掉输入，保留新生成

gen_kwargs = dict(max_new_tokens=32, do_sample=False)  # 温度=0时禁用采样

# ========= 1) 基座（未加载 LoRA）推理 =========
base_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_PATH, device_map=DEVICE, torch_dtype=DTYPE
).eval()
if hasattr(base_model.config, "pad_token_id") and base_model.config.pad_token_id is None:
    base_model.config.pad_token_id = tok.pad_token_id
if hasattr(base_model.config, "use_cache"):
    base_model.config.use_cache = True

with torch.inference_mode():
    base_out = base_model.generate(**inputs, **gen_kwargs)
base_new = base_out[:, input_len:]  # 只取新增token
base_text = processor.batch_decode(base_new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ========= 2) 已微调（加载 LoRA + projector）推理 =========
ft_model = LlavaForConditionalGeneration.from_pretrained(
    BASE_PATH, device_map=DEVICE, torch_dtype=DTYPE
)
ft_model = PeftModel.from_pretrained(ft_model, ADAPTER_PATH, adapter_name=ADAPTER_NAME)
ft_model.eval()
# 需要单文件部署/更省显存时，可合并（合并后不便继续训练，谨慎）：
# ft_model = ft_model.merge_and_unload().eval()

if hasattr(ft_model.config, "pad_token_id") and ft_model.config.pad_token_id is None:
    ft_model.config.pad_token_id = tok.pad_token_id
if hasattr(ft_model.config, "use_cache"):
    ft_model.config.use_cache = True

with torch.inference_mode():
    ft_out = ft_model.generate(**inputs, **gen_kwargs)
ft_new = ft_out[:, input_len:]
ft_text = processor.batch_decode(ft_new, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# ========= 参数统计（微调后模型） =========
print("\n========== 参数统计（微调后模型） ==========")
print(f"Total model parameters: {count_parameters(ft_model)}")

layer = getattr(ft_model, "multi_modal_projector", None)
print(f"All parameters in layer 'multi_modal_projector': {count_parameters(layer)}")
print(f"Train parameters in layer 'multi_modal_projector': {count_parameters_trainable(layer)}")
if layer is not None:
    for name, param in layer.named_parameters():
        print(name, param.requires_grad)

lm = getattr(ft_model, "language_model", None)
print(f"All parameters in layer 'language_model': {count_parameters(lm)}")
print(f"Train parameters in layer 'language_model' (LoRA A/B): {count_parameters_loraAB(lm)}")

vt = getattr(ft_model, "vision_tower", None)
print(f"All parameters in layer 'vision_tower': {count_parameters(vt)}")
print(f"Train parameters in layer 'vision_tower': {count_parameters_trainable(vt)}")

# ========= 同图同问对比输出 =========
print("\n========== 同图同问对比 ==========")
print(f"样本索引: {SAMPLE_IDX}")
print(f"图片路径: {image_path}")
clean_q = q_text.replace("<image>\n", "")
print(f"问题(Q): {clean_q}")
print(f"[未微调 基座] A: {base_text}")
print(f"[已微调 (LoRA+projector)] A: {ft_text}")

# ========= 可选：与 GT 严格匹配 (EM) =========
if isinstance(gt_text, str) and len(gt_text.strip()) > 0:
    em_base = 1.0 if normalize_text(base_text) == normalize_text(gt_text) else 0.0
    em_ft   = 1.0 if normalize_text(ft_text)  == normalize_text(gt_text)  else 0.0
    print("\n[可选] 与GT的严格匹配(EM):")
    print(f"  GT: {gt_text}")
    print(f"  基座 EM = {em_base} | 微调 EM = {em_ft}")

# ========= 可选：展示图片 =========
try:
    image.show()
except Exception:
    pass
