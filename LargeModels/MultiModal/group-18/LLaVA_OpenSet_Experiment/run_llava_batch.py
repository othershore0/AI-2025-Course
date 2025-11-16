import os
import json
from tqdm import tqdm
from PIL import Image

import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    get_model_name_from_path,
    process_images,
)
from llava.utils import disable_torch_init


# ================== 路径和参数配置 ==================

# 本地 LLaVA 模型路径
MODEL_PATH = "/media/e509/本地磁盘1/lx_LLaVA/LLaVA/llava-v1.5-7b"

# 已知集（COCO 子集）
KNOWN_IMG_DIR = "data/known_images"
KNOWN_META_CSV = "data/known_meta.csv"

# 私有集（宿舍 / 教室 / 公告栏）
PRIVATE_IMG_DIR = "data/private_images"
PRIVATE_META_CSV = "data/private_labels.csv"

# 输出结果
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

KNOWN_RESULT_JSONL = os.path.join(RESULT_DIR, "results_known.jsonl")
PRIVATE_RESULT_JSONL = os.path.join(RESULT_DIR, "results_private.jsonl")


PROMPT = (
    "You are a vision classifier. Look at the image and determine the single main object.\n"
    "If it clearly belongs to one of these categories:\n"
    "person, dog, cat, car, bus, bicycle, train, truck, airplane, boat, tv, laptop,\n"
    "then answer with exactly that one word (for example: 'dog').\n"
    "If it does not fit any of those categories, answer with a short English noun phrase "
    "describing the main object or scene, such as 'dorm room', 'classroom', 'office desk', or "
    "'notice board'. Do not use full sentences and do not add any extra words."
)


# ================== 工具函数 ==================


def load_csv(path, split):
    """读取 known_meta.csv / private_labels.csv，统一成 {filename, gt_label, split}"""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            row = dict(zip(header, parts))
            if "gt_label" in row:
                gt_label = row["gt_label"]
            else:
                gt_label = row.get("label", "")
            items.append(
                {
                    "split": split,
                    "filename": row["filename"],
                    "gt_label": gt_label,
                }
            )
    return items


def build_prompt_with_image_token(raw_query: str, model) -> str:
    """
    完全照官方 llava/eval/run_llava.py 的方式：
    - 根据模型 config 决定是否用 <im_start><image><im_end>
    - 如果没有 IMAGE_PLACEHOLDER，则在问题前面插入 image token
    """
    qs = raw_query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

    if IMAGE_PLACEHOLDER in qs:
        # 我们其实没用占位符，但保留这段逻辑，不影响
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = qs.replace(IMAGE_PLACEHOLDER, image_token_se)
        else:
            qs = qs.replace(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    return qs


@torch.inference_mode()
def answer_single_image(
    image_path,
    raw_query,
    tokenizer,
    model,
    image_processor,
    conv_mode,
):
    """
    对单张图片 + 文本问题，调用一次 LLaVA，返回 answer 字符串。
    这里严格模仿官方 llava/eval/run_llava.py 里的 eval_model 写法。
    """

    # 1. 构造带 image token 的文本（qs）
    qs = build_prompt_with_image_token(raw_query, model)

    # 2. 构建对话模板
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 3. 加载图像
    image = Image.open(image_path).convert("RGB")
    images = [image]
    image_sizes = [image.size]

    # 4. 图像预处理（完全照官方：process_images(images, image_processor, model.config)）
    images_tensor = process_images(images, image_processor, model.config)
    if isinstance(images_tensor, list):
        images_tensor = [img.to(model.device, dtype=torch.float16) for img in images_tensor]
    else:
        images_tensor = images_tensor.to(model.device, dtype=torch.float16)

    # 5. 文本 token，插入 IMAGE_TOKEN_INDEX
    input_ids = (
        tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        .unsqueeze(0)
        .to(model.device)
    )

    # 6. generate（参数基本照官方，只是 temperature=0，关掉采样）
    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0.0,
        top_p=None,
        num_beams=1,
        max_new_tokens=64,
        use_cache=True,
    )

    # 7. 像官方那样直接 batch_decode 整个输出（不再切掉前面的 prompt）
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    # 有些模板会把 prompt 内容也解出来，这时我们可以简单取最后一段
    # 避免把用户问题也算进 answer 里
    # 常见情况是类似 "...ASSISTANT: dog"，我们取最后一行 / 最后一句
    if "\n" in outputs:
        last_line = outputs.split("\n")[-1].strip()
        if last_line:
            outputs = last_line

    return outputs


# ================== 主流程 ==================


if __name__ == "__main__":
    disable_torch_init()

    # 0. 加载模型
    model_path = MODEL_PATH
    model_name = get_model_name_from_path(model_path)
    print(f"[INFO] model_name = {model_name}")

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        model_base=None,
        model_name=model_name,
        load_8bit=False,
        load_4bit=True,   # 和你之前 --load-4bit 一致
        device="cuda",
    )

    # 自动推断对话模板（和官方代码一致）
    name_lower = model_name.lower()
    if "llama-2" in name_lower:
        conv_mode = "llava_llama_2"
    elif "mistral" in name_lower:
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in name_lower:
        conv_mode = "chatml_direct"
    elif "v1" in name_lower:
        conv_mode = "llava_v1"
    elif "mpt" in name_lower:
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    print(f"[INFO] using conv_mode = {conv_mode}")

    # 1. 读取已知集 & 私有集元信息
    known_items = load_csv(KNOWN_META_CSV, "known")
    private_items = load_csv(PRIVATE_META_CSV, "private")

    print(f"[INFO] known samples: {len(known_items)}")
    print(f"[INFO] private samples: {len(private_items)}")

    # 2. 跑已知集
    out_known = []
    print(f"[INFO] running split=known, num_samples={len(known_items)}")

    for item in tqdm(known_items, desc="known"):
        fname = item["filename"]
        img_path = os.path.join(KNOWN_IMG_DIR, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}, skip.")
            continue

        try:
            ans = answer_single_image(
                img_path,
                PROMPT,
                tokenizer,
                model,
                image_processor,
                conv_mode,
            )
        except Exception as e:
            print(f"[ERROR] failed on known image {fname}: {e}")
            ans = ""

        out_known.append(
            {
                "split": "known",
                "filename": fname,
                "gt_label": item["gt_label"],
                "question": PROMPT,
                "answer": ans,
            }
        )

    with open(KNOWN_RESULT_JSONL, "w", encoding="utf-8") as f:
        for r in out_known:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] saved {len(out_known)} results to {KNOWN_RESULT_JSONL}")

    # 3. 跑私有集
    out_private = []
    print(f"[INFO] running split=private, num_samples={len(private_items)}")

    for item in tqdm(private_items, desc="private"):
        fname = item["filename"]
        img_path = os.path.join(PRIVATE_IMG_DIR, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] image not found: {img_path}, skip.")
            continue

        try:
            ans = answer_single_image(
                img_path,
                PROMPT,
                tokenizer,
                model,
                image_processor,
                conv_mode,
            )
        except Exception as e:
            print(f"[ERROR] failed on private image {fname}: {e}")
            ans = ""

        out_private.append(
            {
                "split": "private",
                "filename": fname,
                "gt_label": item["gt_label"],
                "question": PROMPT,
                "answer": ans,
            }
        )

    with open(PRIVATE_RESULT_JSONL, "w", encoding="utf-8") as f:
        for r in out_private:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] saved {len(out_private)} results to {PRIVATE_RESULT_JSONL}")
