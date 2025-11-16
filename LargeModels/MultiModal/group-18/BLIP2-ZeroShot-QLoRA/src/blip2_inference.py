from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration

# ==== 可调节参数 ====
MODEL_ID = "Salesforce/blip2-opt-2.7b"
USE_4BIT = False                      # True 表示采用 4bit 量化，False 则为 8bit
MAX_NEW_TOKENS = 30                   # 控制生成文本的长度上限
# ===================

# 模型只加载一次，后续调用复用
quant_args = {}
if USE_4BIT:
    quant_args = dict(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
else:
    quant_args = dict(load_in_8bit=True)

# 初始化处理器与模型
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    **quant_args
)


def generate_caption(image: Image.Image) -> str:
    """
    根据输入图像生成一句描述。
    """
    inputs = processor(images=image, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return caption


def answer(image: Image.Image, question: str) -> str:
    """
    针对图像中的内容回答给定问题，实现简单的视觉问答。
    """
    vqa_prompt = f"Question: {question} Answer:"
    vqa_inputs = processor(images=image, text=vqa_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        vqa_ids = model.generate(
            **vqa_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    vqa_text = processor.batch_decode(vqa_ids, skip_special_tokens=True)[0].strip()
    # 如果模型把提示词一起重复输出，这里进行裁剪
    if vqa_text.startswith(vqa_prompt):
        vqa_text = vqa_text[len(vqa_prompt):].strip()
    return vqa_text


if __name__ == "__main__":
    # 简单示例：对一张图片做描述和问答
    IMG_PATH = "examples/food_01.jpg"
    img = Image.open(IMG_PATH).convert("RGB")
    print("Caption:", generate_caption(img))
    print("Answer:", answer(img, "What is in the image?"))
