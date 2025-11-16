from PIL import Image
from src.blip2_inference import generate_caption, answer

img_path = "baseline_images/pic_01.jpg"  
img = Image.open(img_path).convert("RGB")

print("Caption:", generate_caption(img))
print("Answer:", answer(img, "What is in the image?"))

