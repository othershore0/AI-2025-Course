#!/usr/bin/env python3
"""
Create baseline.jsonl using zero-shot inference (same as mini_infer.py but for multiple images).
This uses the exact same approach as the original author.
"""
import json
import glob
import os
from PIL import Image
from src.blip2_inference import generate_caption, answer

# Use the same images as original author
image_files = sorted(glob.glob("baseline_images/pic_*.jpg"))[:10]

os.makedirs("outputs", exist_ok=True)

with open("outputs/baseline.jsonl", "w", encoding="utf-8") as f:
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        caption = generate_caption(img)
        answer_text = answer(img, "What is in the image?")
        
        record = {
            "image_path": img_path,
            "caption": caption,
            "question": "What is in the image?",
            "answer": answer_text
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✓ {img_path}")

print(f"\nSaved to outputs/baseline.jsonl")

