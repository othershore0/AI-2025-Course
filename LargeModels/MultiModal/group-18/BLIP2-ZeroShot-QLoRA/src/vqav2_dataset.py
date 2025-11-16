"""
VQAv2 Dataset Loader for BLIP-2 fine-tuning.

VQAv2 dataset structure:
- Images: COCO 2014 train/val images (e.g., COCO_train2014_000000XXXXXX.jpg)
- Questions: v2_OpenEnded_mscoco_*_questions.json
- Annotations: v2_mscoco_*_annotations.json
"""

import json
import os
from typing import Dict, List, Optional
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_vqav2_questions(questions_json_path: str) -> Dict[int, Dict]:
    """
    Load VQAv2 questions JSON file.
    
    Format: {"questions": [{"question_id": int, "image_id": int, "question": str}, ...]}
    
    Returns:
        Dict mapping question_id to question data
    """
    with open(questions_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    questions_dict = {}
    for q in data["questions"]:
        questions_dict[q["question_id"]] = q
    
    return questions_dict


def load_vqav2_annotations(annotations_json_path: str) -> Dict[int, Dict]:
    """
    Load VQAv2 annotations JSON file.
    
    Format: {"annotations": [{"question_id": int, "image_id": int, 
                              "answers": [{"answer": str, "answer_confidence": str}, ...],
                              "multiple_choice_answer": str}, ...]}
    
    Returns:
        Dict mapping question_id to annotation data
    """
    with open(annotations_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    annotations_dict = {}
    for ann in data["annotations"]:
        annotations_dict[ann["question_id"]] = ann
    
    return annotations_dict


def get_vqav2_image_path(image_id: int, images_dir: str, split: str = "train") -> Optional[str]:
    """
    Construct VQAv2 image path from image_id.
    
    Format: COCO_train2014_000000XXXXXX.jpg or COCO_val2014_000000XXXXXX.jpg
    
    The function checks both direct path and subdirectory structure:
    - Direct: images_dir/COCO_train2014_000000XXXXXX.jpg
    - Subdirectory: images_dir/train2014/COCO_train2014_000000XXXXXX.jpg
    """
    image_filename = f"COCO_{split}2014_{image_id:012d}.jpg"
    
    # Try subdirectory structure first (most common: train2014/, val2014/)
    subdir_path = os.path.join(images_dir, f"{split}2014", image_filename)
    if os.path.exists(subdir_path):
        return subdir_path
    
    # Try direct path in images_dir
    direct_path = os.path.join(images_dir, image_filename)
    if os.path.exists(direct_path):
        return direct_path
    
    # Try alternative split in subdirectory (in case image is in wrong split folder)
    for alt_split in ["train", "val"]:
        if alt_split != split:
            alt_subdir_path = os.path.join(images_dir, f"{alt_split}2014", f"COCO_{alt_split}2014_{image_id:012d}.jpg")
            if os.path.exists(alt_subdir_path):
                return alt_subdir_path
    
    return None


def get_top_answer(answers: List[Dict]) -> str:
    """
    Get the most frequent answer from VQAv2 annotations.
    If multiple answers have same frequency, return the first one.
    """
    answer_counts = {}
    for ans in answers:
        answer_text = ans["answer"].strip().lower()
        answer_counts[answer_text] = answer_counts.get(answer_text, 0) + 1
    
    if not answer_counts:
        return ""
    
    # Get the most frequent answer
    top_answer = max(answer_counts.items(), key=lambda x: x[1])[0]
    return top_answer


class VQAv2Dataset(Dataset):
    """
    VQAv2 Dataset class for BLIP-2 fine-tuning.
    
    Supports both training and validation splits of VQAv2.
    """
    
    def __init__(
        self,
        questions_json_path: str,
        annotations_json_path: str,
        images_dir: str,
        processor,
        max_len: int = 128,
        split: str = "train",
        use_multiple_choice_answer: bool = False,
    ):
        """
        Args:
            questions_json_path: Path to VQAv2 questions JSON file
            annotations_json_path: Path to VQAv2 annotations JSON file
            images_dir: Directory containing COCO images
            processor: BLIP-2 processor
            max_len: Maximum sequence length
            split: "train" or "val"
            use_multiple_choice_answer: If True, use multiple_choice_answer field;
                                       If False, use most frequent answer from answers list
        """
        super().__init__()
        
        self.processor = processor
        self.max_len = max_len
        self.images_dir = images_dir
        self.split = split
        self.use_multiple_choice_answer = use_multiple_choice_answer
        
        # Load questions and annotations
        questions_dict = load_vqav2_questions(questions_json_path)
        annotations_dict = load_vqav2_annotations(annotations_json_path)
        
        # Build dataset items: list of (question_id, image_id, question, answer, image_path)
        self.items = []
        for question_id, question_data in questions_dict.items():
            if question_id not in annotations_dict:
                continue  # Skip if no annotation
            
            annotation = annotations_dict[question_id]
            image_id = question_data["image_id"]
            
            # Get answer
            if use_multiple_choice_answer and "multiple_choice_answer" in annotation:
                answer = annotation["multiple_choice_answer"]
            else:
                answer = get_top_answer(annotation.get("answers", []))
            
            # Get image path
            image_path = get_vqav2_image_path(image_id, images_dir, split)
            if image_path is None:
                continue  # Skip if image not found
            
            self.items.append({
                "question_id": question_id,
                "image_id": image_id,
                "question": question_data["question"],
                "answer": answer,
                "image_path": image_path,
            })
        
        print(f"Loaded {len(self.items)} VQAv2 samples from {split} split")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        rec = self.items[idx]
        
        # Load image
        try:
            image = Image.open(rec["image_path"]).convert("RGB")
        except Exception as e:
            # Return a dummy image if loading fails
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))
            print(f"Warning: Failed to load image {rec['image_path']}: {e}")
        
        # Build prompt and target text
        # Use same prompt format as evaluation for consistency
        prompt = f"Question: {rec['question']}\nAnswer with one or two words:"
        target = rec["answer"]
        
        # Combine prompt and target, and mask the prompt in labels
        text_full = prompt + " " + target + self.processor.tokenizer.eos_token
        enc_full = self.processor.tokenizer(
            text_full,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc_prompt = self.processor.tokenizer(
            prompt,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = enc_full.input_ids[0]
        attn_mask = enc_full.attention_mask[0]
        labels = input_ids.clone()
        
        # Mask out the prompt tokens in labels
        prompt_len = (enc_prompt.attention_mask[0] == 1).sum()
        labels[:prompt_len] = -100
        
        # Process image to pixel values
        pixel = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        
        return {
            "pixel_values": pixel,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }

