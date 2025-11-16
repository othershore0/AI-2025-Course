import os
import json
import csv
import shutil
import random

# ========= 按需修改的配置 =========
#COCO数据集下载后放到对应位置即可
# COCO 图片目录
COCO_IMG_DIR = "/media/e509/本地磁盘1/lx_LLaVA/LLaVA/data/COCO2017/train2017"

# COCO 标注文件
COCO_ANN_FILE = "/media/e509/本地磁盘1/lx_LLaVA/LLaVA/data/COCO2017/annotations/instances_train2017.json"

# 想要当作“已知类”的 COCO 类名（尽量是常作为主对象的大类）
DESIRED_CATEGORIES = [
    "person",
    "dog",
    "cat",
    "car",
    "bus",
    "bicycle",
    "train",
    "truck",
    "airplane",
    "boat",
    "tv",
    "laptop",
]

# 每个类最多抽多少张图
PER_CLASS_MAX_IMAGES = 20

# 主物体面积阈值：bbox面积 / 整图面积，要 >= 这个值才算“主物体”
MIN_MAIN_OBJ_AREA_RATIO = 0.25  # 8%，你可以改成 0.05 或 0.1 试试

# 输出目录（相对于当前 LLaVA 仓库根目录）
OUT_IMG_DIR = "data/known_images"
OUT_META_CSV = "data/known_meta.csv"
OUT_CLASSES_TXT = "data/known_classes.txt"

# 固定随机种子，保证每次选的图一致
random.seed(42)

# ====================================


def load_coco():
    print(f"[INFO] Loading COCO annotations from: {COCO_ANN_FILE}")
    with open(COCO_ANN_FILE, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # image_id -> (file_name, width, height)
    imgid_to_info = {
        img["id"]: (img["file_name"], img["width"], img["height"])
        for img in coco["images"]
    }

    # category 映射
    catid_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    name_to_catid = {cat["name"]: cat["id"] for cat in coco["categories"]}

    # 检查一下想要的类别在不在 COCO 里
    missing = [c for c in DESIRED_CATEGORIES if c not in name_to_catid]
    if missing:
        print("[WARN] These categories are not found in COCO:", missing)

    used_categories = [c for c in DESIRED_CATEGORIES if c in name_to_catid]
    print(f"[INFO] Using {len(used_categories)} categories:", used_categories)

    return coco["annotations"], imgid_to_info, catid_to_name, used_categories


def find_main_object_per_image(annotations, imgid_to_info, catid_to_name):
    """
    对每张图，找到“面积最大的那个物体”（不管是什么类）。
    返回一个 dict: image_id -> {"cat_name": ..., "area_ratio": ...}
    """
    imgid_main = {}  # image_id -> {"cat_name": str, "area_ratio": float}

    for ann in annotations:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        bbox = ann["bbox"]  # [x, y, w, h]

        if img_id not in imgid_to_info:
            continue

        file_name, W, H = imgid_to_info[img_id]
        w, h = bbox[2], bbox[3]
        if W <= 0 or H <= 0 or w <= 0 or h <= 0:
            continue

        area_ratio = (w * h) / (W * H)
        cat_name = catid_to_name.get(cat_id, None)
        if cat_name is None:
            continue

        # 更新这张图的“最大物体”
        if img_id not in imgid_main or area_ratio > imgid_main[img_id]["area_ratio"]:
            imgid_main[img_id] = {
                "cat_name": cat_name,
                "area_ratio": area_ratio,
            }

    print(f"[INFO] Found main object for {len(imgid_main)} images.")
    return imgid_main


def group_images_by_category(imgid_main, imgid_to_info, used_categories):
    """
    只保留：
      - 主物体类别在 used_categories 里
      - 主物体面积比例 >= MIN_MAIN_OBJ_AREA_RATIO
    按类别分组 image_id
    """
    catname_to_imgids = {c: [] for c in used_categories}

    for img_id, info in imgid_main.items():
        cat_name = info["cat_name"]
        area_ratio = info["area_ratio"]
        if cat_name not in used_categories:
            continue
        if area_ratio < MIN_MAIN_OBJ_AREA_RATIO:
            continue

        catname_to_imgids[cat_name].append(img_id)

    for cname, imgids in catname_to_imgids.items():
        print(
            f"[INFO] Category '{cname}' has {len(imgids)} candidate images "
            f"after main-object & area filtering."
        )

    return catname_to_imgids


def sample_and_copy_images(catname_to_imgids, imgid_to_info):
    """
    统一在 OUT_IMG_DIR 下拷贝图片，并生成 known_meta.csv 和 known_classes.txt
    每张图只属于一个类别（我们之前已经是按“主物体”唯一决定的）。
    """
    os.makedirs(OUT_IMG_DIR, exist_ok=True)

    rows = []  # (filename, gt_label)
    used_img_ids = set()

    for cname, imgids in catname_to_imgids.items():
        if not imgids:
            print(f"[WARN] No candidate images for category '{cname}'.")
            continue

        # 打乱之后再取前 PER_CLASS_MAX_IMAGES 张
        random.shuffle(imgids)
        chosen = []
        for img_id in imgids:
            if img_id in used_img_ids:
                continue
            chosen.append(img_id)
            used_img_ids.add(img_id)
            if len(chosen) >= PER_CLASS_MAX_IMAGES:
                break

        print(
            f"[INFO] Category '{cname}': selected {len(chosen)} images "
            f"for final dataset."
        )

        for img_id in chosen:
            file_name, W, H = imgid_to_info[img_id]
            src_path = os.path.join(COCO_IMG_DIR, file_name)
            if not os.path.exists(src_path):
                print(f"[WARN] source image not found: {src_path}")
                continue

            dst_path = os.path.join(OUT_IMG_DIR, file_name)
            shutil.copy2(src_path, dst_path)
            rows.append((file_name, cname))

    print(f"[INFO] Total selected images: {len(rows)}")
    if not rows:
        print("[ERROR] No images selected. Please check settings.")
        return

    # 写 known_meta.csv
    os.makedirs(os.path.dirname(OUT_META_CSV), exist_ok=True)
    with open(OUT_META_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "gt_label"])
        for fname, label in rows:
            writer.writerow([fname, label])
    print(f"[INFO] Saved meta CSV to {OUT_META_CSV}")

    # 写 known_classes.txt（实际用到的类）
    used_class_names = sorted({label for _, label in rows})
    with open(OUT_CLASSES_TXT, "w", encoding="utf-8") as f:
        for cname in used_class_names:
            f.write(cname + "\n")
    print(f"[INFO] Saved known classes to {OUT_CLASSES_TXT}")

    print("[INFO] Done.")


if __name__ == "__main__":
    if not os.path.exists(COCO_IMG_DIR):
        raise FileNotFoundError(f"COCO_IMG_DIR not found: {COCO_IMG_DIR}")
    if not os.path.exists(COCO_ANN_FILE):
        raise FileNotFoundError(f"COCO_ANN_FILE not found: {COCO_ANN_FILE}")

    annotations, imgid_to_info, catid_to_name, used_categories = load_coco()
    imgid_main = find_main_object_per_image(annotations, imgid_to_info, catid_to_name)
    catname_to_imgids = group_images_by_category(imgid_main, imgid_to_info, used_categories)
    sample_and_copy_images(catname_to_imgids, imgid_to_info)
