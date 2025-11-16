import json
import re
from collections import Counter


KNOWN_CLASSES_FILE = "data/known_classes.txt"
RESULTS_KNOWN = "results/results_known.jsonl"
RESULTS_PRIVATE = "results/results_private.jsonl"


def load_known_classes():
    classes = []
    with open(KNOWN_CLASSES_FILE, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                classes.append(s.lower())
    print(f"[INFO] loaded {len(classes)} known classes from {KNOWN_CLASSES_FILE}")
    return classes


def normalize(text: str):
    """小写 + 去掉标点 -> 分词"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    return tokens, " ".join(tokens)


def simple_predict(answer: str, known_classes):
    """
    预测：
    - 对 answer 做分词
    - 如果包含某个已知类（作为完整词，或多词短语），则返回该类
    - 否则返回 None（即 UNKNOWN）
    """
    if not answer:
        return None

    tokens, norm_str = normalize(answer)

    for cls in known_classes:
        cls = cls.lower().strip()
        cls_tokens = cls.split()

        if len(cls_tokens) == 1:
            # 单词类别，直接看是否在 tokens 里
            if cls_tokens[0] in tokens:
                return cls
        else:
            # 多词类别，比如 "traffic light"
            if " ".join(cls_tokens) in norm_str:
                return cls

    return None


def eval_split(jsonl_path: str, known_classes, split_name: str):
    total = 0

    pred_known = 0          # 被判为 KNOWN 的数量
    pred_unknown = 0        # 被判为 UNKNOWN 的数量

    overall_correct = 0     # (仅 known split) 预测类 == gt_label
    correct_on_pred_known = 0  # 在判为 KNOWN 的子集中，预测正确数量

    wrong_known_counter = Counter()  # 统计 misclass 的 (gt -> pred)

    for line in open(jsonl_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("split") != split_name:
            continue

        total += 1
        ans = rec.get("answer", "") or ""
        gt = (rec.get("gt_label", "") or "").lower().strip()

        pred_cls = simple_predict(ans, known_classes)

        if pred_cls is None:
            pred_unknown += 1
        else:
            pred_known += 1

        if split_name == "known":
            # 闭集准确率：只有 pred_cls == gt 才算对
            if pred_cls == gt:
                overall_correct += 1
                correct_on_pred_known += 1
            else:
                if pred_cls is not None:
                    wrong_known_counter[(gt, pred_cls)] += 1

    print(f"==== Split: {split_name} ====")
    print(f"Total samples: {total}")
    print(f"Predicted as KNOWN:   {pred_known} ({pred_known/total*100:.1f}%)")
    print(f"Predicted as UNKNOWN: {pred_unknown} ({pred_unknown/total*100:.1f}%)")

    if split_name == "known":
        print(
            f"Closed-set accuracy (pred_class == gt_label): "
            f"{overall_correct/total*100:.1f}%"
        )
        if pred_known > 0:
            print(
                f"Accuracy among samples predicted as KNOWN: "
                f"{correct_on_pred_known/pred_known*100:.1f}%"
            )
        if wrong_known_counter:
            print("Top-5 misclassification pairs (gt -> pred, count):")
            for (gt, pred), c in wrong_known_counter.most_common(5):
                print(f"  {gt} -> {pred}: {c}")
    else:
        print(
            "For private split, higher UNKNOWN ratio generally "
            "means better open-set detection."
        )

    print()


if __name__ == "__main__":
    known_classes = load_known_classes()
    eval_split(RESULTS_KNOWN, known_classes, split_name="known")
    eval_split(RESULTS_PRIVATE, known_classes, split_name="private")
