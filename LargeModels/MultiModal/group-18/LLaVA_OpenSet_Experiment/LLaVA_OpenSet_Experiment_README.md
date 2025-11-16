# 🧠 LLaVA 开放集识别实验 (Open-Set Recognition with LLaVA)

> 人工智能原理与应用课程 · 第 18 组 · 陆旭

本实验基于 [LLaVA (Large Language and Vision Assistant)](https://github.com/haotian-liu/LLaVA)，旨在探索多模态大模型在 **开放集场景下的识别能力**，即区分「已知类」与「未知类（私有场景）」的能力。

---

## 🧩 实验整体流程

### **1️⃣ 环境准备**

```bash
# 创建虚拟环境
conda create -n llava python=3.10
conda activate llava

# 安装依赖
pip install -e .
```

> ⚠️ 建议运行目录为：
> `/media/e509/本地磁盘1/lx_LLaVA/LLaVA`

---

### **2️⃣ 下载模型权重与编码器**

实验需要两个主要模型：

| 模型名称                  | 说明             | 下载后目录                          |
| --------------------- | -------------- | ------------------------------ |
| **LLaVA v1.5-7B**     | 多模态大模型主体       | `./llava-v1.5-7b`              |
| **CLIP ViT-L/14-336** | 视觉编码器，用于图像特征提取 | `./clip-vit-large-patch14-336` |

```bash
# 下载 LLaVA 模型
huggingface-cli download liuhaotian/llava-v1.5-7b --local-dir ./llava-v1.5-7b

# 下载 CLIP 视觉编码器
huggingface-cli download openai/clip-vit-large-patch14-336 --local-dir ./clip-vit-large-patch14-336
```

---

### **3️⃣ 单图验证（确认环境正常）**

```bash
python -m llava.serve.cli   --model-path "./llava-v1.5-7b"   --image-file "/media/e509/本地磁盘1/lx_LLaVA/LLaVA/1.JPG"   --load-4bit
```

> 若模型能输出结果说明加载正常。

---

### **4️⃣ 数据准备**

#### (1) COCO 已知类数据

路径：

```
/media/e509/本地磁盘1/lx_LLaVA/data/COCO2017/train2017
```

运行数据提取脚本：

```bash
python make_known_from_coco.py
```

输出文件包括：

| 文件名                      | 说明           |
| ------------------------ | ------------ |
| `data/known_images/`     | 筛选出的主物体明显的图片 |
| `data/known_meta.csv`    | 每张图片的路径与标签   |
| `data/known_classes.txt` | 12 个已知类别列表   |

---

#### (2) 私有类数据采集

采集你自己的 **未知场景数据**，包括：

- 宿舍整体场景 (`dorm_room`)
- 教室整体场景 (`classroom_overall`)
- 楼道公告栏/海报墙 (`notice_board`)

放入路径：

```
LLaVA/data/private_images/
```

手动创建一个标签文件 `data/private_labels.csv`：

```csv
filename,label
1.png,dorm_room
10.png,dorm_room
11.png,classroom_overall
12.png,classroom_overall
13.png,office_desk
14.png,notice_board
```

---

### **5️⃣ 批量问答推理**

统一英文模板（避免语言偏差）：

```
You are a vision classifier. Look at the image and determine the single main object.
If it clearly belongs to one of these categories:
person, dog, cat, car, bus, bicycle, train, truck, airplane, boat, tv, laptop,
then answer with exactly that one word (for example: 'dog').
If it does not fit any of those categories, answer with a short English noun phrase 
describing the main object or scene, such as 'dorm room', 'classroom', 'office desk', or 
'notice board'. Do not use full sentences and do not add any extra words.
```

运行批量推理：

```bash
python run_llava_batch.py
```

输出文件：

| 文件名                             | 含义           |
| ------------------------------- | ------------ |
| `results/results_known.jsonl`   | COCO 已知类预测结果 |
| `results/results_private.jsonl` | 私有类预测结果      |

---

### **6️⃣ 开放集评估**

运行评估脚本：

```bash
python open_set_eval_simple.py
```

输出示例：

```
==== Split: known ====
Total samples: 240
Closed-set accuracy: 75.4%
Predicted as Known: 85.0%
Accuracy among Known: 88.7%

==== Split: private ====
Total samples: 30
Predicted as Unknown: 86.7%
```

> - “Closed-set accuracy” 表示模型在已知类上的分类精度  
> - “Predicted as Unknown” 表示模型能正确拒绝未知类的比例（越高越好）

---

### **7️⃣ 结果分析与总结**

- 模型：`LLaVA v1.5-7B`
- 视觉编码器：`CLIP ViT-L/14-336`
- 已知类：COCO12类 (person, dog, cat, car, bus, bicycle, train, truck, airplane, boat, tv, laptop)
- 私有类：3类真实场景（宿舍、教室、公告栏）
- 问题模板：统一英文问题，避免多语言语义偏差
- 评估方法：
  -基于关键词匹配（known vs unknown）

---

📁 **最终生成文件结构示例**

```
LLaVA/
├── data/
│   ├── known_images/
│   ├── private_images/
│   ├── known_meta.csv
│   ├── known_classes.txt
│   ├── private_labels.csv
├── results/
│   ├── results_known.jsonl
│   ├── results_private.jsonl
├── make_known_from_coco.py
├── run_llava_batch.py
├── open_set_eval_simple.py
└── README.md
```

---

---

### 作者署名

**SY2503513-陆旭**
如有问题请联系：sdluxu2003@163.com

---

### **8️⃣ 引用说明**

本实验基于以下项目扩展实现：

> [LLaVA: Large Language and Vision Assistant (Liu et al., 2023)](https://github.com/haotian-liu/LLaVA)

请在论文或报告中引用原始工作：

```bibtex
@article{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and Li, Chunyuan and others},
  journal={arXiv preprint arXiv:2304.08485},
  year={2023}
}
```
