# COCO 2014 图像目录

请将 COCO 2014 图像解压到此目录的相应子目录中。

## 目录结构

```
data/images/
├── train2014/          # 解压 train2014.zip 到此目录
│   ├── COCO_train2014_000000000009.jpg
│   ├── COCO_train2014_000000000025.jpg
│   └── ...
└── val2014/            # 解压 val2014.zip 到此目录
    ├── COCO_val2014_000000000139.jpg
    └── ...
```

## 解压命令

```bash
# 假设 ZIP 文件在当前目录
unzip train2014.zip -d data/images/train2014/
unzip val2014.zip -d data/images/val2014/
```

或者直接解压到当前目录，然后移动文件：

```bash
unzip train2014.zip
mv train2014/* data/images/train2014/
unzip val2014.zip
mv val2014/* data/images/val2014/
```

