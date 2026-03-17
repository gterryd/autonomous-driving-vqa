# 自动驾驶场景视觉问答系统（autonomous-driving-vqa）

基于 CLIP + MLP 分类器，在 nuScenes mini 数据集上训练的自动驾驶场景视觉问答（VQA）系统。

## 实验结果

| 指标 | 数值 |
|------|------|
| 最佳验证准确率 | **95.8%** |
| 训练轮数 | 20 epochs |
| 数据集 | nuScenes mini（404张图，6464条问答） |
| 答案类别数 | 7 |

## Demo 演示

![demo](assets/demo.png)

```bash
python app.py
# 浏览器打开 http://127.0.0.1:7860
```

> **注意：问题必须用英文输入**，例如 `How many cars are there?`
> 原因：CLIP 文本编码器基于英文预训练，中文输入会导致特征质量下降。

## 支持的问题类型

模型支持对以下 8 类目标进行问答：

| 目标类别 | 问题类型 | 示例 |
|----------|----------|------|
| car, truck, bus, motorcycle, bicycle | 计数 | `How many cars are there?` |
| pedestrian, traffic cone, barrier | 存在性 | `Is there a pedestrian in the scene?` |

**答案词表：** `0`, `1`, `2`, `3`, `more than 3`, `yes`, `no`

## 模型结构

```
CLIP ViT-B/32（冻结，不参与训练）
    ├── 图像编码器 → 512维
    └── 文本编码器 → 512维
              ↓ 拼接 → 1024维
         MLP 分类头
         1024 → 512 → 256 → 7
```

- 主干网络：`openai/clip-vit-base-patch32`（权重冻结）
- 只训练 MLP 分类头
- 优化器：Adam，lr=1e-3
- 损失函数：CrossEntropyLoss

## 数据集

- **图片来源**：nuScenes mini，10个驾驶场景，404张前置摄像头关键帧
- **问答生成**：基于 `sample_annotation.json` 真实3D标注自动生成
- **数据划分**：80% 训练（5168条）/ 20% 验证（1296条）

## 环境配置

```bash
conda create -n multimodal python=3.10
conda activate multimodal
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers gradio pillow tqdm
```

下载 [nuScenes mini](https://www.nuscenes.org/nuscenes)，解压到 `data/nuscenes/v1.0-mini/`。

## 训练

模型权重文件（`driving_vqa_v2_best.pth`，约580MB）未包含在仓库中，需自行训练生成：

```bash
python prepare_data_v2.py   # 从标注生成问答对
python train.py             # 训练20轮（RTX 4060约40分钟），自动保存最佳权重
```

## 运行环境

- Python 3.10，PyTorch 2.6.0+cu124，transformers 5.0.0
- NVIDIA RTX 4060 Laptop GPU（8GB显存）
