# 😊 人脸情绪识别系统

## 项目介绍

使用深度学习识别人脸照片中的情绪表情。上传一张人脸照片，自动输出情绪类别和概率。

## 支持的情绪（7种）

| 情绪 | 英文 |
|------|------|
| 😊 开心 | Happy |
| 😢 难过 | Sad |
| 😠 生气 | Angry |
| 😐 平静 | Neutral |
| 😨 害怕 | Fear |
| 🤢 厌恶 | Disgust |
| 😲 惊讶 | Surprise |

## 技术栈

- **模型**：ResNet18（ImageNet 预训练 + 微调）
- **框架**：PyTorch 2.5
- **数据集**：FER2013（约28000张训练，7000张验证）
- **界面**：Gradio
- **加速**：CUDA（GPU）

## 训练效果

- 最佳验证准确率：62.51%
- 训练轮数：10 Epoch
- FER2013 人类标注准确率约 65%，模型表现接近人类水平

## 安装

```bash
conda activate multimodal
pip install torch torchvision gradio
```

## 使用

```bash
python app.py
```

浏览器自动打开，上传人脸照片即可识别。

## 项目结构

```
├── train.ipynb         # 训练代码
├── predict.ipynb       # 测试代码
├── app.py              # 网页应用
├── emotion_model.pth   # 训练好的模型
├── README.md           # 项目说明
└── data/emotion/       # 数据集
```

## 作者

[gterryd]