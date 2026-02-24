import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

# ============================================================
# 第1部分：加载模型
# 和 predict.ipynb 里一样，加载训练好的模型
# ============================================================

model = models.resnet18()
model.fc = nn.Linear(512, 7)
model.load_state_dict(torch.load('emotion_model.pth'))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 预处理（和训练时一致）
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

print("模型加载完成")

# ============================================================
# 第2部分：定义预测函数
# Gradio 会把用户上传的图片传给这个函数
# 函数返回一个字典：{情绪名: 概率}，Gradio 自动显示成柱状图
# ============================================================

def predict_emotion(image):
    """
    输入：PIL 图片（Gradio 自动把上传的图片转成 PIL 格式）
    输出：字典，key 是情绪名，value 是概率
    """
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)[0]

    # 7种情绪（按字母顺序，和训练时一致）
    emotions = ['😠 生气', '🤢 厌恶', '😨 害怕', '😊 开心', '😐 平静', '😢 难过', '😲 惊讶']

    return {emotions[i]: float(probs[i]) for i in range(7)}

# ============================================================
# 第3部分：创建网页界面
# gr.Interface 是最简单的方式：指定函数、输入类型、输出类型
# ============================================================

demo = gr.Interface(
    fn=predict_emotion,                              # 调用哪个函数
    inputs=gr.Image(type="pil", label="上传人脸照片"),  # 输入：图片
    outputs=gr.Label(label="情绪识别结果", num_top_classes=7),  # 输出：标签（显示概率）
    title="😊 人脸情绪识别系统",
    description="上传一张人脸照片，AI 自动识别情绪（7种：开心、难过、生气、惊讶、害怕、厌恶、平静）",
    examples=[                                       # 示例图片（可选，方便演示）
        ["./data/emotion/val/happy/happy_1.jpg"],
        ["./data/emotion/val/sad/sadshushu.jpg"],
    ]
)

# ============================================================
# 第4部分：启动
# ============================================================

if __name__ == "__main__":
    demo.launch()