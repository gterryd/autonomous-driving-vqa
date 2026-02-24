#!/bin/bash
# 一键环境搭建脚本

echo "╔════════════════════════════════════════╗"
echo "║   多模态AI学习环境搭建               ║"
echo "╚════════════════════════════════════════╝"

# 检查设备
echo ""
echo "提示：建议先运行设备检查："
echo "  python tools/check_device.py"
echo ""
read -p "是否继续安装？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# 创建conda环境
echo ""
echo "1. 创建conda环境..."
conda create -n multimodal python=3.10 -y
source activate multimodal

# 安装PyTorch (根据你的CUDA版本修改)
echo "2. 安装PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 安装transformers和相关库
echo "3. 安装transformers..."
pip install transformers datasets accelerate peft

# 安装视觉和工具库
echo "4. 安装其他库..."
pip install pillow opencv-python matplotlib seaborn
pip install scikit-learn pandas numpy
pip install jupyter ipywidgets

# 安装FAISS
echo "5. 安装FAISS..."
pip install faiss-cpu

# 安装Gradio
echo "6. 安装Gradio..."
pip install gradio

# 安装其他工具
pip install tqdm wandb tensorboard

# 测试安装
echo "7. 测试安装..."
python -c "
import torch
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')

import transformers
print(f'Transformers版本: {transformers.__version__}')

print('\\n✅ 所有库安装成功！')
"

echo "=== 环境搭建完成 ==="
echo "使用: conda activate multimodal"








