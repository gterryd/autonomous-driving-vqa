# 数据集下载脚本

import os
from pathlib import Path

def download_cifar10():
    """下载CIFAR-10数据集"""
    print("下载CIFAR-10...")
    from torchvision import datasets
    datasets.CIFAR10(root='./data', train=True, download=True)
    datasets.CIFAR10(root='./data', train=False, download=True)
    print("✅ CIFAR-10下载完成")

def download_mnist():
    """下载MNIST数据集"""
    print("下载MNIST...")
    from torchvision import datasets
    datasets.MNIST(root='./data', train=True, download=True)
    datasets.MNIST(root='./data', train=False, download=True)
    print("✅ MNIST下载完成")

def download_clip_model():
    """下载CLIP模型"""
    print("下载CLIP模型...")
    from transformers import CLIPModel, CLIPProcessor
    
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 保存到本地
    save_path = "./models/clip-vit-base"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"✅ CLIP模型下载完成，保存在 {save_path}")

def download_blip_model():
    """下载BLIP模型"""
    print("下载BLIP模型...")
    from transformers import BlipForConditionalGeneration, BlipProcessor
    
    model_name = "Salesforce/blip-image-captioning-base"
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    processor = BlipProcessor.from_pretrained(model_name)
    
    save_path = "./models/blip-base"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    
    print(f"✅ BLIP模型下载完成，保存在 {save_path}")

def download_coco_captions():
    """下载COCO Captions（小样本）"""
    print("下载COCO Captions...")
    from datasets import load_dataset
    
    # 只下载验证集的前1000个样本
    dataset = load_dataset("HuggingFaceM4/COCO", split="validation[:1000]")
    dataset.save_to_disk("./data/coco_val_1k")
    
    print("✅ COCO Captions下载完成")

if __name__ == '__main__':
    print("=== 数据集和模型下载 ===\n")
    
    # 创建目录
    Path("./data").mkdir(exist_ok=True)
    Path("./models").mkdir(exist_ok=True)
    
    # 选择下载
    print("选择要下载的内容：")
    print("1. CIFAR-10数据集")
    print("2. MNIST数据集")
    print("3. CLIP模型")
    print("4. BLIP模型")
    print("5. COCO Captions（小样本）")
    print("6. 全部下载")
    
    choice = input("\n输入选项（1-6）: ")
    
    if choice == '1':
        download_cifar10()
    elif choice == '2':
        download_mnist()
    elif choice == '3':
        download_clip_model()
    elif choice == '4':
        download_blip_model()
    elif choice == '5':
        download_coco_captions()
    elif choice == '6':
        download_cifar10()
        download_mnist()
        download_clip_model()
        download_blip_model()
        download_coco_captions()
    else:
        print("无效选项")
    
    print("\n=== 下载完成 ===")















