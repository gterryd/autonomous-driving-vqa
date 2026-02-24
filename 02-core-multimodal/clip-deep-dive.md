# CLIP: 多模态的基石

## 为什么学CLIP
- 小鹏岗位明确要求
- 多模态模型的核心思想
- 图文检索、分类、生成的基础

## 1. CLIP原理

### 核心思想
**对比学习：让匹配的图文对在embedding空间中靠近**

### 架构
```
Image                    Text
  ↓                       ↓
Image Encoder          Text Encoder
(ViT/ResNet)           (Transformer)
  ↓                       ↓
Image Embedding        Text Embedding
  ↓                       ↓
    计算相似度矩阵
  ↓                       ↓
    对比学习损失
```

### 训练过程
```python
# 伪代码
images = [img1, img2, ..., imgN]
texts = [txt1, txt2, ..., txtN]

# 编码
image_embeds = image_encoder(images)  # [N, D]
text_embeds = text_encoder(texts)     # [N, D]

# L2归一化
image_embeds = F.normalize(image_embeds, dim=-1)
text_embeds = F.normalize(text_embeds, dim=-1)

# 计算相似度矩阵
logits = image_embeds @ text_embeds.T * temperature  # [N, N]

# 对比学习损失
labels = torch.arange(N)  # 对角线为正样本
loss_i2t = F.cross_entropy(logits, labels)       # 图->文
loss_t2i = F.cross_entropy(logits.T, labels)     # 文->图
loss = (loss_i2t + loss_t2i) / 2
```

## 2. 使用CLIP（HuggingFace）

### 安装
```bash
pip install transformers pillow
```

### 零样本图像分类
```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 加载图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# 定义候选类别
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]

# 处理输入
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

# 前向传播
outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # [1, 3]
probs = logits_per_image.softmax(dim=1)      # [1, 3]

# 打印结果
for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob:.4f}")
```

### 图文检索
```python
import torch

# 假设有多张图片和多个文本
images = [image1, image2, image3]
texts = ["a cat", "a dog", "a bird"]

# 编码
image_inputs = processor(images=images, return_tensors="pt")
text_inputs = processor(text=texts, return_tensors="pt", padding=True)

with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)
    text_embeds = model.get_text_features(**text_inputs)
    
    # 归一化
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    
    # 相似度矩阵
    similarity = image_embeds @ text_embeds.T  # [3, 3]
    print(similarity)

# 图->文检索：给定图片找最相关文本
image_idx = 0
top_text_idx = similarity[image_idx].argmax()
print(f"图片{image_idx}最匹配文本: {texts[top_text_idx]}")

# 文->图检索：给定文本找最相关图片
text_idx = 0
top_image_idx = similarity[:, text_idx].argmax()
print(f"文本{text_idx}最匹配图片: {top_image_idx}")
```

## 3. 微调CLIP（重要）

### 场景
你有一批图文对数据，想让CLIP更适应你的领域

### 代码
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

# 自定义数据集
class ImageTextDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        text = self.captions[idx]
        
        # 处理
        inputs = self.processor(
            text=text, 
            images=image, 
            return_tensors="pt",
            padding='max_length',
            truncation=True
        )
        
        # 移除batch维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs

# 对比学习损失
class CLIPLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, image_embeds, text_embeds):
        # 归一化
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        # 相似度
        logits = (image_embeds @ text_embeds.T) / self.temperature
        
        # 标签
        batch_size = image_embeds.shape[0]
        labels = torch.arange(batch_size).to(logits.device)
        
        # 双向损失
        loss_i2t = nn.functional.cross_entropy(logits, labels)
        loss_t2i = nn.functional.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2

# 训练
def train_clip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.to(device)
    
    # 准备数据（示例）
    image_paths = ['img1.jpg', 'img2.jpg', ...]
    captions = ['a cat', 'a dog', ...]
    dataset = ImageTextDataset(image_paths, captions, processor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 优化器和损失
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = CLIPLoss()
    
    # 训练循环
    model.train()
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            # 移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向
            outputs = model(**batch)
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # 损失
            loss = criterion(image_embeds, text_embeds)
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')
    
    # 保存
    model.save_pretrained('./finetuned_clip')

if __name__ == '__main__':
    train_clip()
```

## 4. CLIP + FAISS 图像检索系统

### 完整流程
```python
import faiss
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os

class ImageSearchEngine:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.index = None
        self.image_paths = []
    
    def encode_images(self, image_paths):
        """批量编码图片"""
        embeddings = []
        for img_path in image_paths:
            image = Image.open(img_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                embed = self.model.get_image_features(**inputs)
                embed = embed / embed.norm(dim=-1, keepdim=True)
            
            embeddings.append(embed.cpu().numpy())
        
        return np.vstack(embeddings)  # [N, D]
    
    def build_index(self, image_folder):
        """构建FAISS索引"""
        # 获取所有图片路径
        self.image_paths = [
            os.path.join(image_folder, f) 
            for f in os.listdir(image_folder) 
            if f.endswith(('.jpg', '.png'))
        ]
        
        # 编码
        print(f"编码 {len(self.image_paths)} 张图片...")
        embeddings = self.encode_images(self.image_paths)
        
        # 创建FAISS索引
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (余弦相似度)
        self.index.add(embeddings.astype('float32'))
        
        print("索引构建完成！")
    
    def search_by_text(self, query_text, top_k=5):
        """文本搜图"""
        # 编码文本
        inputs = self.processor(text=query_text, return_tensors="pt")
        with torch.no_grad():
            text_embed = self.model.get_text_features(**inputs)
            text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        
        # 搜索
        text_embed = text_embed.cpu().numpy().astype('float32')
        scores, indices = self.index.search(text_embed, top_k)
        
        # 返回结果
        results = [
            (self.image_paths[idx], score) 
            for idx, score in zip(indices[0], scores[0])
        ]
        return results
    
    def search_by_image(self, query_image_path, top_k=5):
        """以图搜图"""
        # 编码图片
        query_embed = self.encode_images([query_image_path])
        
        # 搜索
        scores, indices = self.index.search(query_embed.astype('float32'), top_k)
        
        # 返回结果
        results = [
            (self.image_paths[idx], score) 
            for idx, score in zip(indices[0], scores[0])
        ]
        return results

# 使用
engine = ImageSearchEngine()
engine.build_index('./my_images/')

# 文本搜图
results = engine.search_by_text("a cute cat", top_k=5)
for img_path, score in results:
    print(f"{img_path}: {score:.4f}")

# 以图搜图
results = engine.search_by_image('./query.jpg', top_k=5)
for img_path, score in results:
    print(f"{img_path}: {score:.4f}")
```

## 5. 面试题

**Q1: CLIP的训练数据规模？**  
A: 4亿图文对，从互联网收集

**Q2: CLIP为什么不需要标注？**  
A: 使用图文对的自然对应关系，自监督学习

**Q3: CLIP的局限性？**  
A: 细粒度分类较弱；依赖文本质量；计算量大

**Q4: 如何提升CLIP性能？**  
A: 数据增强；温度参数调优；hard negative mining

**Q5: CLIP vs 传统图像分类？**  
A: CLIP零样本，传统需要训练；CLIP理解语义，传统只分类

## 6. 作业

**任务1：零样本分类**
- 下载10张不同类别的图片
- 用CLIP做零样本分类
- 分析错误案例

**任务2：图像检索系统**
- 收集100张图片
- 实现CLIP+FAISS检索
- 测试文本查询和图像查询

**任务3：微调CLIP**
- 找一个小数据集（如Flickr8k）
- 微调CLIP
- 对比微调前后性能

## 检查点

- ✅ 理解CLIP对比学习原理
- ✅ 能用CLIP做零样本分类
- ✅ 能实现图文检索系统
- ✅ 能微调CLIP

这是多模态岗位的核心技能！















