# Day 8-9: CLIP（2.13-2.14）

## Day 8 (2.13): CLIP原理

### 对比学习
```python
# 伪代码
image_embeds = image_encoder(images)  # [N, D]
text_embeds = text_encoder(texts)     # [N, D]

# 归一化
image_embeds = F.normalize(image_embeds, dim=-1)
text_embeds = F.normalize(text_embeds, dim=-1)

# 相似度矩阵
logits = image_embeds @ text_embeds.T  # [N, N]

# 对角线为正样本
labels = torch.arange(N)
loss_i2t = F.cross_entropy(logits, labels)
loss_t2i = F.cross_entropy(logits.T, labels)
loss = (loss_i2t + loss_t2i) / 2
```

### CLIP架构
```
Image → ViT/ResNet → Image Embedding
Text  → Transformer → Text Embedding
        ↓
    计算相似度 → 对比学习损失
```

---

## Day 9 (2.14): CLIP实战

### 零样本分类
```python
from transformers import CLIPModel, CLIPProcessor
from PIL import Image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open("dog.jpg")
texts = ["a dog", "a cat", "a bird"]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
outputs = model(**inputs)

probs = outputs.logits_per_image.softmax(dim=1)
print(probs)
```

### 图文检索
```python
# 编码图片
images = [img1, img2, img3]
image_inputs = processor(images=images, return_tensors="pt")

with torch.no_grad():
    image_embeds = model.get_image_features(**image_inputs)
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

# 编码文本
texts = ["a dog", "a cat"]
text_inputs = processor(text=texts, return_tensors="pt", padding=True)

with torch.no_grad():
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

# 相似度
similarity = image_embeds @ text_embeds.T
print(similarity)
```

### FAISS索引
```python
import faiss
import numpy as np

# 图像embedding
embeds = image_embeds.cpu().numpy().astype('float32')

# 创建索引
index = faiss.IndexFlatIP(embeds.shape[1])
index.add(embeds)

# 搜索
query_embed = text_embeds[0].cpu().numpy().astype('float32').reshape(1, -1)
scores, indices = index.search(query_embed, k=5)

print(f"Top 5 similar images: {indices}")
```

---

## 完整检索系统
```python
class ImageSearchEngine:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index = None
        self.image_paths = []
    
    def build_index(self, image_folder):
        import os
        embeddings = []
        
        for filename in os.listdir(image_folder):
            if not filename.endswith(('.jpg', '.png')):
                continue
            
            path = os.path.join(image_folder, filename)
            image = Image.open(path)
            
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embed = self.model.get_image_features(**inputs)
                embed = embed / embed.norm(dim=-1, keepdim=True)
            
            embeddings.append(embed.cpu().numpy())
            self.image_paths.append(path)
        
        embeds = np.vstack(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeds.shape[1])
        self.index.add(embeds)
    
    def search(self, query_text, top_k=5):
        inputs = self.processor(text=query_text, return_tensors="pt")
        with torch.no_grad():
            embed = self.model.get_text_features(**inputs)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        
        embed = embed.cpu().numpy().astype('float32')
        scores, indices = self.index.search(embed, top_k)
        
        return [(self.image_paths[idx], score) for idx, score in zip(indices[0], scores[0])]
```

---

## 面试题

**Q1: CLIP训练数据？**
4亿图文对

**Q2: CLIP为什么不需要标注？**
使用图文自然对应关系，自监督学习

**Q3: CLIP vs 传统图像分类？**
CLIP零样本，传统需要训练

**Q4: 如何提升CLIP性能？**
数据增强、温度参数调优、hard negative mining

---

## 检查点

- [ ] 理解对比学习
- [ ] 能用CLIP做零样本分类
- [ ] 能实现图文检索
- [ ] 会用FAISS

**CLIP是多模态的核心！**



