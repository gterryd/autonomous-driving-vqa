# 项目1：图文检索引擎

## 项目目标
构建一个完整的图文检索系统，展示你的多模态能力

## 技术栈
- CLIP（图文编码）
- FAISS（向量检索）
- Gradio（Web界面）
- Docker（部署）

## 功能需求

### 核心功能
1. **文本搜图**：输入描述，返回相关图片
2. **以图搜图**：上传图片，返回相似图片
3. **图文对比**：计算图片和文本的相似度

### 进阶功能
4. **多模态融合**：文本+图片联合查询
5. **个性化**：用户反馈优化排序
6. **可视化**：注意力图、embedding空间可视化

## 项目结构

```
project1-image-text-retrieval/
├── data/
│   ├── images/              # 图片数据
│   └── captions.json        # 图片描述
├── models/
│   └── clip_finetuned/      # 微调后的模型
├── src/
│   ├── data_loader.py       # 数据加载
│   ├── encoder.py           # CLIP编码器
│   ├── indexer.py           # FAISS索引
│   ├── retrieval.py         # 检索逻辑
│   └── app.py               # Gradio界面
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_clip_finetuning.ipynb
│   └── 03_evaluation.ipynb
├── tests/
│   └── test_retrieval.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## 实现步骤

### Week 1: 数据准备
1. 选择数据集（推荐Flickr30k或COCO）
2. 数据预处理：图片resize、文本清洗
3. 数据分析：可视化分布、统计信息
4. 划分训练/验证/测试集

### Week 2: 模型训练
1. 加载预训练CLIP
2. 在你的数据上微调
3. 超参数调优
4. 评估指标：Recall@K

### Week 3: 系统开发
1. 实现FAISS索引
2. 实现检索API
3. 开发Gradio界面
4. 性能优化

### Week 4: 部署和文档
1. Docker容器化
2. 写README和文档
3. 准备演示视频
4. GitHub发布

## 核心代码

### encoder.py
```python
import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder:
    def __init__(self, model_path="openai/clip-vit-base-patch32"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_path).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model.eval()
    
    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        image_embeds = self.model.get_image_features(**inputs)
        return self._normalize(image_embeds)
    
    @torch.no_grad()
    def encode_texts(self, texts):
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        text_embeds = self.model.get_text_features(**inputs)
        return self._normalize(text_embeds)
    
    def _normalize(self, embeds):
        return embeds / embeds.norm(dim=-1, keepdim=True)
```

### indexer.py
```python
import faiss
import numpy as np
import pickle

class FAISSIndexer:
    def __init__(self, dim=512):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product
        self.metadata = []
    
    def add(self, embeddings, metadata):
        """添加向量到索引"""
        self.index.add(embeddings.astype('float32'))
        self.metadata.extend(metadata)
    
    def search(self, query_embedding, top_k=10):
        """搜索最相似的向量"""
        scores, indices = self.index.search(
            query_embedding.astype('float32'), top_k
        )
        results = [
            {'metadata': self.metadata[idx], 'score': float(score)}
            for idx, score in zip(indices[0], scores[0])
        ]
        return results
    
    def save(self, path):
        """保存索引"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path):
        """加载索引"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'rb') as f:
            self.metadata = pickle.load(f)
```

### app.py（Gradio界面）
```python
import gradio as gr
from PIL import Image
from encoder import CLIPEncoder
from indexer import FAISSIndexer

# 初始化
encoder = CLIPEncoder()
indexer = FAISSIndexer()
indexer.load('./index/')

def search_by_text(query_text, top_k):
    """文本搜图"""
    # 编码文本
    text_embed = encoder.encode_texts([query_text]).cpu().numpy()
    
    # 搜索
    results = indexer.search(text_embed, top_k)
    
    # 返回图片
    images = [Image.open(r['metadata']['path']) for r in results]
    return images

def search_by_image(query_image, top_k):
    """以图搜图"""
    # 编码图片
    image_embed = encoder.encode_images([query_image]).cpu().numpy()
    
    # 搜索
    results = indexer.search(image_embed, top_k)
    
    # 返回图片
    images = [Image.open(r['metadata']['path']) for r in results]
    return images

# Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# 图文检索引擎")
    
    with gr.Tab("文本搜图"):
        text_input = gr.Textbox(label="输入描述")
        text_top_k = gr.Slider(1, 20, value=5, step=1, label="返回数量")
        text_button = gr.Button("搜索")
        text_output = gr.Gallery(label="结果")
        text_button.click(search_by_text, [text_input, text_top_k], text_output)
    
    with gr.Tab("以图搜图"):
        image_input = gr.Image(type="pil", label="上传图片")
        image_top_k = gr.Slider(1, 20, value=5, step=1, label="返回数量")
        image_button = gr.Button("搜索")
        image_output = gr.Gallery(label="结果")
        image_button.click(search_by_image, [image_input, image_top_k], image_output)

demo.launch()
```

## 评估指标

### Recall@K
```python
def recall_at_k(query_embeds, db_embeds, ground_truth, k=10):
    """
    query_embeds: [N, D]
    db_embeds: [M, D]
    ground_truth: [N] 每个query的正确答案索引
    """
    similarity = query_embeds @ db_embeds.T  # [N, M]
    top_k_indices = similarity.topk(k, dim=1).indices  # [N, k]
    
    recall = 0
    for i, gt in enumerate(ground_truth):
        if gt in top_k_indices[i]:
            recall += 1
    
    return recall / len(ground_truth)
```

## 展示要点（面试时讲）

1. **问题定义**：为什么做图文检索？应用场景？
2. **技术选型**：为什么用CLIP？为什么用FAISS？
3. **数据处理**：如何清洗数据？如何处理不平衡？
4. **模型优化**：微调策略？超参数选择？
5. **工程实现**：如何处理大规模数据？如何加速检索？
6. **效果展示**：Recall@K多少？Bad case分析？

## 加分项

- [ ] 支持中文查询（用Chinese CLIP）
- [ ] 增量更新索引
- [ ] 多GPU推理加速
- [ ] A/B测试框架
- [ ] 完整的CI/CD流程
- [ ] 性能监控和日志

## 时间规划

- Week 1: 数据准备 + EDA（10h）
- Week 2: 模型微调（15h）
- Week 3: 系统开发（20h）
- Week 4: 部署和文档（10h）

**总计：55小时**

开始动手！这个项目完成后，你就有小鹏多模态岗位50%的能力了。















