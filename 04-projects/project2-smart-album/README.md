# 项目2：智能相册助手

## 项目亮点
- 实用性强，每个人都需要
- 技术全面：分类、检索、生成、聚类
- 数据获取简单，用自己的照片
- 演示效果好，面试加分

## 核心功能

### 1. 智能分类
- 场景识别（风景、美食、人物、动物等）
- 时间地点自动标注
- 相似照片聚类

### 2. 智能搜索
- 文本搜图："海边日落的照片"
- 以图搜图：找相似照片
- 多条件组合搜索

### 3. 自动描述
- 生成照片caption
- 提取关键信息
- 生成相册故事

### 4. 人脸管理
- 人脸检测和聚类
- 人物识别（家人、朋友）
- 按人物搜索

### 5. 智能推荐
- 精选照片推荐
- 制作回忆视频
- 生成相册集

## 技术栈

```
图像分类    → ResNet/ViT
场景识别    → CLIP
图文检索    → CLIP + FAISS
Caption生成 → BLIP
人脸识别    → MTCNN + FaceNet
推荐系统    → 协同过滤
前端界面    → Gradio/Streamlit
```

## 数据集

### 公开数据集（训练用）
1. **COCO**: 通用场景
2. **Places365**: 场景分类
3. **CelebA**: 人脸数据
4. **Google Landmarks**: 地标识别

### 自建数据（测试用）
- 手机相册导出
- 朋友圈照片
- 网上下载测试图片

## 实现步骤

### Week 1: 基础功能（4-5天）

**Day 1-2: 场景分类**
```python
from transformers import CLIPModel, CLIPProcessor
import torch

class SceneClassifier:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # 定义场景类别
        self.categories = [
            "outdoor landscape", "indoor room", "food and dining",
            "people and portrait", "animal and pet", "city and street",
            "beach and ocean", "mountain and nature", "building and architecture",
            "night scene", "sports and activity", "celebration and party"
        ]
    
    def classify(self, image):
        """分类单张图片"""
        texts = [f"a photo of {cat}" for cat in self.categories]
        
        inputs = self.processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        outputs = self.model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]
        
        # 返回top3类别
        top3_idx = probs.topk(3).indices
        results = [
            {
                'category': self.categories[idx],
                'confidence': probs[idx].item()
            }
            for idx in top3_idx
        ]
        
        return results
```

**Day 3-4: 图文检索**
```python
import faiss
import numpy as np
from PIL import Image
import pickle

class PhotoSearchEngine:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.index = None
        self.photo_metadata = []
    
    def build_index(self, photo_folder):
        """构建照片索引"""
        import os
        from tqdm import tqdm
        
        embeddings = []
        photo_paths = []
        
        for filename in tqdm(os.listdir(photo_folder)):
            if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            path = os.path.join(photo_folder, filename)
            image = Image.open(path).convert('RGB')
            
            # 提取embedding
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                embed = self.clip_model.get_image_features(**inputs)
                embed = embed / embed.norm(dim=-1, keepdim=True)
            
            embeddings.append(embed.cpu().numpy())
            
            # 保存元数据
            self.photo_metadata.append({
                'path': path,
                'filename': filename,
                'timestamp': os.path.getmtime(path)
            })
        
        # 创建FAISS索引
        embeddings = np.vstack(embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        
        print(f"索引构建完成！共 {len(self.photo_metadata)} 张照片")
    
    def search_by_text(self, query, top_k=10):
        """文本搜图"""
        inputs = self.processor(text=query, return_tensors="pt")
        with torch.no_grad():
            embed = self.clip_model.get_text_features(**inputs)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        
        embed = embed.cpu().numpy().astype('float32')
        scores, indices = self.index.search(embed, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                **self.photo_metadata[idx],
                'score': float(score)
            })
        
        return results
    
    def search_by_image(self, query_image, top_k=10):
        """以图搜图"""
        inputs = self.processor(images=query_image, return_tensors="pt")
        with torch.no_grad():
            embed = self.clip_model.get_image_features(**inputs)
            embed = embed / embed.norm(dim=-1, keepdim=True)
        
        embed = embed.cpu().numpy().astype('float32')
        scores, indices = self.index.search(embed, top_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.photo_metadata):
                results.append({
                    **self.photo_metadata[idx],
                    'score': float(score)
                })
        
        return results
    
    def save_index(self, path):
        """保存索引"""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'wb') as f:
            pickle.dump(self.photo_metadata, f)
    
    def load_index(self, path):
        """加载索引"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/metadata.pkl", 'rb') as f:
            self.photo_metadata = pickle.load(f)
```

### Week 2: 高级功能（4-5天）

**Day 5-6: Caption生成**
```python
from transformers import BlipForConditionalGeneration, BlipProcessor

class PhotoCaptioner:
    def __init__(self):
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
    
    def generate_caption(self, image):
        """生成照片描述"""
        inputs = self.processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
        
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def batch_caption(self, image_folder, output_file):
        """批量生成描述"""
        import json
        
        captions = {}
        for filename in os.listdir(image_folder):
            if not filename.lower().endswith(('.jpg', '.png')):
                continue
            
            path = os.path.join(image_folder, filename)
            image = Image.open(path).convert('RGB')
            caption = self.generate_caption(image)
            captions[filename] = caption
            print(f"{filename}: {caption}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(captions, f, indent=2, ensure_ascii=False)
```

**Day 7-8: 人脸聚类**
```python
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.cluster import DBSCAN
import numpy as np

class FaceManager:
    def __init__(self):
        self.mtcnn = MTCNN(keep_all=True)
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval()
    
    def detect_faces(self, image):
        """检测人脸"""
        boxes, probs = self.mtcnn.detect(image)
        
        if boxes is None:
            return []
        
        faces = []
        for box, prob in zip(boxes, probs):
            if prob > 0.9:  # 置信度阈值
                faces.append({
                    'box': box.tolist(),
                    'confidence': float(prob)
                })
        
        return faces
    
    def extract_face_embedding(self, image):
        """提取人脸特征"""
        face = self.mtcnn(image)
        if face is None:
            return None
        
        with torch.no_grad():
            embedding = self.facenet(face.unsqueeze(0))
        
        return embedding.cpu().numpy()[0]
    
    def cluster_faces(self, photo_folder):
        """聚类照片中的人脸"""
        embeddings = []
        metadata = []
        
        for filename in os.listdir(photo_folder):
            if not filename.lower().endswith(('.jpg', '.png')):
                continue
            
            path = os.path.join(photo_folder, filename)
            image = Image.open(path).convert('RGB')
            
            # 提取人脸
            embedding = self.extract_face_embedding(image)
            if embedding is not None:
                embeddings.append(embedding)
                metadata.append({'path': path, 'filename': filename})
        
        if len(embeddings) == 0:
            return []
        
        # DBSCAN聚类
        embeddings = np.array(embeddings)
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # 整理结果
        clusters = {}
        for label, meta in zip(labels, metadata):
            if label == -1:  # 噪声点
                continue
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(meta)
        
        return clusters
```

### Week 3: 系统集成（4天）

**完整系统**
```python
import gradio as gr
from PIL import Image

class SmartAlbum:
    def __init__(self):
        self.classifier = SceneClassifier()
        self.search_engine = PhotoSearchEngine()
        self.captioner = PhotoCaptioner()
        self.face_manager = FaceManager()
    
    def analyze_photo(self, image):
        """分析单张照片"""
        # 场景分类
        categories = self.classifier.classify(image)
        
        # 生成描述
        caption = self.captioner.generate_caption(image)
        
        # 人脸检测
        faces = self.face_manager.detect_faces(image)
        
        return {
            'caption': caption,
            'categories': categories,
            'num_faces': len(faces)
        }

# Gradio界面
def create_app():
    album = SmartAlbum()
    
    # 加载索引
    try:
        album.search_engine.load_index('./index')
        print("索引加载成功")
    except:
        print("未找到索引，请先构建")
    
    with gr.Blocks(title="智能相册助手") as demo:
        gr.Markdown("# 📸 智能相册助手")
        
        with gr.Tab("照片分析"):
            with gr.Row():
                input_image = gr.Image(type="pil", label="上传照片")
                with gr.Column():
                    caption_output = gr.Textbox(label="照片描述", lines=2)
                    category_output = gr.JSON(label="场景分类")
                    faces_output = gr.Number(label="人脸数量")
            
            analyze_btn = gr.Button("分析照片")
            
            def analyze(image):
                result = album.analyze_photo(image)
                return (
                    result['caption'],
                    result['categories'],
                    result['num_faces']
                )
            
            analyze_btn.click(
                analyze,
                inputs=input_image,
                outputs=[caption_output, category_output, faces_output]
            )
        
        with gr.Tab("智能搜索"):
            with gr.Row():
                with gr.Column():
                    search_text = gr.Textbox(label="搜索描述", 
                        placeholder="例如: 海边日落")
                    search_image = gr.Image(type="pil", label="或上传照片搜索")
                    top_k = gr.Slider(1, 20, value=6, step=1, label="返回数量")
                    search_btn = gr.Button("搜索")
                
                result_gallery = gr.Gallery(label="搜索结果", columns=3)
            
            def search(text, image, k):
                if text:
                    results = album.search_engine.search_by_text(text, int(k))
                elif image:
                    results = album.search_engine.search_by_image(image, int(k))
                else:
                    return []
                
                # 返回图片路径列表
                return [r['path'] for r in results]
            
            search_btn.click(
                search,
                inputs=[search_text, search_image, top_k],
                outputs=result_gallery
            )
        
        with gr.Tab("批量处理"):
            folder_input = gr.Textbox(label="照片文件夹路径")
            process_btn = gr.Button("构建索引")
            status_output = gr.Textbox(label="状态", lines=5)
            
            def process_folder(folder):
                try:
                    album.search_engine.build_index(folder)
                    album.search_engine.save_index('./index')
                    return "索引构建完成！"
                except Exception as e:
                    return f"错误: {str(e)}"
            
            process_btn.click(
                process_folder,
                inputs=folder_input,
                outputs=status_output
            )
    
    return demo

if __name__ == '__main__':
    app = create_app()
    app.launch()
```

## 创新功能（加分项）

### 1. 智能相册生成
```python
def create_smart_album(photos, theme):
    """根据主题生成相册"""
    # 选择符合主题的照片
    # 按时间/地点排序
    # 生成故事线
    pass
```

### 2. 回忆推荐
```python
def recommend_memories():
    """推荐"xx年前的今天"等回忆"""
    pass
```

### 3. 照片评分
```python
def score_photo(image):
    """评估照片质量（清晰度、构图等）"""
    pass
```

### 4. 重复照片检测
```python
def find_duplicates(photos):
    """找出重复/相似照片"""
    pass
```

## 评估指标

1. **分类准确率**: 场景分类的准确性
2. **检索Recall@K**: 搜索结果的相关性
3. **Caption质量**: BLEU/CIDEr分数
4. **人脸聚类纯度**: 同一簇的人脸相似度
5. **系统响应时间**: 搜索速度

## 展示要点（面试时讲）

1. **实用性**：每个人都需要管理照片
2. **技术全面**：分类、检索、生成、聚类
3. **工程实现**：索引优化、增量更新
4. **用户体验**：界面友好、响应快速
5. **扩展性**：可以加更多功能

## 优势对比

**vs 其他相册App：**
- Google Photos: 隐私问题，需要上传
- iCloud: 只能苹果设备
- **你的系统**: 本地运行，隐私安全，开源免费

## 时间规划

- **Week 1（4-5天）**: 基础功能（分类+搜索）
- **Week 2（4-5天）**: 高级功能（Caption+人脸）
- **Week 3（4天）**: 系统集成+界面
- **Week 4（2天）**: 优化+文档

**总计**: 约14-16天，40-50小时

## 技术亮点

✅ CLIP零样本场景分类  
✅ FAISS高效向量检索  
✅ BLIP图像描述生成  
✅ FaceNet人脸识别  
✅ 完整的工程实现  

这个项目实用性强，技术全面，面试时很好讲！








