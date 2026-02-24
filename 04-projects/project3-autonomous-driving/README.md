# 项目3：自动驾驶场景理解（对齐小鹏业务）

## 项目目标
基于多模态模型，实现自动驾驶场景的理解和描述，直接对应小鹏岗位需求

## 核心功能
1. 场景描述生成：给定行车记录仪图像，生成文字描述
2. 风险检测：识别潜在危险（行人、车辆、障碍物）
3. 决策建议：根据场景给出驾驶建议

## 数据集

### nuScenes（推荐）
- 官网：https://www.nuscenes.org/
- 1000个场景，40万张图片
- 有3D标注、语义分割、场景描述

### Waymo Open Dataset
- 更大规模，但下载较慢

### BDD100K
- 伯克利驾驶数据集
- 有天气、时间标注

## 技术方案

### 方案1：CLIP + 场景分类
```
图像 → CLIP → 特征 → 分类器 → 场景类别
                              → 风险等级
```

### 方案2：VLM生成描述（推荐）
```
图像 → VLM (BLIP/LLaVA) → 场景描述
                          → 风险分析
                          → 决策建议
```

## 实现步骤

### Week 1-2: 数据准备

**下载nuScenes mini数据集**
```python
# 使用nuscenes-devkit
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='./data/nuscenes', verbose=True)

# 提取场景
def extract_scenes():
    data = []
    for scene in nusc.scene:
        sample = nusc.get('sample', scene['first_sample_token'])
        
        # 获取前视图像
        cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
        img_path = nusc.dataroot + '/' + cam_data['filename']
        
        # 获取标注
        annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]
        
        # 生成描述
        caption = generate_caption(annotations)
        
        data.append({
            'image': img_path,
            'caption': caption,
            'scene_token': scene['token']
        })
    
    return data

def generate_caption(annotations):
    """根据标注生成场景描述"""
    objects = [ann['category_name'] for ann in annotations]
    
    # 简单模板
    caption = f"A driving scene with "
    caption += f"{len([o for o in objects if 'vehicle' in o])} vehicles, "
    caption += f"{len([o for o in objects if 'pedestrian' in o])} pedestrians, "
    caption += "on the road."
    
    return caption
```

### Week 3-4: 模型训练

**微调BLIP做场景描述**
```python
from transformers import BlipForConditionalGeneration, BlipProcessor
import torch
from torch.utils.data import Dataset, DataLoader

class DrivingSceneDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image = Image.open(item['image']).convert('RGB')
        caption = item['caption']
        
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

def train():
    device = torch.device('cuda')
    
    # 模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    
    # 数据
    data = extract_scenes()
    dataset = DrivingSceneDataset(data, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 训练
    model.train()
    for epoch in range(5):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    model.save_pretrained('./driving_scene_model')
```

### Week 5-6: 风险检测

**结合目标检测**
```python
from transformers import DetrImageProcessor, DetrForObjectDetection

class RiskDetector:
    def __init__(self):
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    
    def detect(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        
        # 后处理
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.5
        )[0]
        
        # 分析风险
        risks = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            obj_name = self.model.config.id2label[label.item()]
            
            # 行人在道路上 = 高风险
            if 'person' in obj_name and box[1] > image.height * 0.5:
                risks.append({
                    'type': 'pedestrian',
                    'level': 'high',
                    'position': box.tolist()
                })
        
        return risks
```

### Week 7-8: 完整系统

**多模态决策**
```python
class AutonomousDrivingAssistant:
    def __init__(self):
        self.caption_model = BlipForConditionalGeneration.from_pretrained('./driving_scene_model')
        self.caption_processor = BlipProcessor.from_pretrained('./driving_scene_model')
        self.risk_detector = RiskDetector()
    
    def analyze_scene(self, image):
        # 1. 场景描述
        inputs = self.caption_processor(images=image, return_tensors="pt")
        caption_ids = self.caption_model.generate(**inputs)
        caption = self.caption_processor.decode(caption_ids[0], skip_special_tokens=True)
        
        # 2. 风险检测
        risks = self.risk_detector.detect(image)
        
        # 3. 决策建议
        decision = self.make_decision(caption, risks)
        
        return {
            'description': caption,
            'risks': risks,
            'decision': decision
        }
    
    def make_decision(self, caption, risks):
        if len(risks) > 0:
            high_risks = [r for r in risks if r['level'] == 'high']
            if high_risks:
                return "减速并保持警惕，前方有行人"
        
        if 'congestion' in caption or 'traffic' in caption:
            return "减速，注意车距"
        
        return "保持当前速度，注意观察"

# Gradio界面
import gradio as gr

assistant = AutonomousDrivingAssistant()

def predict(image):
    result = assistant.analyze_scene(image)
    
    # 格式化输出
    output = f"**场景描述：**\n{result['description']}\n\n"
    
    if result['risks']:
        output += "**检测到的风险：**\n"
        for risk in result['risks']:
            output += f"- {risk['type']} (等级: {risk['level']})\n"
    else:
        output += "**检测到的风险：** 无\n"
    
    output += f"\n**驾驶建议：**\n{result['decision']}"
    
    return output

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传行车图像"),
    outputs=gr.Textbox(label="分析结果", lines=10),
    title="自动驾驶场景理解助手",
    description="上传行车记录仪图像，AI分析场景并给出驾驶建议"
)

demo.launch()
```

## 评估指标

1. **Caption质量**: BLEU, CIDEr
2. **检测准确率**: mAP
3. **决策正确率**: 人工评估

## 创新点

1. **多传感器融合**（加分）：图像 + 激光雷达点云
2. **时序建模**：视频序列理解，预测轨迹
3. **可解释性**：注意力可视化，解释决策依据
4. **实时性**：模型压缩，推理加速

## 对齐小鹏岗位

**岗位要求：**
- 自动驾驶场景的大规模预训练
- 多模态数据处理
- 模型对齐与效果评估

**你的项目展示：**
- ✅ 使用nuScenes自动驾驶数据集
- ✅ 多模态模型（视觉+语言）
- ✅ 数据处理pipeline
- ✅ 模型微调和评估

## 面试讲解要点

1. **为什么做这个项目**：对自动驾驶感兴趣，想对齐小鹏业务
2. **技术选型**：为什么用VLM而不是纯检测？
3. **数据处理**：nuScenes数据格式，如何构建训练集？
4. **挑战**：长尾场景、实时性、安全性
5. **改进方向**：多帧融合、3D理解、端到端决策

## 时间规划
- Week 1-2: 数据准备（20h）
- Week 3-4: 场景描述模型（20h）
- Week 5-6: 风险检测（15h）
- Week 7-8: 系统集成（15h）

完成这个项目，你就有小鹏自动驾驶岗位80%的能力了！















