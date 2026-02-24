# VLM (Vision-Language Models)

## 核心架构

### BLIP (Bootstrapping Language-Image Pre-training)

**架构：**
```
Image → ViT Encoder → Image Features
Text  → BERT Encoder → Text Features
         ↓
   Cross Attention (Image <-> Text)
         ↓
   Multi-task Training
```

**三大任务：**
1. Image-Text Contrastive (ITC): 对比学习
2. Image-Text Matching (ITM): 匹配预测
3. Image-grounded Text Generation (LM): 生成caption

**使用示例：**
```python
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# 加载模型
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# 图像caption生成
image = Image.open("cat.jpg")
inputs = processor(image, return_tensors="pt")
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)  # "a cat sitting on a table"

# 条件生成（给定prompt）
text = "a photo of"
inputs = processor(image, text, return_tensors="pt")
outputs = model.generate(**inputs)
```

### LLaVA (Large Language and Vision Assistant)

**架构：**
```
Image → Vision Encoder (CLIP ViT) → Image Tokens
                                        ↓
                        Projection Layer (MLP)
                                        ↓
Text Tokens ──────────────→ LLM (Vicuna/Llama)
                                        ↓
                                    Response
```

**核心思想：** 将图像转为LLM能理解的"视觉token"

**训练流程：**
1. **Stage 1**: 只训练Projection Layer（冻结ViT和LLM）
2. **Stage 2**: 微调Projection + LLM（冻结ViT）

**使用示例：**
```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# VQA (Visual Question Answering)
image = Image.open("dog.jpg")
prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"

inputs = processor(text=prompt, images=image, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
```

### Qwen-VL

**特点：**
- 支持中文
- 多图理解
- 细粒度定位

**使用：**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

query = tokenizer.from_list_format([
    {'image': 'image.jpg'},
    {'text': '这张图片里有什么？'}
])

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
```

## 关键技术

### 1. Image-Text Alignment

**对齐方法：**
- Contrastive Learning (CLIP)
- Cross Attention (BLIP)
- Projection (LLaVA)

### 2. Instruction Tuning

**数据格式：**
```json
{
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\n描述这张图片"
    },
    {
      "from": "gpt",
      "value": "这是一只在草地上玩耍的狗"
    }
  ]
}
```

### 3. 微调策略

**LoRA微调：**
```python
from peft import LoraConfig, get_peft_model

# LoRA配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 应用LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4M / 7B = 0.06%
```

## 实战：微调BLIP做图像描述

```python
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CaptionDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        caption = self.captions[idx]
        
        # 编码
        encoding = self.processor(
            images=image,
            text=caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        # 移除batch维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        return encoding

# 训练
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    
    # 数据
    image_paths = [...]  # 你的图片路径
    captions = [...]     # 对应的描述
    dataset = CaptionDataset(image_paths, captions, processor)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    # 训练循环
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # 前向
            outputs = model(**batch)
            loss = outputs.loss
            
            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}')
    
    # 保存
    model.save_pretrained('./blip_finetuned')

if __name__ == '__main__':
    train()
```

## 面试重点

**Q1: BLIP和CLIP的区别？**  
A: CLIP只做对比学习，双塔结构，无法生成；BLIP有Cross Attention，支持生成任务

**Q2: LLaVA如何将图像输入给LLM？**  
A: 用CLIP ViT提取特征 → MLP投影 → 当作特殊token输入LLM

**Q3: 为什么需要Instruction Tuning？**  
A: 让模型理解多样化的指令，提升泛化能力和对话能力

**Q4: 如何评估VLM？**  
A: 
- Caption: BLEU, CIDEr, METEOR
- VQA: Accuracy
- 人工评估: 可信度、幻觉检测

## 作业

1. 用BLIP生成100张图片的caption，分析质量
2. 微调BLIP在自己的数据上
3. 对比BLIP和LLaVA在VQA任务上的表现















