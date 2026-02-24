# 面试高频问题汇总

## 基础理论

### PyTorch

**Q1: PyTorch的动态图和静态图的区别？**  
A: PyTorch是动态图（Define-by-Run），每次前向传播都构建计算图，灵活但速度稍慢；TensorFlow 1.x是静态图（Define-and-Run），先定义后执行，速度快但不够灵活。PyTorch 2.0引入torch.compile支持图优化。

**Q2: 什么时候用model.train()和model.eval()？**  
A: train模式启用Dropout和BatchNorm的训练行为；eval模式禁用。推理时必须用eval，否则结果不稳定。

**Q3: 梯度累积怎么实现？**  
```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Q4: 如何冻结部分参数？**  
```python
# 冻结backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# 只优化head
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)
```

### Transformer

**Q5: 手写Self-Attention（10分钟）**  
```python
def self_attention(X, W_q, W_k, W_v):
    """
    X: [B, L, D]
    返回: [B, L, D]
    """
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    
    scores = Q @ K.transpose(-2, -1) / math.sqrt(D)
    attn = torch.softmax(scores, dim=-1)
    output = attn @ V
    return output
```

**Q6: Transformer的复杂度是多少？如何优化？**  
A: O(n²d)，n是序列长度。优化方法：
- Flash Attention（减少显存访问）
- Sparse Attention（只计算部分位置）
- Linear Attention（近似计算）

**Q7: 为什么Transformer需要Position Encoding？**  
A: Attention是置换不变的，打乱顺序结果不变。Position Encoding引入位置信息。

**Q8: Multi-Head Attention的好处？**  
A: 不同head学习不同的关系模式（如语法、语义、长距离依赖等），增强表达能力。

### CLIP

**Q9: CLIP的训练目标是什么？**  
A: 对比学习，让匹配的图文对在embedding空间靠近，不匹配的远离。具体是InfoNCE损失。

**Q10: CLIP如何做零样本分类？**  
```python
# 构造文本prompt
texts = [f"a photo of a {cls}" for cls in classes]

# 计算相似度
text_embeds = model.encode_text(texts)
image_embed = model.encode_image(image)
similarity = image_embed @ text_embeds.T

# 预测
pred = similarity.argmax()
```

**Q11: CLIP的局限性？**  
A: 
- 细粒度分类能力弱（如狗的品种）
- 依赖文本描述质量
- OCR任务表现差
- 计算量大

### 多模态

**Q12: VLM和CLIP的区别？**  
A: 
- CLIP: 图文对比学习，双塔结构，无法生成
- VLM: 图文生成模型，Encoder-Decoder，可以生成caption

**Q13: 如何融合图像和文本特征？**  
A: 
- Early Fusion: 拼接后输入Transformer
- Late Fusion: 分别编码后融合
- Cross Attention: 图像attend文本（BLIP）

**Q14: 介绍一下BLIP/LLaVA？**  
A: 
- BLIP: Bootstrap Language-Image Pre-training，用合成caption增强训练
- LLaVA: 将视觉encoder连接到LLM，做视觉指令微调

## 算法和数学

**Q15: 解释一下梯度消失和梯度爆炸**  
A: 
- 梯度消失: 反向传播时梯度不断衰减，深层网络难训练
- 梯度爆炸: 梯度不断放大，导致参数更新过大
- 解决: 残差连接、LayerNorm、梯度裁剪

**Q16: BatchNorm和LayerNorm的区别？**  
A: 
- BatchNorm: 对batch维度归一化，CV常用
- LayerNorm: 对特征维度归一化，NLP常用（因为序列长度不固定）

**Q17: 交叉熵损失的公式？**  
```
L = -∑ y_i * log(p_i)
```
多分类时，y是one-hot，简化为 `-log(p_true_class)`

**Q18: Adam优化器的原理？**  
A: 结合momentum（一阶动量）和RMSprop（二阶动量），自适应学习率。

**Q19: 过拟合怎么办？**  
A: 
- 数据增强
- Dropout、L2正则
- Early Stopping
- 减小模型复杂度

**Q20: 如何评估检索系统？**  
A: 
- Recall@K: top-K中正确答案的比例
- MRR: Mean Reciprocal Rank
- mAP: mean Average Precision

## 项目经验

**Q21: 介绍一下你的图文检索项目**  
**回答框架：**
1. 背景：为什么做这个项目？
2. 方案：用了什么技术？为什么这么选？
3. 实现：遇到什么挑战？如何解决？
4. 效果：指标如何？有什么改进空间？

**Q22: 你的模型效果不好，如何debug？**  
A: 
1. 检查数据：可视化、统计分布
2. 简化模型：先跑通小模型
3. 检查loss：是否收敛？过拟合？
4. 可视化：attention map、embedding
5. 对比baseline：差距在哪？

**Q23: 如何处理类别不平衡？**  
A: 
- 过采样/欠采样
- 类别权重
- Focal Loss
- 数据增强

**Q24: 大模型如何加速推理？**  
A: 
- 量化（INT8/FP16）
- 剪枝
- 蒸馏
- 批处理
- KV Cache（生成任务）

## 编程题

**Q25: 手写NMS（非极大值抑制）**  
```python
def nms(boxes, scores, iou_threshold):
    """
    boxes: [[x1, y1, x2, y2], ...]
    scores: [s1, s2, ...]
    """
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep = []
    
    while indices:
        i = indices[0]
        keep.append(i)
        indices = indices[1:]
        
        # 过滤掉与i的IoU > threshold的框
        indices = [
            j for j in indices 
            if iou(boxes[i], boxes[j]) < iou_threshold
        ]
    
    return keep
```

**Q26: 实现Top-K**  
```python
def top_k(arr, k):
    """返回最大的k个元素"""
    import heapq
    return heapq.nlargest(k, arr)
```

**Q27: 手写BN层**  
```python
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.eps = eps
        self.momentum = momentum
    
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            
            # 更新running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

## 开放性问题

**Q28: 介绍一篇你最近读的论文**  
准备2-3篇核心论文：CLIP、LLaVA、Llama等

**Q29: 如何看待AI的未来发展？**  
展示你的思考和视野，结合多模态、具身智能、自动驾驶等

**Q30: 你有什么问题要问我？**  
**推荐问题：**
- 团队的技术栈和方向？
- 实习生的成长路径？
- 多模态在公司的应用场景？

## 准备清单

面试前一天：
- [ ] 手写Attention代码3遍
- [ ] 复习CLIP原理和代码
- [ ] 准备项目讲解（5分钟版本）
- [ ] 刷5道LeetCode中等题
- [ ] 看3篇核心论文摘要
- [ ] 准备自我介绍（1分钟）

记住：**原理要懂，代码要会写，项目要讲清楚！**















