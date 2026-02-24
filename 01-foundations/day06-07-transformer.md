# Day 6-7: Transformer（2.11-2.12）⭐

## Day 6 (2.11): 手写Attention ⭐

### 第一步：理解Attention机制

**Attention（注意力）：让模型知道"哪些词更重要"**

举例：
```
句子："我爱吃苹果"
问题："我爱吃什么？"

Attention会让模型重点关注"苹果"这个词
而不是平等对待所有词
```

**Self-Attention流程：**
```
1. 输入句子的每个词
2. 计算每个词对其他词的"注意力分数"
3. 根据分数，加权组合其他词的信息
4. 得到新的表示（融合了上下文信息）
```

### 第二步：手写Self-Attention代码

```python
import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    """
    自注意力机制
    
    输入：一句话的词向量 [batch_size, 句子长度, 词向量维度]
    输出：考虑了上下文的新向量 [batch_size, 句子长度, 词向量维度]
    
    例如：
    输入：[2, 10, 512] - 2个句子，每句10个词，每词512维
    输出：[2, 10, 512] - 形状不变，但融合了上下文
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim  # 词向量维度，比如512
        
        # 【三个线性变换】
        # Q（Query）：查询向量，"我要找什么"
        # K（Key）：键向量，"我是什么"
        # V（Value）：值向量，"我的内容是什么"
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        """
        x: [batch_size, seq_len, embed_dim]
        例如：[2, 10, 512]
        """
        # 【步骤1】：计算Q、K、V
        # 把输入x通过三个不同的线性变换
        Q = self.W_q(x)  # [2, 10, 512] "我想找什么"
        K = self.W_k(x)  # [2, 10, 512] "我是什么"
        V = self.W_v(x)  # [2, 10, 512] "我的内容"
        
        # 【步骤2】：计算注意力分数
        # Q @ K^T：每个词和其他词的相似度
        # 形状：[2, 10, 512] @ [2, 512, 10] = [2, 10, 10]
        # 结果：10x10矩阵，表示10个词之间两两的相似度
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 除以sqrt(embed_dim)：缩放，防止分数太大
        scores = scores / math.sqrt(self.embed_dim)
        
        # 【步骤3】：Softmax归一化
        # 把分数转成概率（和为1）
        # 形状不变：[2, 10, 10]
        attn = torch.softmax(scores, dim=-1)
        # attn[i, j]表示：第i个词对第j个词的注意力权重
        
        # 【步骤4】：加权求和
        # 用注意力权重，对V进行加权平均
        # [2, 10, 10] @ [2, 10, 512] = [2, 10, 512]
        output = torch.matmul(attn, V)
        
        # 输出：融合了上下文信息的新表示
        return output, attn

# 【测试代码】
model = SelfAttention(embed_dim=512)

# 假设有2个句子，每句10个词，每词512维
x = torch.randn(2, 10, 512)
print("输入形状：", x.shape)  # [2, 10, 512]

output, attention = model(x)
print("输出形状：", output.shape)  # [2, 10, 512]
print("注意力形状：", attention.shape)  # [2, 10, 10]

# attention[0, 3, :]就是：第1个句子的第4个词，对所有词的注意力分布
print("\n第1个句子，第4个词对其他词的注意力：")
print(attention[0, 3, :])
# 输出类似：tensor([0.05, 0.08, 0.12, 0.15, 0.20, ...])
# 表示对第5个词注意力最高（0.20）
```

**理解了吗？**
- Q、K、V：三个不同的视角看同一个输入
- Attention分数：计算相似度
- Softmax：转成概率
- 加权求和：融合信息

### Multi-Head Attention
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        B, L, D = x.shape
        
        Q = self.W_q(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        output = output.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(output)
```

### Position Encoding
```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### Transformer Block
```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
    
    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x
```

**作业：** 在纸上默写一遍！

---

## Day 7 (2.12): BERT/GPT使用

### HuggingFace基础
```python
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "Hello, world!"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

print(outputs.last_hidden_state.shape)
```

### 文本分类
```python
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 简单微调（不用Trainer）
```python
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.cuda()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 面试必问

**Q1: 手写Self-Attention流程**
```
1. 输入 x: [B, L, D]
2. 计算 Q=xW_q, K=xW_k, V=xW_v
3. scores = QK^T / sqrt(d)
4. attn = softmax(scores)
5. output = attn @ V
```

**Q2: 为什么除以sqrt(d)？**
防止点积过大导致softmax梯度消失

**Q3: Multi-Head的好处？**
不同head学习不同的关系模式

**Q4: Transformer复杂度？**
O(n²d)，n是序列长度

---

## 检查点

- [ ] 能手写Self-Attention（5分钟内）
- [ ] 理解Multi-Head原理
- [ ] 会用HuggingFace
- [ ] 能微调BERT

**Day 6-7是重点，必须掌握！**



