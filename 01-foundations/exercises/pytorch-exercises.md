# PyTorch 核心练习题

## 练习1：Tensor操作（必做）

```python
import torch

# 题目1：创建一个3x4的随机矩阵，计算每行的和
# TODO: 你的代码

# 题目2：将上述矩阵转置，然后与原矩阵相乘
# TODO: 你的代码

# 题目3：创建一个mask，保留矩阵中大于0.5的元素
# TODO: 你的代码
```

## 练习2：自动求导

```python
# 题目：手写一个简单的神经网络（不用nn.Module）
# 输入: [B, 10]
# 隐藏层: [10, 5]，激活函数ReLU
# 输出层: [5, 1]
# 损失: MSE

# TODO: 实现前向传播和反向传播
```

## 练习3：Dataset实现

```python
# 题目：实现一个生成正弦波数据的Dataset
# 输入: x在[0, 2π]之间
# 输出: y = sin(x) + noise

from torch.utils.data import Dataset

class SineDataset(Dataset):
    def __init__(self, num_samples=1000):
        # TODO: 生成数据
        pass
    
    def __len__(self):
        # TODO
        pass
    
    def __getitem__(self, idx):
        # TODO
        pass
```

## 练习4：模型训练

```python
# 题目：训练一个模型拟合 y = x^2
# 使用MSE损失，Adam优化器
# 可视化训练过程和最终结果

# TODO: 完整实现
```

## 练习5：多分类

```python
# 题目：实现一个MLP做MNIST分类
# 要求：
# - 至少2个隐藏层
# - 使用ReLU激活
# - 使用Dropout防止过拟合
# - 测试集准确率 > 95%

# TODO: 完整实现
```

## 标准答案（先自己做，实在不会再看）

### 答案1
```python
import torch

# 题目1
x = torch.randn(3, 4)
row_sums = x.sum(dim=1)
print(row_sums)

# 题目2
x_t = x.transpose(0, 1)
result = torch.matmul(x, x_t)
print(result.shape)  # [3, 3]

# 题目3
mask = x > 0.5
masked_x = x[mask]
print(masked_x)
```

### 答案3
```python
import torch
import numpy as np
from torch.utils.data import Dataset

class SineDataset(Dataset):
    def __init__(self, num_samples=1000, noise_std=0.1):
        self.x = torch.linspace(0, 2*np.pi, num_samples).unsqueeze(1)
        self.y = torch.sin(self.x) + torch.randn_like(self.x) * noise_std
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```















