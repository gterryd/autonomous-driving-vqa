# Day 1-5: PyTorch + CNN（2.6-2.10）

## Day 1 (2.6): PyTorch基础

### 第一步：什么是Tensor？

**Tensor就是多维数组，是PyTorch的核心数据结构**

```python
import torch

# 【创建Tensor】
# randn表示随机生成，3行4列的矩阵
x = torch.randn(3, 4)
print("创建一个3x4的随机矩阵：")
print(x)
# 输出类似：
# tensor([[ 0.1234, -0.5678,  1.2345, -0.9876],
#         [ 0.5432, -0.1234,  0.7890, -0.3456],
#         [-0.2345,  0.6789, -0.4567,  0.8901]])

# 【转置操作】
# transpose(0, 1)表示交换第0维和第1维，就是行列互换
y = x.transpose(0, 1)
print("\n转置后变成4x3：")
print(y.shape)  # torch.Size([4, 3])

# 【改变形状】
# view(-1)表示拉平成一维，-1表示自动计算这个维度的大小
z = x.view(-1)
print("\n拉平成一维向量：")
print(z.shape)  # torch.Size([12]) 因为3*4=12

# 【索引操作】
# [0, :]表示取第0行，所有列
a = x[0, :]
print("\n取第一行：", a)

# [:, 1]表示取所有行，第1列
b = x[:, 1]
print("取第二列：", b)
```

### 第二步：自动求导（神经网络的核心）

**自动求导可以自动计算梯度，不用手动推导公式**

```python
# ⚠️ 重要提示：在Jupyter中运行这段代码
# 如果多次运行会报错，那是正常的！
# 解决方法：每次重新运行整个cell（包括x的创建）

import torch

# 【创建需要求导的变量】
# requires_grad=True 告诉PyTorch：这个变量我要求导
x = torch.tensor([2.0], requires_grad=True)
print("创建变量 x =", x.item())  # x = 2.0

# 【定义函数】
# 假设 y = x² + 3x
y = x ** 2 + 3 * x
print("计算 y = x² + 3x =", y.item())  # y = 4 + 6 = 10

# 【自动求导】
# backward()会自动计算 dy/dx
y.backward()

# 【查看梯度】
# dy/dx = 2x + 3，当x=2时，dy/dx = 7
print("梯度 dy/dx =", x.grad.item())  # 7.0

print("\n✅ 如果看到上面的7.0，说明求导成功！")

# 【常见错误】
# 如果再次运行 y.backward() 会报错
# 因为计算图已经释放了
# 解决：重新运行整个cell（从x的创建开始）

# 为什么要求导？因为训练神经网络需要知道：
# "改变参数，损失会怎么变化"，这样才能优化参数
```

**如果报错了怎么办：**
1. 重新运行整个格子（从import torch开始）
2. 或者点击上方 Kernel → Restart，重新开始

### 第三步：手写线性回归（理解训练流程）

**目标：让电脑学会 y = 3x + 2 这个关系**

```python
import torch

# 【第1步：生成训练数据】
# 我们知道真实关系是 y = 3x + 2
# 生成100个样本点，x是随机的
x = torch.randn(100, 1)  # 100个样本，每个1维
# y = 3x + 2 + 一点噪声（模拟真实数据的误差）
y = 3 * x + 2 + torch.randn(100, 1) * 0.1

print("生成了100个训练数据")
print("前3个样本：")
for i in range(3):
    print(f"  x={x[i].item():.2f}, y={y[i].item():.2f}")

# 【第2步：初始化参数】
# 我们要让电脑学习w和b，初始值是随机的
w = torch.randn(1, 1, requires_grad=True)  # 权重
b = torch.zeros(1, requires_grad=True)     # 偏置
print(f"\n初始参数：w={w.item():.2f}, b={b.item():.2f}")

# 【第3步：训练100轮】
for epoch in range(100):
    # ① 前向传播：用当前参数预测y
    # @ 表示矩阵乘法，相当于 y_pred = w*x + b
    y_pred = x @ w + b
    
    # ② 计算损失：预测值和真实值的差距
    # mean()表示求平均，这叫"均方误差损失"
    loss = ((y_pred - y) ** 2).mean()
    
    # ③ 反向传播：自动计算梯度
    loss.backward()
    
    # ④ 更新参数：朝着减小损失的方向调整参数
    with torch.no_grad():  # 这里不需要求导
        w -= 0.1 * w.grad  # w = w - 学习率 * 梯度
        b -= 0.1 * b.grad  # b = b - 学习率 * 梯度
        
        # ⑤ 清空梯度（重要！不清空会累积）
        w.grad.zero_()
        b.grad.zero_()
    
    # 每20轮打印一次进度
    if (epoch + 1) % 20 == 0:
        print(f'第{epoch+1}轮，损失={loss.item():.4f}')

# 【第4步：查看学到的参数】
print(f'\n训练完成！')
print(f'学到的参数：w={w.item():.2f}, b={b.item():.2f}')
print(f'真实参数：w=3.00, b=2.00')
print('如果接近，说明学习成功！')

# 输出示例：
# 第20轮，损失=0.0234
# 第40轮，损失=0.0123
# ...
# 学到的参数：w=2.98, b=2.01
# 很接近真实值！
```

**这就是神经网络训练的核心流程：**
1. 准备数据
2. 初始化参数
3. 循环训练（前向→计算损失→反向求导→更新参数）
4. 得到最优参数

---

## Day 2 (2.7): CNN + MNIST

### CNN基础
```python
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64*7*7)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
```

### MNIST完整训练
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

model = CNN().cuda()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    
    print(f'Epoch {epoch+1}, Acc: {100.*correct/len(test_data):.2f}%')
```

目标：准确率>95%

---

## Day 3 (2.8): CIFAR-10

### 数据增强
```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
```

### 修改CNN
```python
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 输入3通道
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256*4*4, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 256*4*4)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)
```

目标：准确率>70%

---

## Day 4 (2.9): ResNet

### 使用预训练模型
```python
from torchvision import models

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 10)
model = model.cuda()

# 方案1：全部训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 方案2：只训练fc层
for param in model.parameters():
    param.requires_grad = False
model.fc.weight.requires_grad = True
model.fc.bias.requires_grad = True
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### 迁移学习
```python
# 在CIFAR-10上微调
train_loader = DataLoader(train_data, batch_size=32)

for epoch in range(10):
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## Day 5 (2.10): 保存和加载

### 保存模型
```python
# 保存整个模型
torch.save(model, 'model.pth')

# 只保存参数（推荐）
torch.save(model.state_dict(), 'model_weights.pth')

# 保存checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')
```

### 加载模型
```python
# 加载参数
model = CNN()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

# 加载checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
```

---

## 检查点

Day 5结束时，你应该：
- [ ] 理解PyTorch训练流程
- [ ] 能手写CNN
- [ ] MNIST准确率>95%
- [ ] CIFAR-10准确率>70%
- [ ] 会使用预训练模型

**前5天搞定PyTorch和CV基础！**



