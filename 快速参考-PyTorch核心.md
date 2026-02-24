# PyTorch核心知识速查

## 遇到不会的时候查这个

---

## 1. Tensor操作

```python
import torch

# 创建
x = torch.randn(3, 4)        # 随机
x = torch.zeros(3, 4)        # 全0
x = torch.ones(3, 4)         # 全1
x = torch.tensor([1, 2, 3])  # 从列表

# 形状
x.shape                      # 查看形状
x.view(2, 6)                 # 改变形状
x.unsqueeze(0)               # 增加维度
x.squeeze()                  # 去掉维度1

# GPU
x = x.cuda()                 # CPU → GPU
x = x.cpu()                  # GPU → CPU
```

---

## 2. 数据加载

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])

# 加载
dataset = datasets.ImageFolder('./data/train', transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 使用
for images, labels in loader:
    # images: [32, 3, 224, 224]
    # labels: [32]
    pass
```

---

## 3. 训练循环

```python
model = YourModel().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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

## 4. 保存和加载

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 5. 常用操作

```python
# 不计算梯度
with torch.no_grad():
    output = model(input)

# 训练/评估模式
model.train()
model.eval()

# 查看形状
print(x.shape)

# 取最大值索引
_, predicted = output.max(1)
```

**遇到不会的就查这个！**

