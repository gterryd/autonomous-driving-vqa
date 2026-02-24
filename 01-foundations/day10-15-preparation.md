# Day 10-15: 项目准备（2.15-2.20）

## Day 10-12 (2.15-2.17): Git + 项目框架

### Git基础
```bash
# 初始化
git init
git config user.name "your_name"
git config user.email "your_email"

# 基本操作
git add .
git commit -m "initial commit"
git branch -M main

# GitHub
git remote add origin https://github.com/username/repo.git
git push -u origin main

# 常用命令
git status
git log
git diff
git checkout -b new_branch
```

### Python项目结构
```
my_project/
├── data/
├── models/
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── notebooks/
├── tests/
├── requirements.txt
├── README.md
└── .gitignore
```

### requirements.txt
```
torch==2.5.1
torchvision
transformers
pillow
numpy
matplotlib
tqdm
```

### README模板
```markdown
# 项目名称

## 功能
- 功能1
- 功能2

## 安装
pip install -r requirements.txt

## 使用
python src/train.py

## 结果
准确率: 95%
```

---

## Day 13-15 (2.18-2.20): LeetCode + 代码模板

### LeetCode必刷10题

**1. 两数之和**
```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

**2. 反转链表**
```python
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_node = curr.next
        curr.next = prev
        prev = curr
        curr = next_node
    return prev
```

**3. 有效括号**
```python
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}
    for char in s:
        if char in mapping:
            if not stack or stack.pop() != mapping[char]:
                return False
        else:
            stack.append(char)
    return not stack
```

**4. 最大子数组和**
```python
def max_subarray(nums):
    max_sum = curr_sum = nums[0]
    for num in nums[1:]:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
```

**5. 爬楼梯**
```python
def climb_stairs(n):
    if n <= 2:
        return n
    prev, curr = 1, 2
    for _ in range(3, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

**6-10题见：** `05-interview-prep/leetcode-essentials.md`

---

## 代码模板背诵

### 训练循环
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        data, target = batch
        data, target = data.to(device), target.to(device)
        
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_loss = validate(model, val_loader)
    
    print(f'Epoch {epoch+1}, Val Loss: {val_loss:.4f}')
```

### Dataset模板
```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
```

---

## 面试准备

### 自我介绍（1分钟）
```
您好，我是数学与应用数学专业的学生。

最近3个月系统学习了深度学习和多模态AI，
包括PyTorch、Transformer、CLIP等技术。

完成了X个项目，包括图像分类、图文检索等。

熟悉PyTorch、HuggingFace，能手写Transformer核心代码。

对AI算法很感兴趣，希望能加入贵公司学习成长。
```

### 项目介绍（3分钟）
```
项目名称：图像检索系统
技术栈：CLIP + FAISS
功能：文本搜图、以图搜图
难点：如何优化检索速度
效果：Recall@10达到XX%
```

---

## 检查点

- [ ] 会Git基本操作
- [ ] 能搭建项目结构
- [ ] LeetCode 10题全AC
- [ ] 能背诵3个代码模板
- [ ] 准备好自我介绍

**Day 15结束，基础阶段完成！**



