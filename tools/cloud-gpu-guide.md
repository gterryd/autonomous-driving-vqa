# 云GPU使用指南

## 为什么需要云GPU？

如果你的设备：
- 没有GPU
- GPU显存 < 6GB
- 内存 < 8GB

建议使用云GPU服务！

---

## 方案对比

| 服务 | 价格 | GPU | 优点 | 缺点 | 推荐度 |
|------|------|-----|------|------|--------|
| **Google Colab** | 免费 | T4 | 零配置，开箱即用 | 12小时限制 | ⭐⭐⭐⭐⭐ |
| **Kaggle** | 免费 | P100 | 30小时/周 | 需要注册 | ⭐⭐⭐⭐⭐ |
| **AutoDL** | 0.8元/h | RTX 3090 | 国内快，灵活 | 需付费 | ⭐⭐⭐⭐ |
| **阿里云PAI** | 按量 | 多种 | 稳定 | 较贵 | ⭐⭐⭐ |

---

## 1. Google Colab（最推荐）

### 优势
- ✅ 完全免费（免费版足够）
- ✅ 零配置，打开就能用
- ✅ 预装PyTorch、TensorFlow
- ✅ 可以直接访问Google Drive

### 限制
- ⏰ 连续运行12小时后断开
- 💾 临时存储，关闭后数据丢失
- 🚫 不能挖矿、不能长时间空闲

### 快速开始

**Step 1: 打开Colab**
```
https://colab.research.google.com/
```

**Step 2: 新建Notebook**
点击 "文件" → "新建笔记本"

**Step 3: 启用GPU**
点击 "代码执行程序" → "更改运行时类型" → 选择 "T4 GPU"

**Step 4: 测试GPU**
```python
import torch
print(torch.cuda.is_available())  # 应该输出 True
print(torch.cuda.get_device_name(0))  # Tesla T4
```

**Step 5: 安装依赖**
```python
!pip install transformers datasets
```

### 保存数据

**方案1: 挂载Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# 数据会保存到你的Google Drive
```

**方案2: 下载到本地**
```python
from google.colab import files
files.download('model.pth')
```

### 最佳实践

1. **定期保存checkpoint**
   ```python
   if epoch % 10 == 0:
       torch.save(model.state_dict(), f'/content/drive/MyDrive/model_epoch{epoch}.pth')
   ```

2. **使用小数据集测试**
   先在小数据上跑通，再用完整数据

3. **避免长时间空闲**
   会被断开连接

---

## 2. Kaggle Notebooks

### 优势
- ✅ 免费GPU（P100）
- ✅ 30小时/周
- ✅ 可以保存notebook
- ✅ 有公开数据集

### 快速开始

**Step 1: 注册Kaggle**
```
https://www.kaggle.com/
```

**Step 2: 手机号验证**
设置 → Account → Phone Verification

**Step 3: 创建Notebook**
Code → New Notebook → 右侧开启 GPU

**Step 4: 使用**
```python
import torch
print(torch.cuda.is_available())
```

### 技巧

- 每周30小时，合理安排
- 可以下载数据集到notebook
- 可以fork别人的notebook学习

---

## 3. AutoDL（国内推荐）

### 优势
- ✅ 国内服务器，速度快
- ✅ 按量计费，灵活
- ✅ 配置丰富（RTX 3090/4090/A100等）
- ✅ 支持SSH、Jupyter

### 价格（2026年）
- RTX 3090: 1.5元/小时
- RTX 4090: 3元/小时
- A100: 8元/小时

### 快速开始

**Step 1: 注册充值**
```
https://www.autodl.com/
```
首次充值100元（新手够用1-2个月）

**Step 2: 创建实例**
- 选择GPU：RTX 3090（推荐）
- 选择镜像：PyTorch 2.0
- 选择地区：离你近的

**Step 3: 启动**
点击"启动"，等待30秒

**Step 4: 使用JupyterLab**
点击"JupyterLab"，在浏览器打开

**Step 5: 或使用SSH**
```bash
ssh root@xxx.xxx.xxx.xxx -p xxxxx
```

### 省钱技巧

1. **用完立即关机**
   不用时关机，只按使用时间计费

2. **选便宜的GPU**
   学习阶段3090够用，不需要A100

3. **使用镜像**
   选择预装环境，节省配置时间

4. **数据集缓存**
   第一次下载后保存，避免重复下载

---

## 4. 配置对比建议

### 学习阶段（Day 1-30）
推荐：**Google Colab免费版**
- 每天学习2-3小时
- 足够运行所有基础代码
- 成本：0元

### 项目阶段（3-7月）
推荐：**AutoDL按量付费**
- 需要长时间训练
- RTX 3090：1.5元/小时
- 预算：每月150-300元

### 大模型微调
推荐：**AutoDL + A100**
- 7B模型需要大显存
- A100 40GB: 8元/小时
- 或者用Google Colab Pro（$9.99/月）

---

## 常见问题

### Q1: Colab 12小时限制怎么办？
**A:** 
- 定期保存checkpoint
- 分多次训练
- 或升级Colab Pro（$9.99/月，24小时）

### Q2: 数据怎么传到云端？
**A:**
- Colab: 挂载Google Drive
- AutoDL: scp或通过JupyterLab上传
- Kaggle: 使用Kaggle Datasets

### Q3: 云GPU速度慢？
**A:**
- 检查网络（下载数据集可能慢）
- 使用国内镜像源
- 数据预处理在本地做好

### Q4: 预算有限怎么办？
**A:**
- 优先使用免费服务（Colab + Kaggle）
- 组合使用：理论在本地，训练在云端
- AutoDL按需使用，用完关机

---

## 推荐方案

### 方案1: 纯免费（适合学生）
```
Colab（基础学习） + Kaggle（项目训练）
成本: 0元
限制: 时间限制，不能长训练
```

### 方案2: 小预算（推荐）
```
本地（Day 1-20） + AutoDL（Day 21-36 + 项目）
成本: 200-500元（5个月）
优点: 灵活，性价比高
```

### 方案3: 土豪版
```
本地工作站（RTX 4090）+ 云端备份
成本: 2万+ （硬件）
优点: 随时可用，速度快
```

---

## 下一步

1. **选择云服务**
   - 新手：Google Colab
   - 国内：AutoDL
   - 两者结合最佳

2. **测试环境**
   ```python
   # 在云端运行
   !git clone https://github.com/你的用户名/项目
   cd 项目
   !pip install -r requirements.txt
   !python train.py
   ```

3. **开始学习**
   → 回到 START_HERE.md 继续

---

**记住：云GPU不是必须，但能让学习更顺畅！**








