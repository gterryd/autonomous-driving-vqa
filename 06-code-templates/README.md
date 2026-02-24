# 06 代码模板

## 文件说明

### training_template.py
通用训练模板，包含：
- Config配置类
- Trainer训练器
- 自动保存checkpoint
- 支持验证集
- 进度条显示

### data_loader_template.py
数据加载模板，包含：
- 图像分类Dataset
- 图文对Dataset
- 数据增强
- Collate函数

---

## 使用方法

**直接复用：**
```python
from training_template import Trainer, Config

config = Config()
trainer = Trainer(model, train_loader, val_loader, config)
trainer.fit()
```

**修改配置：**
```python
config.batch_size = 64
config.lr = 0.001
config.epochs = 20
```

**省时间，不要从零写！**
