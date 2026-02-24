# 常见问题 FAQ

## 环境相关

### Q: 怎么激活环境？
```bash
conda activate multimodal
```

### Q: 怎么启动Jupyter？
```bash
cd 项目目录
jupyter notebook
```

### Q: 怎么安装新包？
```bash
conda activate multimodal
pip install 包名
```

---

## 代码相关

### Q: 代码报错了怎么办？
1. 看错误信息最后一行
2. Google搜索错误
3. 查看 `常见错误解决.md`
4. 问我

### Q: CUDA out of memory？
```python
# 减小batch_size
batch_size = 8  # 改小一点
```

### Q: 训练太慢？
- 减少数据量
- 减少epoch数
- 使用更小的模型

---

## 项目相关

### Q: 数据从哪里来？
- 项目0：网上下载20张
- 项目1：Kaggle下载
- 项目2：用COCO数据集

### Q: 准确率太低怎么办？
- 增加训练轮数
- 数据增强
- 用更大的模型

### Q: 怎么上传GitHub？
```bash
git init
git add .
git commit -m "first commit"
git remote add origin https://github.com/...
git push -u origin main
```

---

## 学习相关

### Q: 看不懂代码怎么办？
- 先跑通
- 再慢慢理解
- 不懂的问我

### Q: 时间不够怎么办？
- 放慢速度
- 每天4小时就好
- 6个月够用

### Q: 感觉学不会？
- 很正常
- 慢慢来
- 做出第一个项目就有信心了

---

## 求职相关

### Q: 春招来得及吗？
A: 2月14日开始，春招来不及。专心秋招。

### Q: 秋招来得及吗？
A: 完全来得及！6个月够用。

### Q: 需要几个项目？
A: 2-3个完整项目就够。

---

**有问题随时查这个文件或问我！**

