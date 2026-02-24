# 模型部署与优化

## 部署流程

```
训练模型 → 模型优化 → 导出模型 → 部署服务 → 监控维护
```

## 1. 模型优化

### 量化（Quantization）

**INT8量化：**

```python
import torch
from transformers import CLIPModel

# 加载模型
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# 动态量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# 保存
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

**效果：** 模型大小减少4倍，速度提升2-3倍

### 剪枝（Pruning）

```python
import torch.nn.utils.prune as prune

# 对某一层剪枝
prune.l1_unstructured(model.layer1, name='weight', amount=0.3)

# 全局剪枝
parameters_to_prune = [
    (model.layer1, 'weight'),
    (model.layer2, 'weight'),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
```

### 蒸馏（Distillation）

```python
# 学生模型学习教师模型
teacher_model = LargeModel()
student_model = SmallModel()

def distillation_loss(student_logits, teacher_logits, labels, temperature=3.0):
    # 软标签损失
    soft_loss = nn.KLDivLoss()(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1)
    ) * (temperature ** 2)
    
    # 硬标签损失
    hard_loss = nn.CrossEntropyLoss()(student_logits, labels)
    
    return 0.7 * soft_loss + 0.3 * hard_loss

# 训练学生模型
for batch in dataloader:
    with torch.no_grad():
        teacher_logits = teacher_model(batch)
    
    student_logits = student_model(batch)
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
```

## 2. 模型导出

### ONNX

```python
import torch.onnx

# PyTorch → ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# 验证
import onnx
onnx_model = onnx.load("model.onnx")
onnx.checker.check_model(onnx_model)
```

### TorchScript

```python
# 方法1: Tracing
traced_model = torch.jit.trace(model, example_input)
traced_model.save("model_traced.pt")

# 方法2: Scripting
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# 加载
loaded_model = torch.jit.load("model_traced.pt")
```

## 3. 推理加速

### ONNX Runtime

```python
import onnxruntime as ort

# 创建推理会话
session = ort.InferenceSession(
    "model.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 推理
inputs = {'input': image.numpy()}
outputs = session.run(None, inputs)
```

### TensorRT（NVIDIA GPU）

```python
import tensorrt as trt

# TensorRT优化
# 需要先转换为ONNX，然后用trtexec工具
```

```bash
trtexec --onnx=model.onnx --saveEngine=model.trt --fp16
```

### OpenVINO（Intel CPU）

```bash
# 模型优化
mo --input_model model.onnx --output_dir openvino_model
```

## 4. API服务部署

### FastAPI

```python
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch

app = FastAPI()

# 加载模型（启动时加载一次）
model = torch.load('model.pt')
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 读取图片
    image = Image.open(io.BytesIO(await file.read()))
    
    # 预处理
    input_tensor = preprocess(image)
    
    # 推理
    with torch.no_grad():
        output = model(input_tensor)
    
    # 返回结果
    return {"prediction": output.tolist()}

@app.get("/health")
async def health():
    return {"status": "ok"}
```

**启动：**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Gradio快速部署

```python
import gradio as gr

def predict(image):
    # 推理逻辑
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label()
)

demo.launch(server_name="0.0.0.0", server_port=7860)
```

### Docker容器化

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码和模型
COPY . .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建和运行：**
```bash
docker build -t my-model-api .
docker run -p 8000:8000 my-model-api
```

## 5. 批处理优化

```python
class BatchPredictor:
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size
        self.queue = []
    
    async def predict(self, image):
        # 添加到队列
        self.queue.append(image)
        
        # 达到batch_size或超时，执行批处理
        if len(self.queue) >= self.batch_size:
            batch = torch.stack(self.queue)
            with torch.no_grad():
                results = self.model(batch)
            
            self.queue = []
            return results
```

## 6. 缓存策略

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def predict_cached(image_hash):
    # 缓存预测结果
    return model(image)

def predict_with_cache(image):
    # 计算图片hash
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    return predict_cached(image_hash)
```

## 7. 监控和日志

```python
import logging
from prometheus_client import Counter, Histogram
import time

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus指标
request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.post("/predict")
async def predict(file: UploadFile):
    start_time = time.time()
    request_count.inc()
    
    try:
        result = await model_predict(file)
        logger.info(f"Prediction successful")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise
    finally:
        duration = time.time() - start_time
        request_duration.observe(duration)
```

## 8. 性能基准测试

```python
import time
import numpy as np

def benchmark(model, input_data, num_runs=100):
    # 预热
    for _ in range(10):
        _ = model(input_data)
    
    # 测试
    times = []
    for _ in range(num_runs):
        start = time.time()
        _ = model(input_data)
        times.append(time.time() - start)
    
    print(f"平均延迟: {np.mean(times)*1000:.2f}ms")
    print(f"P50: {np.percentile(times, 50)*1000:.2f}ms")
    print(f"P95: {np.percentile(times, 95)*1000:.2f}ms")
    print(f"P99: {np.percentile(times, 99)*1000:.2f}ms")
```

## 9. 生产环境部署

### Kubernetes部署

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-api
  template:
    metadata:
      labels:
        app: model-api
    spec:
      containers:
      - name: model-api
        image: my-model-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### 负载均衡

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: model-api
spec:
  selector:
    app: model-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 10. 性能优化清单

**推理速度优化：**
- [ ] 使用ONNX Runtime / TensorRT
- [ ] 混合精度推理（FP16）
- [ ] 批处理
- [ ] 模型量化
- [ ] 使用更快的硬件（GPU/TPU）

**显存/内存优化：**
- [ ] 模型量化
- [ ] 模型剪枝
- [ ] KV Cache（生成任务）

**吞吐量优化：**
- [ ] 批处理
- [ ] 异步处理
- [ ] 负载均衡
- [ ] 多进程/多线程

**延迟优化：**
- [ ] 结果缓存
- [ ] 预加载模型
- [ ] 减少数据传输

## 检查点

- [ ] 了解量化、剪枝、蒸馏
- [ ] 会导出ONNX模型
- [ ] 能用FastAPI部署API
- [ ] 能用Docker容器化
- [ ] 了解性能优化方法

模型部署是算法落地的关键！








