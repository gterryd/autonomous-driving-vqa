# 分布式训练（Day 31-33）

## 为什么需要分布式训练

单GPU训练大模型的问题：
- 显存不够（7B模型需要28GB+）
- 训练太慢（几周甚至几月）
- 无法扩展

## 核心技术

### 1. 数据并行（Data Parallelism）

**原理：** 每个GPU复制完整模型，处理不同数据

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# 初始化进程组
dist.init_process_group(backend='nccl')

# 创建模型
model = MyModel()
model = model.to(rank)  # rank是当前GPU编号

# 包装为DDP
model = DDP(model, device_ids=[rank])

# 训练循环
for data, target in dataloader:
    data, target = data.to(rank), target.to(rank)
    
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

**启动命令：**
```bash
torchrun --nproc_per_node=4 train.py
```

### 2. 模型并行（Model Parallelism）

**原理：** 模型切分到多个GPU

```python
# 简单模型并行
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
        self.layer2 = nn.Linear(1000, 1000).to('cuda:1')
        self.layer3 = nn.Linear(1000, 10).to('cuda:1')
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        x = self.layer3(x)
        return x
```

### 3. DeepSpeed ZeRO

**ZeRO Stage 1-3：**
- Stage 1: 切分优化器状态
- Stage 2: 切分梯度
- Stage 3: 切分模型参数

```python
import deepspeed

# 配置文件 ds_config.json
{
    "train_batch_size": 32,
    "gradient_accumulation_steps": 4,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 3e-5
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}

# 初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config="ds_config.json"
)

# 训练
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**启动：**
```bash
deepspeed --num_gpus=4 train.py --deepspeed_config ds_config.json
```

### 4. FSDP (Fully Sharded Data Parallel)

**PyTorch原生分布式方案：**

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

model = MyLargeModel()
model = FSDP(model)

# 训练同DDP
```

## 实战：多GPU训练CLIP

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers import CLIPModel, CLIPProcessor

def setup(rank, world_size):
    """初始化分布式环境"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """清理"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    
    # 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 数据
    dataset = MyDataset()
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size,
        rank=rank
    )
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )
    
    # 训练
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(10):
        sampler.set_epoch(epoch)  # 重要！
        
        for batch in dataloader:
            batch = {k: v.to(rank) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:  # 只在主进程打印
                print(f'Epoch {epoch}, Loss: {loss.item()}')
    
    cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(
        train_ddp,
        args=(world_size,),
        nprocs=world_size
    )
```

## 混合精度训练

**自动混合精度（AMP）：**

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    
    # 使用fp16前向传播
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    
    # 缩放loss，反向传播
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**节省显存约50%，速度提升2-3倍！**

## 梯度累积

**模拟更大batch size：**

```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output, target)
    
    # 除以累积步数
    loss = loss / accumulation_steps
    loss.backward()
    
    # 每N步更新一次
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 梯度检查点

**节省显存，代价是速度：**

```python
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def forward(self, x):
        # 使用checkpoint
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x
```

## 常见问题

**Q: 多GPU训练速度没提升？**
- 检查数据加载是否是瓶颈
- 增加 `num_workers`
- 使用 `pin_memory=True`

**Q: 显存不够？**
1. 减小batch size
2. 使用梯度累积
3. 使用混合精度
4. 使用梯度检查点
5. 使用DeepSpeed ZeRO-3

**Q: Loss不收敛？**
- 检查学习率（多GPU可能需要调整）
- 检查数据是否正确划分
- 检查BN层（用SyncBatchNorm）

## 监控和调试

```python
# 打印GPU使用情况
import torch
print(torch.cuda.memory_summary())

# 查看进程状态
nvidia-smi

# 性能分析
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    model(inputs)

print(prof.key_averages().table())
```

## 实战作业

1. 用DDP训练CIFAR-10（2个GPU）
2. 用DeepSpeed训练CLIP（4个GPU）
3. 对比单GPU vs 多GPU的速度

## 检查点

- [ ] 理解数据并行和模型并行
- [ ] 能用DDP训练模型
- [ ] 了解DeepSpeed ZeRO
- [ ] 会用混合精度和梯度累积

分布式训练是大模型必备技能！








