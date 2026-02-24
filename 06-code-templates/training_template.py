# 标准训练模板（直接复用）

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
import argparse
from pathlib import Path

class Config:
    """配置类"""
    def __init__(self):
        # 数据
        self.data_path = './data'
        self.batch_size = 32
        self.num_workers = 4
        
        # 模型
        self.model_name = 'resnet18'
        self.num_classes = 10
        
        # 训练
        self.epochs = 10
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.warmup_epochs = 1
        
        # 其他
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_dir = './checkpoints'
        self.log_interval = 10

class Trainer:
    """通用训练器"""
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 最佳指标
        self.best_acc = 0.0
        
        # 创建保存目录
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config.epochs}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            
            # 前向
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # 反向
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in tqdm(self.val_loader, desc='Validating'):
            inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
        }
        
        # 保存最新
        path = Path(self.config.save_dir) / 'latest.pth'
        torch.save(checkpoint, path)
        
        # 保存最佳
        if is_best:
            path = Path(self.config.save_dir) / 'best.pth'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_acc = checkpoint['best_acc']
        return checkpoint['epoch']
    
    def fit(self):
        """完整训练流程"""
        for epoch in range(self.config.epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate()
            
            # 学习率调度
            self.scheduler.step()
            
            # 日志
            print(f'\nEpoch {epoch+1}/{self.config.epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # WandB日志（可选）
            # wandb.log({
            #     'train_loss': train_loss,
            #     'train_acc': train_acc,
            #     'val_loss': val_loss,
            #     'val_acc': val_acc,
            #     'lr': self.optimizer.param_groups[0]['lr']
            # })
            
            # 保存检查点
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
            self.save_checkpoint(epoch, is_best)
        
        print(f'\n训练完成！最佳验证准确率: {self.best_acc:.2f}%')

# 使用示例
if __name__ == '__main__':
    # 配置
    config = Config()
    
    # 数据（需要自己实现）
    # train_dataset = YourDataset(...)
    # val_dataset = YourDataset(...)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 模型（需要自己定义）
    # model = YourModel(...)
    
    # 训练
    # trainer = Trainer(model, train_loader, val_loader, config)
    # trainer.fit()
    
    pass















