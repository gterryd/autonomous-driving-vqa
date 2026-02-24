# 数据加载模板

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple

# ====================
# 图像分类数据集
# ====================
class ImageClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        root_dir/
            train/
                class1/
                    img1.jpg
                class2/
            val/
        """
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        
        # 收集所有图片路径和标签
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.glob('*.jpg'):
                self.samples.append((img_path, self.class_to_idx[cls]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ====================
# 图文对数据集
# ====================
class ImageTextDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None, tokenizer=None):
        """
        annotation_file: JSON格式
        [
            {"image": "img1.jpg", "caption": "a cat"},
            {"image": "img2.jpg", "caption": "a dog"}
        ]
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.tokenizer = tokenizer
        
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # 加载图片
        img_path = self.image_dir / ann['image']
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理文本
        caption = ann['caption']
        if self.tokenizer:
            caption = self.tokenizer(caption)
        
        return image, caption

# ====================
# 多模态检索数据集
# ====================
class RetrievalDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # captions.json: {"img1.jpg": ["caption1", "caption2"], ...}
        with open(captions_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_files = list(self.data.keys())
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # 图片
        image = Image.open(self.image_dir / img_file).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 随机选一个caption
        captions = self.data[img_file]
        caption = captions[torch.randint(len(captions), (1,)).item()]
        
        return {
            'image': image,
            'text': caption,
            'image_id': idx
        }

# ====================
# 数据增强
# ====================
from torchvision import transforms

# 训练时的数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证/测试时的数据增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ====================
# Collate函数（批处理）
# ====================
def collate_fn_retrieval(batch):
    """自定义批处理函数"""
    images = torch.stack([item['image'] for item in batch])
    texts = [item['text'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch])
    
    return {
        'images': images,
        'texts': texts,
        'image_ids': image_ids
    }

# ====================
# 使用示例
# ====================
if __name__ == '__main__':
    # 图像分类
    train_dataset = ImageClassificationDataset(
        root_dir='./data/imagenet',
        split='train',
        transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 图文检索
    retrieval_dataset = RetrievalDataset(
        image_dir='./data/coco/images',
        captions_file='./data/coco/captions.json',
        transform=train_transform
    )
    retrieval_loader = DataLoader(
        retrieval_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn_retrieval,
        num_workers=4
    )
    
    # 测试
    for batch in train_loader:
        images, labels = batch
        print(f"Images: {images.shape}, Labels: {labels.shape}")
        break
    
    for batch in retrieval_loader:
        print(f"Images: {batch['images'].shape}")
        print(f"Texts: {len(batch['texts'])}")
        print(f"Image IDs: {batch['image_ids'].shape}")
        break















