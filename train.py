"""
train.py
自动驾驶 VQA 模型训练脚本
运行：python train.py
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
TRAIN_JSON  = "./data/driving_vqa_v2/qa_train.json"
VAL_JSON    = "./data/driving_vqa_v2/qa_val.json"
DATA_ROOT   = "./data/nuscenes/v1.0-mini"
SAVE_PATH   = "./driving_vqa_v2_best.pth"
CLIP_MODEL  = "openai/clip-vit-base-patch32"
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────


class DrivingVQADataset(Dataset):
    def __init__(self, json_path, processor, answer_to_idx=None):
        with open(json_path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor
        self.data_root = Path(DATA_ROOT)

        if answer_to_idx is None:
            answers = sorted(set(d["answer"] for d in self.data))
            self.answer_to_idx = {a: i for i, a in enumerate(answers)}
        else:
            self.answer_to_idx = answer_to_idx

        self.idx_to_answer = {i: a for a, i in self.answer_to_idx.items()}
        self.num_answers = len(self.answer_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self.data_root / item["image"]
        image = Image.open(img_path).convert("RGB")

        inputs = self.processor(
            images=image,
            text=item["question"],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        label = self.answer_to_idx.get(item["answer"], 0)
        return {
            "pixel_values":  inputs["pixel_values"].squeeze(0),
            "input_ids":     inputs["input_ids"].squeeze(0),
            "attention_mask":inputs["attention_mask"].squeeze(0),
            "label":         torch.tensor(label, dtype=torch.long),
        }


class DrivingVQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL)
        for p in self.clip.parameters():
            p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_answers),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            img_out = self.clip.vision_model(pixel_values=pixel_values)
            txt_out = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
            img_feat = self.clip.visual_projection(img_out.pooler_output)
            txt_feat = self.clip.text_projection(txt_out.pooler_output)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
        return self.classifier(torch.cat([img_feat, txt_feat], dim=-1))


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            preds = logits.argmax(dim=1)
            correct += preds.eq(batch["label"]).sum().item()
            total   += batch["label"].size(0)
    return 100.0 * correct / total


def main():
    print(f"设备: {DEVICE}")
    print("加载 CLIP Processor...")
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    print("加载数据集...")
    train_ds = DrivingVQADataset(TRAIN_JSON, processor)
    val_ds   = DrivingVQADataset(VAL_JSON,   processor, answer_to_idx=train_ds.answer_to_idx)
    print(f"训练集: {len(train_ds)} 条，验证集: {len(val_ds)} 条，答案类别: {train_ds.num_answers}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("加载模型...")
    model = DrivingVQAModel(num_answers=train_ds.num_answers).to(DEVICE)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    print("\n开始训练...\n")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = correct = total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{NUM_EPOCHS}"):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["pixel_values"], batch["input_ids"], batch["attention_mask"])
            loss   = criterion(logits, batch["label"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += preds.eq(batch["label"]).sum().item()
            total   += batch["label"].size(0)

        train_acc = 100.0 * correct / total
        val_acc   = evaluate(model, val_loader)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "answer_to_idx": train_ds.answer_to_idx,
                "num_answers": train_ds.num_answers,
            }, SAVE_PATH)
            tag = " ← best"
        else:
            tag = ""

        print(f"Epoch {epoch:02d}  loss={total_loss/len(train_loader):.4f}  "
              f"train={train_acc:.1f}%  val={val_acc:.1f}%{tag}")

    print(f"\n✅ 训练完成！最佳验证准确率: {best_acc:.1f}%")
    print(f"模型已保存到 {SAVE_PATH}")


if __name__ == "__main__":
    main()
