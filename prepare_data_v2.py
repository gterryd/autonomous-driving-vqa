import json
import os
import random
from collections import defaultdict, Counter

NUSCENES_ROOT = "data/nuscenes/v1.0-mini"
META_DIR = os.path.join(NUSCENES_ROOT, "v1.0-mini")
OUT_DIR = "data/driving_vqa_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# 加载元数据
with open(os.path.join(META_DIR, "sample_data.json")) as f:
    sample_data = json.load(f)
with open(os.path.join(META_DIR, "sample_annotation.json")) as f:
    annotations = json.load(f)
with open(os.path.join(META_DIR, "category.json")) as f:
    categories = json.load(f)
with open(os.path.join(META_DIR, "instance.json")) as f:
    instances = json.load(f)

# sample_token -> CAM_FRONT 图片路径
token_to_img = {}
for item in sample_data:
    if "samples/CAM_FRONT/" in item["filename"] and item["is_key_frame"]:
        token_to_img[item["sample_token"]] = item["filename"]

print(f"CAM_FRONT key frames: {len(token_to_img)}")

# instance_token -> category_token
instance_to_cat_token = {inst["token"]: inst["category_token"] for inst in instances}
# category_token -> name
cat_token_to_name = {cat["token"]: cat["name"] for cat in categories}

# 简化类别名
def simplify(name):
    if name.startswith("vehicle.car"): return "car"
    if name.startswith("vehicle.truck"): return "truck"
    if name.startswith("vehicle.bus"): return "bus"
    if name.startswith("vehicle.motorcycle"): return "motorcycle"
    if name.startswith("vehicle.bicycle"): return "bicycle"
    if name.startswith("human.pedestrian"): return "pedestrian"
    if name.startswith("movable_object.trafficcone"): return "traffic cone"
    if name.startswith("movable_object.barrier"): return "barrier"
    return None

# sample_token -> {category: count}
sample_to_counts = defaultdict(lambda: defaultdict(int))
for ann in annotations:
    token = ann["sample_token"]
    cat_token = instance_to_cat_token.get(ann["instance_token"])
    if cat_token:
        cat_name = simplify(cat_token_to_name.get(cat_token, ""))
        if cat_name:
            sample_to_counts[token][cat_name] += 1

# 生成QA
CATEGORIES = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian", "traffic cone", "barrier"]

def count_to_str(n):
    if n == 0: return "0"
    if n == 1: return "1"
    if n == 2: return "2"
    if n == 3: return "3"
    if n >= 4: return "more than 3"

qa_data = []
for token, img_path in token_to_img.items():
    counts = sample_to_counts[token]
    for cat in CATEGORIES:
        n = counts.get(cat, 0)
        # 计数问题
        qa_data.append({
            "image": img_path,
            "question": f"How many {cat}s are there?",
            "answer": count_to_str(n),
            "category": "count"
        })
        # 存在性问题
        qa_data.append({
            "image": img_path,
            "question": f"Is there a {cat} in the scene?",
            "answer": "yes" if n > 0 else "no",
            "category": "existence"
        })

print(f"Total QA pairs: {len(qa_data)}")

# 统计答案分布
all_answers = [d["answer"] for d in qa_data]
print("Answer distribution:", Counter(all_answers).most_common(10))

# 按sample_token划分train/val（8:2）
all_tokens = list(token_to_img.keys())
random.seed(42)
random.shuffle(all_tokens)
split = int(len(all_tokens) * 0.8)
train_tokens = set(all_tokens[:split])
val_tokens = set(all_tokens[split:])

train_data = [d for d in qa_data if any(token_to_img[t] == d["image"] for t in train_tokens)]
val_data = [d for d in qa_data if any(token_to_img[t] == d["image"] for t in val_tokens)]

# 更高效的划分
token_split = {}
for t in train_tokens: token_split[token_to_img[t]] = "train"
for t in val_tokens: token_split[token_to_img[t]] = "val"

train_data = [d for d in qa_data if token_split.get(d["image"]) == "train"]
val_data = [d for d in qa_data if token_split.get(d["image"]) == "val"]

print(f"train: {len(train_data)}, val: {len(val_data)}")

with open(os.path.join(OUT_DIR, "qa_train.json"), "w") as f:
    json.dump(train_data, f)
with open(os.path.join(OUT_DIR, "qa_val.json"), "w") as f:
    json.dump(val_data, f)

print(f"Saved to {OUT_DIR}/")
