"""本地训练(20 类):在 data/<植物名>/ 上微调 MobileNetV3-small。

支持三种训练策略,便于在报告中做受控对比:
  --mode full    全参数微调(弱增强、无权重衰减、不早停),代表“朴素基线”
  --mode reg     全参数微调 + 强数据增强 + 权重衰减
  --mode frozen  冻结骨干、只训分类头 + 强增强 + 权重衰减 + 早停(最终采用)

用法:
  python train.py --mode full
  python train.py --mode reg
  python train.py --mode frozen

产物:models/<mode>.pt、models/hist_<mode>.json、models/classes.json;
      当 mode=frozen 时同时覆盖 models/best.pt(演示用最终模型)。
固定随机种子(42),保证三种策略使用同一训练/验证划分,对比公平。
"""

import argparse
import json
import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import config as cfg

# ---------- 参数 ----------
ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["full", "reg", "frozen"], default="frozen")
ap.add_argument("--epochs", type=int, default=None)
args = ap.parse_args()
MODE = args.mode

os.makedirs(os.path.join(cfg.BASE_DIR, "models"), exist_ok=True)

# 固定随机种子,保证三种模式的划分/初始化一致
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ---------- 数据增强 ----------
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transform_eval = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
transform_min = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), norm,
])
transform_rich = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), norm,
])
transform_train = transform_min if MODE == "full" else transform_rich

# ---------- 数据集(固定划分) ----------
ds = datasets.ImageFolder(cfg.DATA_DIR)
num_classes = len(ds.classes)
print(f"[{MODE}] 类别({num_classes}): {ds.classes}")
print(f"[{MODE}] 图片总数: {len(ds)}")

n = len(ds)
n_val = max(1, int(0.15 * n))
indices = list(range(n))
random.shuffle(indices)   # 种子已固定
train_idx, val_idx = indices[n_val:], indices[:n_val]
print(f"[{MODE}] 训练/验证: {len(train_idx)}/{len(val_idx)}")

train_ds = torch.utils.data.Subset(
    datasets.ImageFolder(cfg.DATA_DIR, transform=transform_train), train_idx)
val_ds = torch.utils.data.Subset(
    datasets.ImageFolder(cfg.DATA_DIR, transform=transform_eval), val_idx)

# ---------- 模型 ----------
model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

if MODE == "frozen":
    for p in model.features.parameters():
        p.requires_grad = False
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"[{MODE}] 可训练参数 {sum(p.numel() for p in trainable):,} / 总 {sum(p.numel() for p in model.parameters()):,}")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"[{MODE}] 设备: {device}")

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

criterion = nn.CrossEntropyLoss()
wd = 0.0 if MODE == "full" else 1e-3
optimizer = optim.Adam(trainable, lr=1e-3, weight_decay=wd)

# ---------- 训练 ----------
EPOCHS = args.epochs or (30 if MODE == "frozen" else 12)
PATIENCE = 5 if MODE == "frozen" else 10**9  # 非 frozen 不早停
best_acc = 0.0
no_improve = 0
history = []
ckpt = os.path.join("models", f"{MODE}.pt")

for epoch in range(EPOCHS):
    model.train()
    run_loss, tr_correct, tr_total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        tr_correct += (out.argmax(1) == y).sum().item()
        tr_total += y.size(0)

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            v_correct += (model(x).argmax(1) == y).sum().item()
            v_total += y.size(0)

    train_acc = tr_correct / tr_total
    val_acc = v_correct / v_total
    history.append({"epoch": epoch + 1, "loss": round(run_loss / len(train_loader), 4),
                    "train_acc": round(train_acc, 4), "val_acc": round(val_acc, 4)})
    print(f"[{MODE}] epoch {epoch+1}/{EPOCHS} loss {run_loss/len(train_loader):.3f} "
          f"train_acc {train_acc:.3f} val_acc {val_acc:.3f}", flush=True)

    if val_acc > best_acc:
        best_acc = val_acc
        no_improve = 0
        torch.save(model.state_dict(), ckpt)
        with open(os.path.join("models", "classes.json"), "w", encoding="utf-8") as f:
            json.dump(ds.classes, f, ensure_ascii=False)
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"[{MODE}] 连续 {PATIENCE} 轮无提升,早停于 epoch {epoch+1}", flush=True)
            break

with open(os.path.join("models", f"hist_{MODE}.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=1)

if MODE == "frozen":
    shutil.copyfile(ckpt, cfg.CKPT)  # 演示用最终模型

print(f"[{MODE}] 完成,最佳验证准确率: {best_acc:.3f}", flush=True)
