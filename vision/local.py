"""本地 CV 识别:MobileNetV3-small 推理,重依赖懒加载并缓存模型。"""

import os

from PIL import Image

import config as cfg

_local_model = None
_local_classes = None
_local_transform = None


def _load_local_model():
    global _local_model, _local_classes, _local_transform
    import json

    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open(os.path.join(os.path.dirname(cfg.CKPT), "classes.json"),
              encoding="utf-8") as f:
        _local_classes = json.load(f)
    model = models.mobilenet_v3_small()
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(_local_classes))
    model.load_state_dict(torch.load(cfg.CKPT, map_location=device))
    model.to(device).eval()
    _local_model = model
    _local_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def classify_local(img_path, topk=3):
    """本地模型识别:返回 {class, prob, top3}。模型未训练时报错提示。"""
    import torch

    if _local_model is None:
        _load_local_model()
    img = Image.open(img_path).convert("RGB")
    x = _local_transform(img).unsqueeze(0)
    device = next(_local_model.parameters()).device
    with torch.no_grad():
        probs = torch.softmax(_local_model(x.to(device)), dim=1)[0]
    vals, idxs = probs.topk(topk)
    return {
        "class": _local_classes[idxs[0].item()],
        "prob": vals[0].item(),
        "top3": [(_local_classes[i.item()], round(v.item(), 3))
                 for i, v in zip(idxs, vals)],
    }
