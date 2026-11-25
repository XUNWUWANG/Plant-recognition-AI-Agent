"""工具层:识别工具(大模型)、知识工具、本地模型工具。"""

import base64
import io
import json
import os
import re

from openai import OpenAI
from PIL import Image

import config as cfg

_SYSTEM = "你是植物识别助手,严格按要求的 JSON 格式输出,不要输出其他内容。"


def _get_client():
    """懒加载客户端:无 API Key 时只有在真正调用大模型时才报错。"""
    return OpenAI(api_key=cfg.require_api_key(), base_url=cfg.BASE_URL)

_IDENTIFY_PROMPT = (
    '识别这张图片中的植物。只输出 JSON,不要输出任何其他内容:'
    '{"plant_name":"最可能的植物名","confidence":0到1之间的小数,'
    '"alternatives":["备选1","备选2"]}'
    '若无法识别,plant_name 填"无法确定",alternatives 给可能的方向。'
)


def _image_to_data_url(img_path, max_size=1024):
    """读图→压缩到 1024px 内→转 JPEG→base64。压缩避免图片过大、格式不支持。"""
    img = Image.open(img_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _parse_json(text):
    """先试标准解析;失败再用正则抽 {...};再失败返回 None。"""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def classify_plant(img_path, topk=3):
    """大模型识别:返回 {plant_name, confidence, alternatives, raw, ok}。"""
    data_url = _image_to_data_url(img_path)
    resp = _get_client().chat.completions.create(
        model=cfg.MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": _IDENTIFY_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content
    obj = _parse_json(raw)
    if obj is None:
        return {"plant_name": "无法确定", "confidence": 0.0,
                "alternatives": [], "raw": raw, "ok": False}
    return {
        "plant_name": obj.get("plant_name", "无法确定"),
        "confidence": float(obj.get("confidence", 0.0)),
        "alternatives": obj.get("alternatives", [])[:topk],
        "raw": raw,
        "ok": True,
    }


_ADVICE_SYSTEM = "你是植物养护助手,基于知识库资料输出建议,简洁口语化。"

_ADVICE_PROMPT = (
    "以下是植物知识库中关于「{plant}」的资料:\n{context}\n\n"
    "请基于资料输出该植物的养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:…"
)

_ADVICE_PROMPT_Q = (
    "以下是植物知识库中与问题相关的资料:\n{context}\n\n"
    "用户的问题:{question}\n"
    "请先基于资料回答该问题,再补充必要的养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:…"
)

_GENERIC_ADVICE_PROMPT = (
    "「{plant}」不在我的植物知识库里。请给出通用养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:(说明知识库暂无此植物,建议谨慎参考)"
)

_GENERIC_ADVICE_PROMPT_Q = (
    "「{plant}」不在我的植物知识库里。用户的问题:{question}\n"
    "请针对问题给出通用养护建议,按以下格式、每条一行:\n"
    "特征:…\n光照:…\n浇水:…\n温度:…\n施肥:…\n备注:(说明知识库暂无此植物,建议谨慎参考)"
)


def plant_advice(plant_name=None, question=None, topk=2):
    """先检索知识库(RAG),再让大模型基于资料生成养护建议。

    返回 {content: 建议文本, source: 'kb' | 'generic'}。
    """
    import rag

    q = (question or "").strip()
    query = q or (plant_name or "")
    ctx, hit = rag.retrieve_hit(query, topk=max(topk, cfg.TOP_K),
                                plant_hint=plant_name)
    label = plant_name or "未知植物"
    if hit and ctx:
        if q:
            prompt = _ADVICE_PROMPT_Q.format(plant=label, context="\n".join(ctx),
                                             question=q)
        else:
            prompt = _ADVICE_PROMPT.format(plant=label, context="\n".join(ctx))
        source = "kb"
    else:
        if q:
            prompt = _GENERIC_ADVICE_PROMPT_Q.format(plant=label, question=q)
        else:
            prompt = _GENERIC_ADVICE_PROMPT.format(plant=label)
        source = "generic"

    resp = _get_client().chat.completions.create(
        model=cfg.MODEL,
        messages=[
            {"role": "system", "content": _ADVICE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return {"content": resp.choices[0].message.content, "source": source}


def plant_advice_offline(plant_name=None, question=None, topk=1):
    """完全离线的养护建议:直接取知识库条目原文返回,不调用大模型 API。

    用于“本地模型识别/纯问答”路线,使该路线真正做到零 API、零额度消耗。
    知识库未命中时返回一句提示,建议改用大模型路线获取更完善建议。
    """
    import rag

    q = (question or "").strip()
    query = q or (plant_name or "")
    ctx, hit = rag.retrieve_hit(query, topk=max(topk, cfg.TOP_K),
                                plant_hint=plant_name)
    if hit and ctx:
        body = ctx[0].split("\n", 1)[1] if "\n" in ctx[0] else ctx[0]
        return {"content": body, "source": "kb"}
    label = plant_name or "这个问题"
    return {"content": f"知识库暂无「{label}」的详细资料。可切换“大模型识别”获取更完善的建议。",
            "source": "generic"}


# ---------- 本地模型识别(懒加载 torch,模型只读一次并缓存) ----------
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
