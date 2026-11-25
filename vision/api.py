"""大模型视觉识别:图片压缩、JSON 解析、DeepSeek 多模态调用。"""

import base64
import io
import json
import re

from PIL import Image

import config as cfg
from llm import get_client

_SYSTEM = "你是植物识别助手,严格按要求的 JSON 格式输出,不要输出其他内容。"

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
    resp = get_client().chat.completions.create(
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
