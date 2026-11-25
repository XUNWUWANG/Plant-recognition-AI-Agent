"""编排:识别(选路线)→ 知识检索 → 生成建议,拼装成 Markdown 展示。"""

import time

from advice.service import plant_advice, plant_advice_offline
from vision.api import classify_plant
from vision.local import classify_local

from .render import _conf_color, _flower_lang, _render_advice


def _identify(img_path, route):
    """按路线识别,统一返回 {name, confidence, alternatives}。"""
    if route == "local":
        r = classify_local(img_path)
        top3 = r["top3"]
        return {
            "name": r["class"],
            "confidence": r["prob"],
            "alternatives": [c for c, _ in top3[1:]],
            "source": "本地模型",
        }
    r = classify_plant(img_path)
    return {
        "name": r["plant_name"],
        "confidence": r["confidence"],
        "alternatives": r["alternatives"],
        "source": "大模型",
    }


def agent_handle(img_path=None, route="api", question=None):
    """主流程:识别(可选)→ 检索/生成建议 → Markdown。

    img_path 可为 None(纯文字问答)。route: 'api' / 'local'。
    """
    q = (question or "").strip()
    if not img_path and not q:
        return "请上传植物照片,或输入一个问题。"

    t0 = time.perf_counter()
    r1 = _identify(img_path, route) if img_path else None
    name = r1["name"] if r1 else None

    md = []
    if r1:
        md.append(f"## 识别结果: {name}({r1['source']})")
        md.append(f"置信度: {_conf_color(r1['confidence'])}")
        if r1["alternatives"]:
            md.append("备选: " + "、".join(r1["alternatives"]))
        if r1["confidence"] < 0.4:
            md.append("> 识别置信度较低,建议人工确认。")
    else:
        md.append("## 养护问答")
    if q:
        md.append(f"**问题:** {q}")

    # 本地路线:识别与建议均离线(零 API 额度);大模型路线:建议由大模型生成
    if route == "local":
        adv = plant_advice_offline(name, q)
    else:
        adv = plant_advice(name, q)
    md.append("\n### 养护建议")
    if adv["source"] == "generic":
        md.append("> 知识库暂无该植物的详细资料,以下为通用建议,仅供参考。")
    md.append(_render_advice(adv))

    fl = _flower_lang(name) if name else None
    if fl:
        md.append("\n### 花语")
        md.append(fl)

    elapsed = time.perf_counter() - t0
    route_label = r1["source"] if r1 else ("本地检索" if q else "知识库")
    md.append(f"\n> 本次用时 {elapsed:.1f} 秒,路线:{route_label}。")

    return "\n".join(md)
