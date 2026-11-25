"""Agent 编排:识别(选路线)→ 知识检索 → 生成建议,拼装成 Markdown 展示。"""

import time

import tools

# 常见花卉的花语(子串匹配;非花卉植物不会命中,则不输出花语)
FLOWER_LANG = {
    "向日葵": "忠诚、光明与希望——永远向着太阳,寓意坚定追随。",
    "玫瑰": "热烈的爱情与浪漫,是表达爱意的经典之花。",
    "月季": "美艳而长新,被誉为“花中皇后”,寓意美好与希望。",
    "茉莉": "纯洁、忠贞与质朴,素雅清香。",
    "百合": "百年好合、纯洁庄严,常用于祝福与庆典。",
    "郁金香": "博爱、高雅与富贵,不同花色寓意各异。",
    "康乃馨": "母爱与温馨,是母亲节的象征。",
    "栀子花": "永恒的爱与约定,叶常绿、花清雅。",
    "山茶花": "理想的爱与谦让,亦象征坚毅。",
    "菊花": "清高、长寿与吉祥。",
    "桂花": "收获、吉祥与美好,有“月中桂”之美谈。",
    "杜鹃": "繁荣与吉祥,又名映山红。",
}


def _flower_lang(name):
    """若识别结果是常见花卉,返回其花语文本;否则返回 None。"""
    for key, lang in FLOWER_LANG.items():
        if key in name:
            return lang
    return None


def _conf_color(conf):
    """按置信度给百分比着色:>=0.8 绿、>=0.5 橙、<0.5 红(供界面 Markdown 渲染)。"""
    if conf >= 0.8:
        c = "#2e8b57"
    elif conf >= 0.5:
        c = "#e67e22"
    else:
        c = "#c0392b"
    return f'<span style="color:{c};font-weight:bold">{conf:.0%}</span>'


def _render_advice(adv):
    """把模型返回的 '特征:…\\n光照:…' 文本渲染成 Markdown 列表(兼容全/半角冒号)。"""
    lines = []
    for ln in adv["content"].splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ":" in ln or "：" in ln:
            k, _, v = ln.partition(":") if ":" in ln else ln.partition("：")
            lines.append(f"- **{k.strip()}**: {v.strip()}")
        else:
            lines.append(ln)
    return "\n".join(lines)


def _identify(img_path, route):
    """按路线识别,统一返回 {name, confidence, alternatives}。"""
    if route == "local":
        r = tools.classify_local(img_path)
        top3 = r["top3"]
        return {
            "name": r["class"],
            "confidence": r["prob"],
            "alternatives": [c for c, _ in top3[1:]],
            "source": "本地模型",
        }
    r = tools.classify_plant(img_path)
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
        adv = tools.plant_advice_offline(name, q)
    else:
        adv = tools.plant_advice(name, q)
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
