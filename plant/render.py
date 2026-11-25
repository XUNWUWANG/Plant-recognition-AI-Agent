"""结果渲染:花语、置信度着色、建议文本转 Markdown 列表。"""

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
