"""知识库检索:按植物名/关键字匹配,返回最相关的条目文本(RAG 的检索端)。"""

import os
import re

import config as cfg

_entries = []  # [(名称, 正文)]


def _load():
    global _entries
    _entries = []
    for fn in os.listdir(cfg.KB_DIR):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(cfg.KB_DIR, fn), encoding="utf-8") as f:
            content = f.read()
        # 按 "## 名称" 切成 (名称, 正文) 列表
        blocks = re.split(r"^##\s+(.+)$", content, flags=re.M)
        i = 1
        while i < len(blocks) - 1:
            name = blocks[i].strip()
            text = blocks[i + 1].strip()
            if name:
                _entries.append((name, text))
            i += 2
    if not _entries:
        raise RuntimeError(f"知识库为空,请检查 {cfg.KB_DIR} 目录")


def _score(query, name, text):
    """简单打分:名称命中权重大,正文词元命中给轻分。"""
    q = query.strip().lower()
    s = 0
    if q and (q in name.lower() or name.lower() in q):
        s += 5
    for token in re.findall(r"[\w一-鿿]{2,}", q):
        if token in text:
            s += 1
    return s


def retrieve(query, topk=2):
    if not _entries:
        _load()
    if not query:
        return []
    scored = [e for e in _entries if _score(query, e[0], e[1]) > 0]
    scored.sort(key=lambda e: _score(query, e[0], e[1]), reverse=True)
    return [f"## {n}\n{t}" for n, t in scored[:topk]]
