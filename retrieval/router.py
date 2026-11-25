"""检索路由与回退纠错(纯函数,不 import 重依赖)。"""

import re

from . import settings as st
from .retrievers import tokenize

ROUTE_BM25, ROUTE_DENSE, ROUTE_HYBRID = "bm25", "dense", "hybrid"

SYMPTOM_WORDS = [
    "发黄", "黄叶", "枯", "萎", "蔫", "掉叶", "落叶", "烂根", "烂", "虫", "蚜",
    "介壳", "病", "黑斑", "白粉", "斑", "不开花", "徒长", "晒伤", "冻伤", "掉苞",
    "浇水", "浇", "施肥", "肥", "换盆", "换土", "修剪", "扦插", "水培", "土培",
    "光照", "晒", "温度", "越冬", "休眠", "繁殖", "加湿", "通风",
]


def collect_names(chunks):
    """植物本名 + 别名 + 学名(小写)。返回 (all_names, primary_names)。"""
    all_names, primary = set(), set()
    for c in chunks:
        p = c["plant"].strip().lower()
        if p:
            primary.add(p)
            all_names.add(p)
        for a in c.get("aliases", []):
            a = a.strip().lower()
            if a:
                all_names.add(a)
    return all_names, primary


def content_tokens(query):
    """与 retrievers.tokenize 一致,供覆盖率计算。"""
    return tokenize(query)


def coverage(query, text):
    """query 的内容词元中出现在 text 里的比例;无词元返回 1.0。"""
    toks = content_tokens(query)
    if not toks:
        return 1.0
    text_l = (text or "").lower()
    hit = sum(1 for t in toks if t in text_l)
    return hit / len(toks)


def _has_symptom(q):
    return any(w in q for w in SYMPTOM_WORDS)


def decide(query, plant_hint, names, primary_names=None):
    """按决策表选择检索路线。names/primary_names 均为小写名称集合。"""
    q = (query or "").strip().lower()
    primary = primary_names if primary_names is not None else names

    if not q and plant_hint:
        return ROUTE_BM25
    has_primary = bool(plant_hint) or any(n in q for n in primary)
    has_alias = any(n in q for n in names if n not in primary)
    has_name = has_primary or has_alias
    has_sym = _has_symptom(q)

    if has_name and has_sym:
        return ROUTE_HYBRID
    if has_primary:
        return ROUTE_BM25
    if has_alias:
        return ROUTE_HYBRID
    if has_sym:
        return ROUTE_DENSE
    return ROUTE_HYBRID


def should_retry(route, top1_text, query, top1_cos=None):
    """检索质量回退:BM25 覆盖不足 / Dense 余弦过低 → 改跑 hybrid。"""
    if route == ROUTE_BM25:
        if coverage(query, top1_text) < st.COVERAGE_THRESHOLD:
            return ROUTE_HYBRID
        return None
    if route == ROUTE_DENSE:
        if top1_cos is not None and top1_cos < st.DENSE_COS_THRESHOLD:
            return ROUTE_HYBRID
        return None
    return None
