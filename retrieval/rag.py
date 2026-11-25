"""知识库检索:分块 → 三路检索(BM25/向量/RRF)→ 路由与回退,返回相关片段。

CLI:
    python -m rag --stats    打印 chunk 数量、每植物小节数
    python -m rag --rebuild  强制重建 chunks 缓存
"""

import hashlib
import json
import os
import re
import sys
import time

import config as cfg

_chunks = None
_chunk_by_id = {}
_bm25 = None
_dense = None
_names = None
_primary_names = None
_alias_map = {}   # 名称(本名/别名/学名) → 植物本名
_name_list = []   # 按长度降序,便于优先匹配长名

_ALIAS_LINE = re.compile(r"^别名(?:/学名)?[:：]\s*(.+)$")
_ALIAS_SPLIT = re.compile(r"[、,，;；/]+")
_PUNCT_TRIM = "。.;； \t\r\n"


def _kb_hash():
    """知识库源文件内容哈希,用于判断 chunks 缓存是否过期。"""
    h = hashlib.sha1()
    for fn in sorted(os.listdir(cfg.KB_DIR)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(cfg.KB_DIR, fn), encoding="utf-8") as f:
            h.update(fn.encode("utf-8"))
            h.update(b"\0")
            h.update(f.read().encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()


def _parse_aliases(section, text):
    aliases = []
    for line in text.splitlines():
        m = _ALIAS_LINE.match(line.strip())
        if m:
            raw = m.group(1).split("。", 1)[0]   # 别名行只取第一个句号前
            for part in _ALIAS_SPLIT.split(raw):
                part = part.strip(_PUNCT_TRIM)
                if part and part not in aliases:
                    aliases.append(part)
        elif section == "别名":
            for part in _ALIAS_SPLIT.split(line):
                part = part.strip(_PUNCT_TRIM)
                if part and part not in aliases:
                    aliases.append(part)
    return aliases


def _parse_kb():
    """按 ## 植物 / ### 小节切块;同名植物保留第一条。"""
    chunks, seen_plants, seen_ids = [], set(), set()
    for fn in sorted(os.listdir(cfg.KB_DIR)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(cfg.KB_DIR, fn), encoding="utf-8") as f:
            content = f.read()
        for plant, body in _split(content, r"^##\s+(.+)$"):
            plant = plant.strip()
            if plant in seen_plants:
                continue
            seen_plants.add(plant)
            sections = _split(body, r"^###\s+(.+)$")
            if sections:
                parts = sections
            else:
                parts = [("正文", body)]
            for section, text in parts:
                text = text.strip()
                if len(text) < cfg.CHUNK_MIN_LEN:
                    continue
                section = section.strip() or "正文"
                cid = f"{plant}#{section}"
                n = 2
                while cid in seen_ids:
                    cid = f"{plant}#{section}#{n}"
                    n += 1
                seen_ids.add(cid)
                chunks.append({
                    "id": cid,
                    "plant": plant,
                    "section": section,
                    "text": text,
                    "source": fn,
                    "aliases": _parse_aliases(section, text),
                })
    return chunks


def _split(text, pattern):
    """按标题正则切分,返回 [(标题, 正文)];无标题返回空列表。"""
    parts = re.split(pattern, text, flags=re.M)
    out = []
    i = 1
    while i < len(parts) - 1:
        title = parts[i].strip()
        body = parts[i + 1]
        if title:
            out.append((title, body))
        i += 2
    return out


def _load_chunks(force=False):
    global _chunks, _chunk_by_id
    if _chunks is not None and not force:
        return
    kb_hash = _kb_hash()
    if not force and os.path.exists(cfg.CHUNKS_CACHE):
        try:
            with open(cfg.CHUNKS_CACHE, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("kb_hash") == kb_hash and data.get("chunks"):
                _chunks = data["chunks"]
                _chunk_by_id = {c["id"]: c for c in _chunks}
                return
        except Exception as e:  # noqa: BLE001
            print(f"[rag] chunks 缓存读取失败,重建:{e}")
    _chunks = _parse_kb()
    if not _chunks:
        raise RuntimeError(f"知识库为空,请检查 {cfg.KB_DIR} 目录")
    _chunk_by_id = {c["id"]: c for c in _chunks}
    os.makedirs(os.path.dirname(cfg.CHUNKS_CACHE), exist_ok=True)
    with open(cfg.CHUNKS_CACHE, "w", encoding="utf-8") as f:
        json.dump({"kb_hash": kb_hash, "chunks": _chunks}, f,
                  ensure_ascii=False, indent=1)
    print(f"[rag] 已重建 chunks:{len(_chunks)} 条 → {cfg.CHUNKS_CACHE}")


def _ensure_bm25():
    global _bm25
    _load_chunks()
    if _bm25 is None:
        from . import retrievers
        _bm25 = retrievers.BM25Index(_chunks)
    return _bm25


def _ensure_dense():
    global _dense
    _load_chunks()
    if _dense is None:
        from . import retrievers
        _dense = retrievers.DenseIndex(_chunks)
    return _dense


def _ensure_names():
    global _names, _primary_names, _alias_map, _name_list
    _load_chunks()
    if _names is None:
        from . import router
        _names, _primary_names = router.collect_names(_chunks)
        _alias_map = {}
        for c in _chunks:
            plant = c["plant"]
            _alias_map[plant.lower()] = plant
            for a in c.get("aliases", []):
                _alias_map.setdefault(a.lower(), plant)
        _name_list = sorted(_alias_map, key=len, reverse=True)
    return _names, _primary_names


def _detect_plant(query):
    """查询中出现的植物名/别名 → 植物本名;没有则 None。"""
    _ensure_names()
    q = (query or "").lower()
    for name in _name_list:
        if name in q:
            return _alias_map[name]
    return None


def _filter_plant(ranked, plant):
    return [(cid, s) for cid, s in ranked
            if _chunk_by_id.get(cid, {}).get("plant") == plant]


def get_chunks():
    """供评测与统计复用。"""
    _load_chunks()
    return _chunks


def _fmt(results):
    out = []
    for cid, _score in results:
        c = _chunk_by_id.get(cid)
        if c:
            out.append(f"## {c['plant']} · {c['section']}\n{c['text']}")
    return out


def _search(route, q, topk, plant=None):
    """返回 (ranked, top1_cos)。plant 非空时只在该植物范围内排序。"""
    from . import router
    limit = len(_chunks) if plant else cfg.FUSE_TOP_N

    if route == router.ROUTE_BM25:
        ranked = _ensure_bm25().search(q, limit)
        if plant:
            ranked = _filter_plant(ranked, plant)
        return ranked[:topk], None

    if route == router.ROUTE_DENSE:
        ranked = _ensure_dense().search(q, limit)
        if plant:
            ranked = _filter_plant(ranked, plant)
        ranked = ranked[:topk]
        return ranked, (ranked[0][1] if ranked else None)

    bm = _ensure_bm25().search(q, limit)
    de = _ensure_dense().search(q, limit)
    if plant:
        bm, de = _filter_plant(bm, plant), _filter_plant(de, plant)
    fused = retrievers_rrf(bm, de, topk)
    return fused, (de[0][1] if de else None)


def retrievers_rrf(bm, de, topk):
    from . import retrievers
    return retrievers.rrf_fuse([bm, de], cfg.RRF_K, topk)


def _top1_text(ranked):
    """Top-1 的植物名+小节+正文,用于覆盖率判断(名字命中也能算覆盖)。"""
    if not ranked:
        return ""
    c = _chunk_by_id.get(ranked[0][0], {})
    return f"{c.get('plant', '')} {c.get('section', '')} {c.get('text', '')}"


def _run(query, topk, plant_hint):
    """核心检索流程,返回 (route, retried, ranked, coverage, top1_cos, latency_ms)。"""
    from . import router

    q = (query or "").strip() or (plant_hint or "")
    names, primary = _ensure_names()
    route = router.decide(q, plant_hint, names, primary)
    plant = plant_hint or _detect_plant(q)

    t0 = time.perf_counter()
    ranked, top1_cos = _search(route, q, topk, plant)
    retried = False
    new_route = router.should_retry(route, _top1_text(ranked), q, top1_cos)
    if new_route:
        route = new_route
        retried = True
        ranked, top1_cos = _search(route, q, topk, plant)
    latency = (time.perf_counter() - t0) * 1000.0

    cov = router.coverage(q, _top1_text(ranked)) if q else 1.0
    return route, retried, ranked, cov, top1_cos, latency


def retrieve(query="", topk=None, plant_hint=None):
    """返回 ['## 植物 · 小节\\n正文', ...],长度 ≤ topk。"""
    _load_chunks()
    topk = cfg.TOP_K if topk is None else topk
    if not (query or "").strip() and not plant_hint:
        return []
    _route, _retried, ranked, _cov, _cos, _ms = _run(query, topk, plant_hint)
    return _fmt(ranked)


def retrieve_hit(query="", topk=None, plant_hint=None):
    """返回 (格式化为 ## 植物 · 小节 的片段列表, 是否命中知识库)。

    命中判定:Top-1 覆盖度达到阈值,或 (dense/hybrid 路线下)Top-1 余弦足够高。
    供 tools 判断“库外问题”,不改变 retrieve 的原有行为。
    """
    _load_chunks()
    topk = cfg.TOP_K if topk is None else topk
    if not (query or "").strip() and not plant_hint:
        return [], False
    _route, _retried, ranked, cov, cos, _ms = _run(query, topk, plant_hint)
    hit = cov >= cfg.COVERAGE_THRESHOLD or \
        (cos is not None and cos >= cfg.ANSWER_COS_THRESHOLD)
    return _fmt(ranked), hit


def retrieve_debug(query="", topk=None, plant_hint=None):
    """返回路由/回退/耗时/覆盖度/余弦与结构化结果。"""
    _load_chunks()
    topk = cfg.TOP_K if topk is None else topk
    if not (query or "").strip() and not plant_hint:
        return {"route": None, "retried": False, "latency_ms": 0.0,
                "coverage": 0.0, "top1_cos": None, "results": []}
    route, retried, ranked, cov, cos, ms = _run(query, topk, plant_hint)
    results = []
    for rank, (cid, score) in enumerate(ranked, start=1):
        c = _chunk_by_id.get(cid, {})
        results.append({"id": cid, "plant": c.get("plant"),
                        "section": c.get("section"), "text": c.get("text"),
                        "score": score, "rank": rank})
    return {"route": route, "retried": retried, "latency_ms": ms,
            "coverage": cov, "top1_cos": cos, "results": results}


def _naive_score(query, plant, text):
    """复刻原 rag._score:名字命中 +5,正文词元命中 +1。"""
    q = (query or "").strip().lower()
    s = 0
    if q and (q in plant.lower() or plant.lower() in q):
        s += 5
    for token in re.findall(r"[\w一-鿿]{2,}", q):
        if token in text:
            s += 1
    return s


def naive_rank(query, topk=None):
    """第 1 组基线的排序结果:[(chunk_id, score)]。"""
    _load_chunks()
    topk = cfg.TOP_K if topk is None else topk
    q = (query or "").strip()
    if not q:
        return []
    scored = [(_naive_score(q, c["plant"], c["text"]), c["id"])
              for c in _chunks]
    scored = [x for x in scored if x[0] > 0]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(cid, s) for s, cid in scored[:topk]]


def naive_retrieve(query, topk=None):
    """第 1 组基线:原字面匹配逻辑,作用于 chunk 粒度。"""
    return _fmt(naive_rank(query, topk))


def _stats():
    chunks = get_chunks()
    per = {}
    alias_n = 0
    for c in chunks:
        per[c["plant"]] = per.get(c["plant"], 0) + 1
        alias_n += len(c.get("aliases", []))
    print(f"chunks 总数:{len(chunks)}")
    print(f"植物数:{len(per)},别名/学名条目:{alias_n}")
    for p, n in per.items():
        print(f"  {p}: {n} 节")


def _rebuild():
    for path in (cfg.CHUNKS_CACHE, cfg.EMBED_CACHE):
        if os.path.exists(path):
            os.remove(path)
            print(f"已删除 {path}")
    _load_chunks(force=True)
    print(f"重建完成:{len(_chunks)} 条")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--stats" in sys.argv:
        _stats()
    elif "--rebuild" in sys.argv:
        _rebuild()
    else:
        print(__doc__)
