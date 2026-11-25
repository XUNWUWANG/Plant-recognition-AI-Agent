"""按小节模板生成候选评测题(仅供人工改写,不直接作为最终评测集)。

用法:
    python scripts/build_eval.py
输出:
    data/eval_candidates.jsonl
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import rag, settings as st  # noqa: E402

TEMPLATES = {
    "光照": ("{plant}需要多少光照？", "entity"),
    "浇水": ("{plant}多久浇一次水？", "entity"),
    "施肥": ("{plant}怎么施肥？", "entity"),
    "温度": ("{plant}适宜温度是多少？", "entity"),
    "特征": ("{plant}有什么特点？", "entity"),
    "病虫害": ("{plant}常见病虫害怎么处理？", "symptom"),
    "常见问题": ("{plant}养护常见问题有哪些？", "symptom"),
}


def build():
    chunks = rag.get_chunks()
    by_plant = {}
    for c in chunks:
        by_plant.setdefault(c["plant"], []).append(c)
    out = []
    idx = 0
    for plant, cs in by_plant.items():
        for c in cs:
            tpl = TEMPLATES.get(c["section"])
            if not tpl:
                continue
            query, typ = tpl
            idx += 1
            out.append({
                "id": f"c{idx:03d}",
                "query": query.format(plant=plant),
                "type": typ,
                "plant_hint": plant,
                "relevant": [c["id"]],
                "difficulty": "easy",
                "note": "模板候选,需人工改写",
            })
    return out


def validate(cases):
    ids = {c["id"] for c in rag.get_chunks()}
    for c in cases:
        for r in c["relevant"]:
            assert r in ids, f"未知 chunk_id: {r}"
        assert len(c["query"]) >= 3, f"query 过短: {c}"
    return len(cases)


# ---------- 盲症状扩展集校验(CR-011 阶段 A) ----------

BLIND_EXPECTED = {"symptom": 40, "multi": 20, "unanswerable": 30}


def _load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _grams(text, n=9):
    return {text[i:i + n] for i in range(0, len(text) - n + 1)}


def blind_check(path=None):
    """校验盲题:无名称泄漏、相关 id 存在、连续重合 ≤8 字、唯一性。"""
    path = path or os.path.join(st.EVAL_DIR, "blind_queries.jsonl")
    cases = _load_jsonl(path)
    chunks = rag.get_chunks()
    by_id = {c["id"]: c for c in chunks}

    names = set()
    for c in chunks:
        names.add(c["plant"].strip().lower())
        for a in c.get("aliases", []):
            names.add(a.strip().lower())
    names = {n for n in names if len(n) >= 2}
    grams = set()
    for c in chunks:
        grams |= _grams(c["text"])

    errors, stats, seen = [], {}, set()
    for case in cases:
        cid = case.get("id", "?")
        q = (case.get("query") or "").strip()
        typ = case.get("type")
        stats[typ] = stats.get(typ, 0) + 1

        if cid in seen:
            errors.append(f"{cid}: id 重复")
        seen.add(cid)
        if case.get("plant_hint") is not None:
            errors.append(f"{cid}: plant_hint 必须为 null")
        if not q:
            errors.append(f"{cid}: query 为空")

        ql = q.lower()
        for name in sorted(names):
            if name in ql:
                errors.append(f"{cid}: query 含植物名/别名/学名「{name}」")

        rel = case.get("relevant", [])
        for rid in rel:
            if rid not in by_id:
                errors.append(f"{cid}: relevant 不存在 {rid}")

        tp = case.get("target_plant")
        if typ == "unanswerable":
            if rel:
                errors.append(f"{cid}: unanswerable 的 relevant 必须为空")
            if tp is not None:
                errors.append(f"{cid}: unanswerable 的 target_plant 必须为 null")
        else:
            if typ not in ("symptom", "multi"):
                errors.append(f"{cid}: 未知 type {typ}")
            if not tp:
                errors.append(f"{cid}: 缺少 target_plant")
            if not (1 <= len(rel) <= 2):
                errors.append(f"{cid}: relevant 需 1-2 条")
            plants = {by_id[r]["plant"] for r in rel if r in by_id}
            if plants != {tp}:
                errors.append(
                    f"{cid}: relevant 植物 {sorted(plants)} 与 target_plant {tp} 不一致")

        for g in _grams(q):
            if g in grams:
                errors.append(f"{cid}: 与知识库连续重合超过 8 字「{g}」")
                break

    for typ, want in BLIND_EXPECTED.items():
        got = stats.get(typ, 0)
        if got != want:
            errors.append(f"类型 {typ} 数量 {got} != {want}")
    total = sum(stats.values())
    if total != sum(BLIND_EXPECTED.values()):
        errors.append(f"总数 {total} != {sum(BLIND_EXPECTED.values())}")

    print(f"盲题文件:{path}")
    print(f"总数:{total}(" + " / ".join(
        f"{k} {stats.get(k, 0)}" for k in BLIND_EXPECTED) + ")")
    if errors:
        print(f"校验失败:{len(errors)} 处")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("校验通过:无名称泄漏;related 均存在;连续重合 ≤8 字;"
          "target_plant 与 relevant 一致")
    return True


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if "--blind-check" in sys.argv:
        blind_check()
        raise SystemExit(0)
    cases = build()
    n = validate(cases)
    path = os.path.join(st.EVAL_DIR, "eval_candidates.jsonl")
    os.makedirs(st.EVAL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"已生成 {n} 条候选:{path}")
    print("请人工改写为自然问法,并从中筛选最终 data/eval_queries.jsonl")
