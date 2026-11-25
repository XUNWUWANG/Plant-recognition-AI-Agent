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


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cases = build()
    n = validate(cases)
    path = os.path.join(st.EVAL_DIR, "eval_candidates.jsonl")
    os.makedirs(st.EVAL_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"已生成 {n} 条候选:{path}")
    print("请人工改写为自然问法,并从中筛选最终 data/eval_queries.jsonl")
