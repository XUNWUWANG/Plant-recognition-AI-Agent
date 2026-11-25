"""检索评测:P1 公平协议、P2 原始协议,以及 CR-012 指标与产物。

用法(需在项目根目录运行):
    python -m retrieval.eval
    python -m retrieval.eval --configs P2:bm25,P2:dense
    python -m retrieval.eval --configs bm25,dense       # 简写映射到 P2
    python -m retrieval.eval --configs P1:adaptive-filter  # ASCII "-" 也可
    python -m retrieval.eval --limit 50
    python -m retrieval.eval --cases data/eval/blind_queries.jsonl \
        --configs naive,bm25,dense,hybrid,adaptive \
        --out 指标/retrieval/盲症状对比表.md \
        --err 指标/retrieval/盲症状错误案例.md

P1(主表):纯检索组统一实体过滤;adaptive 为产品路径;另设消融。
P2(辅表):原五组行为不变(adaptive 含自动实体约束)。
"""

import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime

import numpy as np

from . import rag, retrievers, router
from . import settings as st

# 延迟口径(CR-012):固定 dtype / 线程数 / warm-up / 重复次数
_DTYPE_NAME = "bfloat16"
_THREADS = 4
_REPEATS = 3
_BOOTSTRAP_N = 1000
_SEED = 20261125
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, str(_THREADS))

P1_CONFIGS = [
    "P1:dense+filter",
    "P1:hybrid+filter",
    "P1:bm25+filter",
    "P1:adaptive",
    "P1:adaptive−filter",
    "P1:adaptive−retry",
]
P2_CONFIGS = ["P2:naive", "P2:bm25", "P2:dense", "P2:hybrid", "P2:adaptive"]
ALL_CONFIGS = P1_CONFIGS + P2_CONFIGS

_SHORTHAND = {"naive": "P2:naive", "bm25": "P2:bm25", "dense": "P2:dense",
              "hybrid": "P2:hybrid", "adaptive": "P2:adaptive"}
_ASCII_ALIASES = {
    "P1:adaptive-filter": "P1:adaptive−filter",
    "P1:adaptive-retry": "P1:adaptive−retry",
    "adaptive-filter": "P1:adaptive−filter",
    "adaptive-retry": "P1:adaptive−retry",
}


class _FixedDenseIndex(retrievers.DenseIndex):
    """评测专用:显式固定 dtype,避免随环境自动选 bf16/fp32。"""

    def _try_load(self, cls, name):
        import torch
        try:
            return cls(name, trust_remote_code=True,
                       model_kwargs={"dtype": torch.bfloat16})
        except Exception as e:  # noqa: BLE001
            print(f"[eval] Sentence-Transformers 固定 dtype 加载 {name} 失败:{e}")
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                try:
                    return cls(name, trust_remote_code=True,
                               model_kwargs={"dtype": torch.bfloat16})
                except Exception as e2:  # noqa: BLE001
                    print(f"[eval] 镜像加载仍失败:{e2}")
        return None

    def _load_hf(self, name):
        import torch
        try:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                name, trust_remote_code=True, dtype=torch.bfloat16).eval()
            return {"tok": tok, "model": model}
        except Exception as e:  # noqa: BLE001
            print(f"[eval] transformers 固定 dtype 加载 {name} 失败:{e}")
            return None


def _setup_runtime():
    import torch
    torch.set_num_threads(_THREADS)
    rag._load_chunks()
    rag._dense = _FixedDenseIndex(rag.get_chunks())


def display_name(key):
    return key.split(":", 1)[1]


def _label(key):
    return f"{display_name(key)}({key.split(':', 1)[0]})"


def _is_adaptive(key):
    return "adaptive" in key


def load_cases(path=None, limit=None):
    path = path or os.path.join(st.EVAL_DIR, "eval_queries.jsonl")
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def _ids(ranked):
    return [cid for cid, _ in ranked]


def _recall_mrr(ranked, relevant):
    if not relevant:
        return None, None
    top = ranked[:st.TOP_K]
    hit = len(set(top) & set(relevant))
    recall = hit / len(relevant)
    mrr = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            mrr = 1.0 / i
            break
    return recall, mrr


def _coverage_of(ids, query):
    if not ids or not (query or "").strip():
        return 0.0
    c = rag._chunk_by_id.get(ids[0])
    if not c:
        return 0.0
    text = f"{c.get('plant', '')} {c.get('section', '')} {c.get('text', '')}"
    return router.coverage(query, text)


def _target_plant(case):
    if case.get("plant_hint"):
        return case["plant_hint"]
    if case.get("target_plant"):
        return case["target_plant"]
    rel = case.get("relevant") or []
    if rel:
        c = rag._chunk_by_id.get(rel[0])
        if c:
            return c.get("plant")
    return None


# ---------- 各配置的逐题检索 ----------

def _search_unfiltered(route, q, topk):
    if route == router.ROUTE_BM25:
        ranked = rag._ensure_bm25().search(q, st.FUSE_TOP_N)[:topk]
        return ranked, None
    if route == router.ROUTE_DENSE:
        ranked = rag._ensure_dense().search(q, st.FUSE_TOP_N)[:topk]
        return ranked, (ranked[0][1] if ranked else None)
    bm = rag._ensure_bm25().search(q, st.FUSE_TOP_N)
    de = rag._ensure_dense().search(q, st.FUSE_TOP_N)
    fused = retrievers.rrf_fuse([bm, de], st.RRF_K, topk)
    return fused, (de[0][1] if de else None)


def _filtered(route, case):
    q = case["query"].strip() or (case.get("plant_hint") or "")
    plant = case.get("plant_hint") or rag._detect_plant(q)
    t0 = time.perf_counter()
    ranked, top1_cos = rag._search(route, q, st.TOP_K, plant)
    lat = (time.perf_counter() - t0) * 1000
    return _ids(ranked), route, False, None, top1_cos, lat


def _adaptive_variant(case, use_filter, use_retry):
    hint = case.get("plant_hint") or None
    q = (case["query"] or "").strip() or (hint or "")
    names, primary = rag._ensure_names()
    route = router.decide(q, hint, names, primary)
    plant = (hint or rag._detect_plant(q)) if use_filter else None

    t0 = time.perf_counter()
    if use_filter:
        ranked, top1_cos = rag._search(route, q, st.TOP_K, plant)
    else:
        ranked, top1_cos = _search_unfiltered(route, q, st.TOP_K)

    retried = False
    if use_retry:
        new_route = router.should_retry(route, rag._top1_text(ranked), q, top1_cos)
        if new_route:
            route = new_route
            retried = True
            if use_filter:
                ranked, top1_cos = rag._search(route, q, st.TOP_K, plant)
            else:
                ranked, top1_cos = _search_unfiltered(route, q, st.TOP_K)
    lat = (time.perf_counter() - t0) * 1000
    cov = router.coverage(q, rag._top1_text(ranked)) if q else 1.0
    return _ids(ranked), route, retried, cov, top1_cos, lat


def _rank(key, case):
    query = case["query"]
    hint = case.get("plant_hint") or None

    if key == "P2:naive":
        t0 = time.perf_counter()
        ids = [cid for cid, _ in rag.naive_rank(query, st.TOP_K)]
        return ids, "naive", False, None, None, (time.perf_counter() - t0) * 1000

    if key == "P2:bm25":
        t0 = time.perf_counter()
        ids = [cid for cid, _ in
               rag._ensure_bm25().search(query, st.FUSE_TOP_N)[:st.TOP_K]]
        return ids, "bm25", False, None, None, (time.perf_counter() - t0) * 1000

    if key == "P2:dense":
        t0 = time.perf_counter()
        ranked = rag._ensure_dense().search(query, st.FUSE_TOP_N)[:st.TOP_K]
        lat = (time.perf_counter() - t0) * 1000
        return _ids(ranked), "dense", False, None, \
            (ranked[0][1] if ranked else None), lat

    if key == "P2:hybrid":
        t0 = time.perf_counter()
        bm = rag._ensure_bm25().search(query, st.FUSE_TOP_N)
        de = rag._ensure_dense().search(query, st.FUSE_TOP_N)
        fused = retrievers.rrf_fuse([bm, de], st.RRF_K, st.TOP_K)
        lat = (time.perf_counter() - t0) * 1000
        return _ids(fused), "hybrid", False, None, \
            (de[0][1] if de else None), lat

    if key == "P2:adaptive":
        dbg = rag.retrieve_debug(query, st.TOP_K, hint)
        return ([r["id"] for r in dbg["results"]], dbg["route"],
                dbg["retried"], dbg["coverage"], dbg["top1_cos"],
                dbg["latency_ms"])

    if key == "P1:bm25+filter":
        return _filtered(router.ROUTE_BM25, case)
    if key == "P1:dense+filter":
        return _filtered(router.ROUTE_DENSE, case)
    if key == "P1:hybrid+filter":
        return _filtered(router.ROUTE_HYBRID, case)

    if key == "P1:adaptive":
        dbg = rag.retrieve_debug(query, st.TOP_K, hint)
        return ([r["id"] for r in dbg["results"]], dbg["route"],
                dbg["retried"], dbg["coverage"], dbg["top1_cos"],
                dbg["latency_ms"])
    if key == "P1:adaptive−filter":
        return _adaptive_variant(case, use_filter=False, use_retry=True)
    if key == "P1:adaptive−retry":
        return _adaptive_variant(case, use_filter=True, use_retry=False)

    raise ValueError(f"未知配置:{key}")


def _warmup(key):
    if "naive" in key:
        rag.naive_rank("绿萝", 1)
        return
    if "adaptive" in key:
        rag.retrieve_debug("绿萝", 1, None)
        return
    if "bm25" in key or "hybrid" in key:
        rag._ensure_bm25().search("绿萝", 1)
    if "dense" in key or "hybrid" in key:
        rag._ensure_dense().search("绿萝", 1)


def run_config(key, cases):
    """指标用一次结果;延迟按 _REPEATS 次执行取中位数(CR-012)。"""
    _warmup(key)
    rows = []
    for case in cases:
        latencies = []
        result = None
        for _ in range(_REPEATS):
            t0 = time.perf_counter()
            result = _rank(key, case)
            latencies.append((time.perf_counter() - t0) * 1000)
        ids, route, retried, cov, cos, _inner = result
        relevant = case.get("relevant", [])
        recall, mrr = _recall_mrr(ids, relevant)
        top1 = None if not relevant else (
            1.0 if ids and ids[0] in relevant else 0.0)
        rows.append({
            "case_id": case["id"], "type": case.get("type"),
            "route": route, "retried": bool(retried),
            "top1_hit": top1, "recall5": recall, "mrr": mrr,
            "latency_ms": statistics.median(latencies),
            "top1_id": ids[0] if ids else None,
            "top1_score": cos, "gate_cov": _coverage_of(ids, case["query"]),
            "coverage": cov, "top1_cos": cos, "ranked_ids": ids,
        })
    return rows


# ---------- 统计 ----------

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else 0.0


def _p95(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def _wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def _bootstrap_mean_ci(vals, n=_BOOTSTRAP_N, seed=_SEED):
    arr = np.asarray([v for v in vals if v is not None], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def summarize(rows):
    out = {}
    for key, rs in rows.items():
        top1_vals = [r["top1_hit"] for r in rs if r["top1_hit"] is not None]
        if top1_vals:
            t1 = _wilson(int(sum(top1_vals)), len(top1_vals))
        else:
            t1 = (0.0, 0.0, 0.0)
        rec = _bootstrap_mean_ci([r["recall5"] for r in rs])
        mrr = _bootstrap_mean_ci([r["mrr"] for r in rs])
        by_type = {}
        for r in rs:
            if r["recall5"] is None:
                continue
            d = by_type.setdefault(
                r["type"], {"top1": [], "recall5": [], "mrr": []})
            d["top1"].append(r["top1_hit"])
            d["recall5"].append(r["recall5"])
            d["mrr"].append(r["mrr"])
        out[key] = {
            "top1": t1, "recall5": rec, "mrr": mrr,
            "lat_med": _median([r["latency_ms"] for r in rs]),
            "lat_p95": _p95([r["latency_ms"] for r in rs]),
            "lat_mean": _mean([r["latency_ms"] for r in rs]),
            "by_type": {t: {m: _mean(v[m]) for m in ("top1", "recall5", "mrr")}
                        for t, v in by_type.items()},
        }
    return out


def _row_map(rs):
    return {r["case_id"]: r for r in rs}


def _compare(rows, a_key, b_key, metric):
    a = _row_map(rows[a_key])
    b = _row_map(rows[b_key])
    diffs, wins, losses = [], 0, 0
    for cid, r in a.items():
        av = r.get(metric)
        bv = b.get(cid, {}).get(metric)
        if av is None or bv is None:
            continue
        d = av - bv
        diffs.append(d)
        if d > 1e-9:
            wins += 1
        elif d < -1e-9:
            losses += 1
    if not diffs:
        return {"mean": 0.0, "ci": (0.0, 0.0), "win": 0, "tie": 0,
                "loss": 0, "n": 0, "p": None}
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(_SEED)
    idx = rng.integers(0, arr.size, size=(_BOOTSTRAP_N, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    ties = len(diffs) - wins - losses
    p = _mcnemar_exact(wins, losses) if metric == "top1_hit" else None
    return {"mean": float(arr.mean()), "ci": (float(lo), float(hi)),
            "win": wins, "tie": ties, "loss": losses, "n": len(diffs), "p": p}


def _gate_hit(r):
    return r["gate_cov"] >= st.COVERAGE_THRESHOLD or (
        r["top1_cos"] is not None and r["top1_cos"] >= st.ANSWER_COS_THRESHOLD)


def _gate_stats(rs):
    ans = [r for r in rs if r["type"] != "unanswerable"]
    una = [r for r in rs if r["type"] == "unanswerable"]
    fa = sum(1 for r in una if _gate_hit(r))
    fr = sum(1 for r in ans if not _gate_hit(r))
    return _wilson(fa, len(una)), _wilson(fr, len(ans)), len(una), len(ans)


# ---------- 表格 ----------

def _fmt3(mean, lo, hi):
    return f"{mean:.3f} [{lo:.3f},{hi:.3f}]"


def _fmt_pct(x):
    return f"{x:.3f}" if x is not None else "-"


def _config_table(lines, keys, summ):
    lines += ["| 配置 | Top-1 | Recall@5 | MRR | 中位延迟(ms) | P95(ms) | 均值(ms) |",
              "|---|---|---|---|---|---|---|"]
    for key in keys:
        if key not in summ:
            continue
        s = summ[key]
        lines.append(
            f"| {display_name(key)} | {_fmt3(*s['top1'])} "
            f"| {_fmt3(*s['recall5'])} | {_fmt3(*s['mrr'])} "
            f"| {s['lat_med']:.2f} | {s['lat_p95']:.2f} | {s['lat_mean']:.2f} |")


def _by_type_table(lines, keys, summ):
    present = [k for k in keys if k in summ]
    lines += ["| 类型 | " + " | ".join(display_name(k) for k in present) + " |",
              "|---|" + "---|" * len(present)]
    for t in ("entity", "symptom", "alias", "multi"):
        cells = []
        for k in present:
            d = summ[k]["by_type"].get(t)
            cells.append("-" if not d else
                         f"{d['top1']:.3f}/{d['recall5']:.3f}/{d['mrr']:.3f}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("> 单元格格式:Top-1 / Recall@5 / MRR。")


def _route_dist(lines, rows, keys):
    found = False
    for key in keys:
        if key not in rows:
            continue
        found = True
        dist, retried = {}, 0
        for r in rows[key]:
            dist[r["route"]] = dist.get(r["route"], 0) + 1
            retried += 1 if r["retried"] else 0
        lines.append(f"- `{display_name(key)}`:路线 {json.dumps(dist, ensure_ascii=False)},"
                     f"回退 {retried} 次")
    return found


def _compare_table(lines, rows, pairs):
    lines += ["| 对比 | 指标 | 均值差 | 95% CI(配对 bootstrap) | 胜 | 平 | 负 | 检验 |",
              "|---|---|---|---|---|---|---|---|"]
    labels = {"top1_hit": "Top-1", "recall5": "Recall@5", "mrr": "MRR"}
    for a_key, b_key in pairs:
        if a_key not in rows or b_key not in rows:
            continue
        for metric in ("top1_hit", "recall5", "mrr"):
            c = _compare(rows, a_key, b_key, metric)
            if c["n"] == 0:
                continue
            test = (f"McNemar p={c['p']:.3f}" if c["p"] is not None
                    else "paired bootstrap")
            lines.append(
                f"| {display_name(a_key)} vs {display_name(b_key)} "
                f"| {labels[metric]} | {c['mean']:+.3f} "
                f"| [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] "
                f"| {c['win']} | {c['tie']} | {c['loss']} | {test} |")


def _gate_table(lines, rows):
    lines += ["| 配置 | 库外误接收率 | 正常题误拒率 |",
              "|---|---|---|"]
    for key, rs in rows.items():
        fa, fr, n_una, n_ans = _gate_stats(rs)
        lines.append(
            f"| {display_name(key)}({key.split(':')[0]}) "
            f"| {_fmt3(*fa)}(n={n_una}) | {_fmt3(*fr)}(n={n_ans}) |")


def write_tables(summ, rows, cases, out_path, err_path, suite):
    title = "检索对比表" if suite == "main" else "盲症状对比表"
    lines = [f"# {title}", "",
             f"生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"题量:{len(cases)}", "",
             "## 协议说明", "",
             "- P1(主表):纯检索组统一实体过滤;adaptive 为产品路径。",
             "- P2(辅表):naive/bm25/dense/hybrid 不做实体过滤;"
             "adaptive 为产品路径(含自动实体约束)。",
             f"- 比例指标附 Wilson 95% 区间;Recall/MRR 附 bootstrap 95% 区间"
             f"(n={_BOOTSTRAP_N}, seed={_SEED})。", ""]

    p1_present = any(k in summ for k in P1_CONFIGS)
    if p1_present:
        lines += ["## P1 公平消融(主表)", ""]
        _config_table(lines, P1_CONFIGS, summ)
        lines += ["", "### P1 分类型(单元格:Top-1/Recall@5/MRR)", ""]
        _by_type_table(lines, P1_CONFIGS, summ)
        if "P1:adaptive" in rows:
            lines += ["", "### P1 路由效用(逐题配对比较)", ""]
            _compare_table(lines, rows, [
                ("P1:adaptive", "P1:dense+filter"),
                ("P1:adaptive", "P1:hybrid+filter"),
            ])
        lines += ["", "### P1 路由分布", ""]
        if not _route_dist(lines, rows, [k for k in P1_CONFIGS if _is_adaptive(k)]):
            lines.append("- 未运行 adaptive 系列配置")

    p2_present = any(k in summ for k in P2_CONFIGS)
    if p2_present:
        lines += ["", "## P2 原始五组(辅表)", ""]
        _config_table(lines, P2_CONFIGS, summ)
        lines += ["", "### P2 分类型(单元格:Top-1/Recall@5/MRR)", ""]
        _by_type_table(lines, P2_CONFIGS, summ)
        lines += ["", "### P2 路由分布", ""]
        _route_dist(lines, rows, ["P2:adaptive"])

    if suite == "blind":
        baselines = [k for k in ("P2:bm25", "P2:dense", "P2:hybrid") if k in rows]
        if "P2:adaptive" in rows and baselines:
            best = max(baselines, key=lambda k: summ[k]["recall5"][0])
            lines += ["", "## 盲题对比(CR-014 口径)", "",
                      f"- 最佳无过滤基线(按 Recall@5):`{display_name(best)}`", ""]
            _compare_table(lines, rows, [("P2:adaptive", k) for k in baselines])

    lines += ["", "## 库外问题(unanswerable)", "",
              "命中判定:Top-1 覆盖度 ≥ 0.5 或余弦 ≥ 0.55。", ""]
    _gate_table(lines, rows)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _write_errors(rows, cases, err_path)


def _write_errors(rows, cases, err_path):
    err_key = "P1:adaptive" if "P1:adaptive" in rows else "P2:adaptive"
    rs = rows.get(err_key, [])
    case_map = {c["id"]: c for c in cases}
    items = []
    for r in rs:
        c = case_map.get(r["case_id"], {})
        rel = c.get("relevant", [])
        if c.get("type") == "unanswerable":
            if _gate_hit(r):
                items.append((r, "false_accept"))
            continue
        if not rel:
            continue
        top1_ok = r["top1_id"] in rel if r["top1_id"] else False
        if top1_ok and r["recall5"] >= 1.0:
            continue
        target = _target_plant(c)
        top1_plant = None
        if r["top1_id"]:
            cc = rag._chunk_by_id.get(r["top1_id"])
            top1_plant = cc.get("plant") if cc else None
        if target and top1_plant and top1_plant != target:
            cat = "wrong_plant"
        elif target and top1_plant == target and not top1_ok:
            cat = "right_plant_wrong_section"
        elif top1_ok and r["recall5"] < 1.0:
            cat = "missing_secondary"
        else:
            cat = "other"
        items.append((r, cat))

    el = [f"# 检索错误案例({display_name(err_key)},全量)", "",
          "分类:wrong_plant / right_plant_wrong_section / "
          "missing_secondary / false_accept / other", ""]
    for r, cat in items:
        c = case_map.get(r["case_id"], {})
        el += [f"## {r['case_id']} [{c.get('type')}] {cat}",
               f"- Query:{c.get('query')}",
               f"- Route:{r['route']} (retried={r['retried']})",
               f"- Top-1:{r['top1_id']} (hit={r['top1_hit']})",
               f"- Recall@5:{_fmt_pct(r['recall5'])} / MRR:{_fmt_pct(r['mrr'])}",
               f"- 相关:{json.dumps(c.get('relevant', []), ensure_ascii=False)}", ""]
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("\n".join(el) + "\n")


# ---------- CR-012 额外产物 ----------

def _write_latency(rows, out_path):
    lines = ["# 延迟对比", "",
             f"- dtype:{_DTYPE_NAME};线程数:{_THREADS};"
             f"warm-up:每组一次;重复:{_REPEATS} 次取中位",
             "- 同机同 dtype 才可比。", "",
             "| 配置 | 中位(ms) | P95(ms) | 均值(ms) |", "|---|---|---|---|"]
    stats = {}
    for key, rs in rows.items():
        vals = [r["latency_ms"] for r in rs]
        stats[key] = (statistics.median(vals), _p95(vals), _mean(vals))
        lines.append(f"| {_label(key)} "
                     f"| {stats[key][0]:.2f} | {stats[key][1]:.2f} "
                     f"| {stats[key][2]:.2f} |")

    keys = [k for k in rows if _is_adaptive(k)]
    if keys:
        lines += ["", "## 分路线延迟(adaptive 系列)", "",
                  "| 配置 | 路线 | 题数 | 中位(ms) | P95(ms) | 均值(ms) |",
                  "|---|---|---|---|---|---|"]
        for key in keys:
            by_route = {}
            for r in rows[key]:
                by_route.setdefault(r["route"], []).append(r["latency_ms"])
            for route, vals in by_route.items():
                lines.append(f"| {_label(key)} | {route} | {len(vals)} "
                             f"| {statistics.median(vals):.2f} | {_p95(vals):.2f} "
                             f"| {_mean(vals):.2f} |")

    for base in ("P1:hybrid+filter", "P1:dense+filter"):
        if base not in stats:
            continue
        lines += ["", f"## 相对 `{display_name(base)}` 的节省", "",
                  "| 配置 | 中位节省(ms) | 中位节省(%) | 均值节省(ms) | 均值节省(%) |",
                  "|---|---|---|---|---|"]
        for key, (med, _p, mean) in stats.items():
            if key == base:
                continue
            b_med, _b_p, b_mean = stats[base]
            pct_med = (b_med - med) / b_med * 100 if b_med else 0.0
            pct_mean = (b_mean - mean) / b_mean * 100 if b_mean else 0.0
            lines.append(f"| {_label(key)} | {b_med - med:+.2f} "
                         f"| {pct_med:+.1f}% | {b_mean - mean:+.2f} "
                         f"| {pct_mean:+.1f}% |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_unanswerable(rows, out_path):
    lines = ["# 库外问题", "",
             "命中判定:Top-1 覆盖度 ≥ 0.5 或余弦 ≥ 0.55。", "",
             "| 配置 | 库外误接收率 | 正常题误拒率 |", "|---|---|---|"]
    for key, rs in rows.items():
        fa, fr, n_una, n_ans = _gate_stats(rs)
        lines.append(f"| {display_name(key)}({key.split(':')[0]}) "
                     f"| {_fmt3(*fa)}(n={n_una}) | {_fmt3(*fr)}(n={n_ans}) |")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _prep_sensitivity(cases):
    names, primary = rag._ensure_names()
    pre = []
    for case in cases:
        q = case["query"]
        hint = case.get("plant_hint") or None
        plant = hint or rag._detect_plant(q)
        route = router.decide(q, hint, names, primary)
        t0 = time.perf_counter()
        ranked, cos = rag._search(route, q, st.TOP_K, plant)
        lat_i = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        h_ranked, _h_cos = rag._search(router.ROUTE_HYBRID, q, st.TOP_K, plant)
        lat_h = (time.perf_counter() - t0) * 1000
        cov = router.coverage(q, rag._top1_text(ranked)) if q else 1.0
        pre.append({"route": route, "initial": _ids(ranked),
                    "hybrid": _ids(h_ranked), "cov": cov, "cos": cos,
                    "lat_i": lat_i, "lat_h": lat_h,
                    "relevant": case.get("relevant", [])})
    return pre


def _simulate(pre, cov_thr=None, cos_thr=None):
    cov_thr = st.COVERAGE_THRESHOLD if cov_thr is None else cov_thr
    cos_thr = st.DENSE_COS_THRESHOLD if cos_thr is None else cos_thr
    t1, rc, mr, lat, retried = [], [], [], [], 0
    for p in pre:
        use_hybrid = False
        if p["route"] == router.ROUTE_BM25 and p["cov"] < cov_thr:
            use_hybrid = True
        if p["route"] == router.ROUTE_DENSE and p["cos"] is not None \
                and p["cos"] < cos_thr:
            use_hybrid = True
        ids = p["hybrid"] if use_hybrid else p["initial"]
        retried += 1 if use_hybrid else 0
        rel = p["relevant"]
        recall, mrr = _recall_mrr(ids, rel)
        if rel:
            t1.append(1.0 if ids and ids[0] in rel else 0.0)
            rc.append(recall)
            mr.append(mrr)
        lat.append(p["lat_i"] + (p["lat_h"] if use_hybrid else 0))
    retry_rate = retried / len(pre) if pre else 0.0
    return (_mean(t1), _mean(rc), _mean(mr),
            statistics.median(lat) if lat else 0.0, retry_rate)


def _write_sensitivity(cases, out_path):
    pre = _prep_sensitivity(cases)
    lines = ["# 阈值敏感性", "",
             "默认值:覆盖率 0.5;Dense 余弦 0.4;库外闸门余弦 0.55。", ""]

    lines += ["## BM25 覆盖率阈值(回退到 hybrid)", "",
              "| 覆盖率阈值 | Top-1 | Recall@5 | MRR | 回退率 | 中位延迟(ms) |",
              "|---|---|---|---|---|---|"]
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        t1, rc, mr, lat, rr = _simulate(pre, cov_thr=thr)
        lines.append(f"| {thr:.1f} | {t1:.3f} | {rc:.3f} | {mr:.3f} "
                     f"| {rr:.3f} | {lat:.2f} |")

    lines += ["", "## Dense 余弦阈值(回退到 hybrid)", "",
              "| 余弦阈值 | Top-1 | Recall@5 | MRR | 回退率 | 中位延迟(ms) |",
              "|---|---|---|---|---|---|"]
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        t1, rc, mr, lat, rr = _simulate(pre, cos_thr=thr)
        lines.append(f"| {thr:.1f} | {t1:.3f} | {rc:.3f} | {mr:.3f} "
                     f"| {rr:.3f} | {lat:.2f} |")

    key = "P1:adaptive" if "P1:adaptive" in _LAST_ROWS else "P2:adaptive"
    rs = _LAST_ROWS.get(key, [])
    lines += ["", "## 库外闸门余弦阈值", "",
              "| 闸门阈值 | 库外误接收率 | 正常题误拒率 |", "|---|---|---|"]
    for thr in (0.45, 0.55, 0.65):
        fa = fr = 0
        n_una = n_ans = 0
        for r in rs:
            hit = r["gate_cov"] >= st.COVERAGE_THRESHOLD or (
                r["top1_cos"] is not None and r["top1_cos"] >= thr)
            if r["type"] == "unanswerable":
                n_una += 1
                fa += 1 if hit else 0
            else:
                n_ans += 1
                fr += 0 if hit else 1
        fa_ci = _wilson(fa, n_una)
        fr_ci = _wilson(fr, n_ans)
        lines.append(f"| {thr:.2f} | {_fmt3(*fa_ci)}(n={n_una}) "
                     f"| {_fmt3(*fr_ci)}(n={n_ans}) |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


_LAST_ROWS = {}


def write_db(db_path, rows, summ):
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
CREATE TABLE IF NOT EXISTS eval_run(
  run_id INTEGER PRIMARY KEY, config TEXT, created_at TEXT,
  recall5 REAL, mrr REAL, latency_med_ms REAL, latency_p95_ms REAL);
CREATE TABLE IF NOT EXISTS case_result(
  id INTEGER PRIMARY KEY, run_id INTEGER, case_id TEXT, type TEXT,
  route TEXT, retried INTEGER, recall5 REAL, mrr REAL,
  latency_ms REAL, top1_id TEXT, top1_score REAL);
CREATE TABLE IF NOT EXISTS route_trace(
  id INTEGER PRIMARY KEY, run_id INTEGER, case_id TEXT,
  route TEXT, retried INTEGER, coverage REAL, top1_cos REAL);
""")
    cols = [r[1] for r in cur.execute("PRAGMA table_info(case_result)")]
    if "top1_hit" not in cols:
        cur.execute("ALTER TABLE case_result ADD COLUMN top1_hit REAL")

    for key in ALL_CONFIGS:
        if key not in rows:
            continue
        s = summ[key]
        cur.execute("SELECT COALESCE(MAX(run_id),0)+1 FROM eval_run")
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO eval_run VALUES (?,?,?,?,?,?,?)",
                    (run_id, key, datetime.now().isoformat(timespec="seconds"),
                     s["recall5"][0], s["mrr"][0], s["lat_med"], s["lat_p95"]))
        for r in rows[key]:
            cur.execute("INSERT INTO case_result(run_id,case_id,type,route,retried,"
                        "recall5,mrr,latency_ms,top1_id,top1_score,top1_hit) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (run_id, r["case_id"], r["type"], r["route"],
                         1 if r["retried"] else 0, r["recall5"], r["mrr"],
                         r["latency_ms"], r["top1_id"], r["top1_score"],
                         r["top1_hit"]))
            if _is_adaptive(key):
                cur.execute("INSERT INTO route_trace(run_id,case_id,route,retried,"
                            "coverage,top1_cos) VALUES (?,?,?,?,?,?)",
                            (run_id, r["case_id"], r["route"],
                             1 if r["retried"] else 0, r["coverage"], r["top1_cos"]))
    conn.commit()
    conn.close()


def _resolve_configs(tokens):
    keys = []
    for tok in tokens:
        tok = _ASCII_ALIASES.get(tok, tok)
        if tok not in ALL_CONFIGS:
            alt = tok.replace("-", "−")
            if alt in ALL_CONFIGS:
                tok = alt
        if tok in ALL_CONFIGS:
            keys.append(tok)
        elif tok in _SHORTHAND:
            keys.append(_SHORTHAND[tok])
        else:
            raise SystemExit(f"未知配置:{tok};可选:{ALL_CONFIGS} 或简写 {list(_SHORTHAND)}")
    return keys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(ALL_CONFIGS))
    ap.add_argument("--cases", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("指标", "retrieval", "检索对比表.md"))
    ap.add_argument("--err", default=os.path.join("指标", "retrieval", "检索错误案例.md"))
    ap.add_argument("--db", default=st.EVAL_DB)
    args = ap.parse_args()

    configs = _resolve_configs([c.strip() for c in args.configs.split(",") if c.strip()])
    cases = load_cases(args.cases, args.limit or None)
    suite = "blind" if "blind" in os.path.basename(args.cases or "") else "main"
    print(f"[eval] 题量 {len(cases)},配置 {configs},协议 {suite}")

    _setup_runtime()
    rows = {}
    for key in configs:
        print(f"[eval] 运行 {key} …")
        rows[key] = run_config(key, cases)

    summ = summarize(rows)
    _LAST_ROWS.clear()
    _LAST_ROWS.update(rows)

    for key in configs:
        s = summ[key]
        print(f"[eval] {key:22s} Top-1={s['top1'][0]:.3f} "
              f"Recall@5={s['recall5'][0]:.3f} MRR={s['mrr'][0]:.3f} "
              f"中位={s['lat_med']:.2f}ms P95={s['lat_p95']:.2f}ms")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_tables(summ, rows, cases, args.out, args.err, suite)
    write_db(args.db, rows, summ)

    if suite == "main":
        base = os.path.dirname(os.path.abspath(args.out))
        _write_latency(rows, os.path.join(base, "延迟对比.md"))
        _write_unanswerable(rows, os.path.join(base, "库外问题表.md"))
        _write_sensitivity(cases, os.path.join(base, "阈值敏感性.md"))

    print(f"[eval] 已写入 {args.out} / {args.err} / {args.db}")


if __name__ == "__main__":
    main()
