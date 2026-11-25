"""检索评测:P1 公平协议(统一实体过滤)与 P2 原始协议。

用法(需在项目根目录运行):
    python -m retrieval.eval
    python -m retrieval.eval --configs P2:bm25,P2:dense
    python -m retrieval.eval --configs bm25,dense       # 简写映射到 P2
    python -m retrieval.eval --configs P1:adaptive-filter  # ASCII "-" 也可
    python -m retrieval.eval --limit 50

P1(主表):bm25/dense/hybrid 统一使用“case.plant_hint,缺失时从 query 检测”
的实体约束;adaptive 为产品路径;另设 adaptive−filter / adaptive−retry 消融。
P2(辅表):原五组行为不变 —— naive/bm25/dense/hybrid 无过滤,adaptive 为
产品路径(含自动实体约束)。
"""

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime

from . import rag, retrievers, router
from . import settings as st

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

# 简写:旧命令仍是 P2 语义
_SHORTHAND = {"naive": "P2:naive", "bm25": "P2:bm25", "dense": "P2:dense",
              "hybrid": "P2:hybrid", "adaptive": "P2:adaptive"}

# 配置名里的 "−" 是 U+2212;兼容用户输入的 ASCII 连字符
_ASCII_ALIASES = {
    "P1:adaptive-filter": "P1:adaptive−filter",
    "P1:adaptive-retry": "P1:adaptive−retry",
    "adaptive-filter": "P1:adaptive−filter",
    "adaptive-retry": "P1:adaptive−retry",
}


def display_name(key):
    return key.split(":", 1)[1]


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


# ---------- 各配置的逐题检索 ----------

def _search_unfiltered(route, q, topk):
    """路由已定,不做实体过滤。返回 (ranked, top1_cos)。"""
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
    """P1 纯检索组:统一实体过滤。"""
    q = case["query"].strip() or (case.get("plant_hint") or "")
    plant = case.get("plant_hint") or rag._detect_plant(q)
    t0 = time.perf_counter()
    ranked, top1_cos = rag._search(route, q, st.TOP_K, plant)
    lat = (time.perf_counter() - t0) * 1000
    return _ids(ranked), route, False, None, top1_cos, lat


def _adaptive_variant(case, use_filter, use_retry):
    """adaptive 消融:保留规则路由,可关闭实体过滤或质量回退。"""
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
    """返回 (ranked_ids, route, retried, coverage, top1_cos, latency_ms)。"""
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


# ---------- 评测执行与汇总 ----------

def _warmup(key):
    """索引/模型构建不计入计时(只构建该配置需要的部分)。"""
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
    _warmup(key)
    rows = []
    for case in cases:
        ids, route, retried, cov, cos, lat = _rank(key, case)
        relevant = case.get("relevant", [])
        recall, mrr = _recall_mrr(ids, relevant)
        top1 = None if not relevant else (1.0 if ids and ids[0] in relevant else 0.0)
        rows.append({
            "case_id": case["id"], "type": case.get("type"),
            "route": route, "retried": bool(retried),
            "top1_hit": top1, "recall5": recall, "mrr": mrr,
            "latency_ms": lat, "top1_id": ids[0] if ids else None,
            "top1_score": cos, "coverage": cov, "top1_cos": cos,
            "ranked_ids": ids,
        })
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else 0.0


def _p95(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def summarize(rows):
    """每个配置的 Top-1 / Recall@5 / MRR / 延迟 / 分类型 Recall。"""
    out = {}
    for key, rs in rows.items():
        by_type = {}
        for r in rs:
            if r["recall5"] is None:
                continue
            by_type.setdefault(r["type"], []).append(r["recall5"])
        out[key] = {
            "top1": _mean([r["top1_hit"] for r in rs]),
            "recall5": _mean([r["recall5"] for r in rs]),
            "mrr": _mean([r["mrr"] for r in rs]),
            "lat_med": _median([r["latency_ms"] for r in rs]),
            "lat_p95": _p95([r["latency_ms"] for r in rs]),
            "by_type": {t: _mean(v) for t, v in by_type.items()},
        }
    return out


def _row_map(rs):
    return {r["case_id"]: r for r in rs}


def _wtl(rows, a_key, b_key, metric):
    """逐题差值 a − b;平局阈值 1e-9。返回 (mean, median, win, tie, loss, n)。"""
    a = _row_map(rows[a_key])
    b = _row_map(rows[b_key])
    diffs = []
    for cid in a:
        av, bv = a[cid].get(metric), b.get(cid, {}).get(metric)
        if av is None or bv is None:
            continue
        diffs.append(av - bv)
    if not diffs:
        return 0.0, 0.0, 0, 0, 0, 0
    win = sum(1 for d in diffs if d > 1e-9)
    loss = sum(1 for d in diffs if d < -1e-9)
    tie = len(diffs) - win - loss
    return (sum(diffs) / len(diffs), statistics.median(diffs),
            win, tie, loss, len(diffs))


def _route_utility_acc(rows, summ):
    """辅助指标(旧口径):逐题 adaptive ≥ 同题最佳 P1 基线的比例。"""
    base_keys = [k for k in ("P1:dense+filter", "P1:hybrid+filter",
                             "P1:bm25+filter") if k in rows]
    for a_key in ("P1:adaptive", "P2:adaptive"):
        if a_key not in rows or not base_keys:
            continue
        a = _row_map(rows[a_key])
        maps = [_row_map(rows[k]) for k in base_keys]
        ok = []
        for cid, r in a.items():
            if r["recall5"] is None:
                continue
            vals = [m[cid]["recall5"] for m in maps
                    if cid in m and m[cid]["recall5"] is not None]
            if not vals:
                continue
            ok.append(1.0 if r["recall5"] >= max(vals) - 1e-9 else 0.0)
        summary = summ.setdefault(a_key, {})
        summary["route_util_acc"] = _mean(ok)


# ---------- 产物 ----------

def _fmt_pct(x):
    return f"{x:.3f}" if x is not None else "-"


def _config_table(lines, keys, summ):
    lines += ["| 配置 | Top-1 | Recall@5 | MRR | 延迟中位数(ms) | P95(ms) |",
              "|---|---|---|---|---|---|"]
    for key in keys:
        if key not in summ:
            continue
        s = summ[key]
        lines.append(f"| {display_name(key)} | {_fmt_pct(s['top1'])} "
                     f"| {_fmt_pct(s['recall5'])} | {_fmt_pct(s['mrr'])} "
                     f"| {s['lat_med']:.2f} | {s['lat_p95']:.2f} |")


def _by_type_table(lines, keys, summ):
    present = [k for k in keys if k in summ]
    lines += ["| 类型 | " + " | ".join(display_name(k) for k in present) + " |",
              "|---|" + "---|" * len(present)]
    for t in ("entity", "symptom", "alias", "multi"):
        cells = [f"{summ[k]['by_type'].get(t, 0.0):.3f}" for k in present]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")


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


def write_tables(summ, rows, cases, out_path, err_path):
    lines = ["# 检索对比表", "",
             f"生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"题量:{len(cases)}(entity/symptom/alias/multi/unanswerable)", "",
             "## 协议说明", "",
             "- P1(主表):所有纯检索组统一实体过滤(case.plant_hint,缺失时从 query 检测);"
             "adaptive 为产品路径,adaptive−filter/adaptive−retry 为消融。",
             "- P2(辅表):naive/bm25/dense/hybrid 不做实体过滤;"
             "adaptive 为产品路径(含自动实体约束),数字与改造前一致。", ""]

    lines += ["## P1 公平消融(主表)", ""]
    _config_table(lines, P1_CONFIGS, summ)
    lines += ["", "### P1 分类型 Recall@5", ""]
    _by_type_table(lines, P1_CONFIGS, summ)

    if "P1:adaptive" in rows:
        lines += ["", "### P1 路由效用(逐题差值,平局阈值 1e-9)", "",
                  "| 对比 | 指标 | 均值差 | 中位差 | 胜 | 平 | 负 | 有效题数 |",
                  "|---|---|---|---|---|---|---|---|"]
        for base in ("P1:dense+filter", "P1:hybrid+filter"):
            if base not in rows:
                continue
            for metric, label in (("top1_hit", "Top-1"), ("recall5", "Recall@5"),
                                  ("mrr", "MRR")):
                mean, med, win, tie, loss, n = _wtl(
                    rows, "P1:adaptive", base, metric)
                lines.append(f"| adaptive vs {display_name(base)} | {label} "
                             f"| {mean:+.3f} | {med:+.3f} | {win} | {tie} | {loss} | {n} |")

    lines += ["", "### P1 路由分布", ""]
    if not _route_dist(lines, rows, P1_CONFIGS):
        lines.append("- 未运行 adaptive 系列配置")

    lines += ["", "## P2 原始五组(辅表)", ""]
    _config_table(lines, P2_CONFIGS, summ)
    lines += ["", "### P2 分类型 Recall@5", ""]
    _by_type_table(lines, P2_CONFIGS, summ)
    lines += ["", "### P2 路由分布", ""]
    _route_dist(lines, rows, ["P2:adaptive"])

    lines += ["", "## 库外问题(unanswerable)", "",
              "不进入 Top-1/Recall/MRR,仅看系统置信度是否更低。", "",
              "| 配置 | 平均 Top-1 覆盖度 | 平均 Top-1 余弦 |", "|---|---|---|"]
    for key in ALL_CONFIGS:
        if key not in rows:
            continue
        rs = [r for r in rows[key] if r["type"] == "unanswerable"]
        cov = [r["coverage"] for r in rs if r["coverage"] is not None]
        cos = [r["top1_cos"] for r in rs if r["top1_cos"] is not None]
        lines.append(f"| {display_name(key)}({key.split(':')[0]}) "
                     f"| {f'{_mean(cov):.3f}' if cov else '-'} "
                     f"| {f'{_mean(cos):.3f}' if cos else '-'} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 错误案例:沿用 adaptive 产品路径的 Recall@5 < 1
    err_key = "P1:adaptive" if "P1:adaptive" in rows else "P2:adaptive"
    errs = [r for r in rows.get(err_key, [])
            if r["recall5"] is not None and r["recall5"] < 1.0]
    case_map = {c["id"]: c for c in cases}
    errs.sort(key=lambda r: r["recall5"])
    el = [f"# 检索错误案例({display_name(err_key)} Recall@5 < 1, Top 20)", ""]
    for r in errs[:20]:
        c = case_map[r["case_id"]]
        el += [f"## {r['case_id']} [{c.get('type')}]",
               f"- Query:{c['query']}",
               f"- Route:{r['route']} (retried={r['retried']})",
               f"- Recall@5:{r['recall5']:.3f} / MRR:{r['mrr']:.3f}",
               f"- Top-1:{r['top1_id']}",
               f"- 相关:{json.dumps(c.get('relevant', []), ensure_ascii=False)}", ""]
    with open(err_path, "w", encoding="utf-8") as f:
        f.write("\n".join(el) + "\n")


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
    for key in ALL_CONFIGS:
        if key not in rows:
            continue
        s = summ[key]
        cur.execute("SELECT COALESCE(MAX(run_id),0)+1 FROM eval_run")
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO eval_run VALUES (?,?,?,?,?,?,?)",
                    (run_id, key, datetime.now().isoformat(timespec="seconds"),
                     s["recall5"], s["mrr"], s["lat_med"], s["lat_p95"]))
        for r in rows[key]:
            cur.execute("INSERT INTO case_result(run_id,case_id,type,route,retried,"
                        "recall5,mrr,latency_ms,top1_id,top1_score) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, r["case_id"], r["type"], r["route"],
                         1 if r["retried"] else 0, r["recall5"], r["mrr"],
                         r["latency_ms"], r["top1_id"], r["top1_score"]))
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
    print(f"[eval] 题量 {len(cases)},配置 {configs}")

    rows = {}
    for key in configs:
        print(f"[eval] 运行 {key} …")
        rows[key] = run_config(key, cases)

    summ = summarize(rows)
    _route_utility_acc(rows, summ)

    for key in configs:
        s = summ[key]
        print(f"[eval] {key:22s} Top-1={s['top1']:.3f} "
              f"Recall@5={s['recall5']:.3f} MRR={s['mrr']:.3f} "
              f"中位={s['lat_med']:.2f}ms P95={s['lat_p95']:.2f}ms")
        if "route_util_acc" in s:
            print(f"[eval] {key:22s} 路由效用达标率(辅助)={s['route_util_acc']:.3f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_tables(summ, rows, cases, args.out, args.err)
    write_db(args.db, rows, summ)
    print(f"[eval] 已写入 {args.out} / {args.err} / {args.db}")


if __name__ == "__main__":
    main()
