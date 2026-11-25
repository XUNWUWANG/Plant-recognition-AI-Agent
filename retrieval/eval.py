"""检索评测:naive / bm25 / dense / hybrid / adaptive 五组对照。

用法(需在项目根目录运行):
    python -m retrieval.eval
    python -m retrieval.eval --configs bm25,dense
    python -m retrieval.eval --out 指标/检索对比表.md --db 指标/eval.db
    python -m retrieval.eval --limit 50          # 仅用前 N 条(M0 判据)
"""

import argparse
import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime

import config as cfg

from . import rag, retrievers

CONFIGS = ["naive", "bm25", "dense", "hybrid", "adaptive"]


def load_cases(path=None, limit=None):
    path = path or os.path.join(cfg.EVAL_DIR, "eval_queries.jsonl")
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases[:limit] if limit else cases


def _rank(config, query, plant_hint):
    """返回 (ranked_ids, route, retried, coverage, top1_cos, latency_ms)。"""
    if config == "naive":
        t0 = time.perf_counter()
        ids = [cid for cid, _ in rag.naive_rank(query, cfg.TOP_K)]
        return ids, "naive", False, None, None, (time.perf_counter() - t0) * 1000

    if config == "bm25":
        t0 = time.perf_counter()
        ids = [cid for cid, _ in
               rag._ensure_bm25().search(query, cfg.FUSE_TOP_N)[:cfg.TOP_K]]
        return ids, "bm25", False, None, None, (time.perf_counter() - t0) * 1000

    if config == "dense":
        t0 = time.perf_counter()
        ranked = rag._ensure_dense().search(query, cfg.FUSE_TOP_N)[:cfg.TOP_K]
        lat = (time.perf_counter() - t0) * 1000
        return [cid for cid, _ in ranked], "dense", False, None, \
            (ranked[0][1] if ranked else None), lat

    if config == "hybrid":
        t0 = time.perf_counter()
        bm = rag._ensure_bm25().search(query, cfg.FUSE_TOP_N)
        de = rag._ensure_dense().search(query, cfg.FUSE_TOP_N)
        fused = retrievers.rrf_fuse([bm, de], cfg.RRF_K, cfg.TOP_K)
        lat = (time.perf_counter() - t0) * 1000
        return [cid for cid, _ in fused], "hybrid", False, None, \
            (de[0][1] if de else None), lat

    # adaptive
    dbg = rag.retrieve_debug(query, cfg.TOP_K, plant_hint)
    ids = [r["id"] for r in dbg["results"]]
    return (ids, dbg["route"], dbg["retried"], dbg["coverage"],
            dbg["top1_cos"], dbg["latency_ms"])


def _recall_mrr(ranked, relevant):
    if not relevant:
        return None, None
    top = ranked[:cfg.TOP_K]
    hit = len(set(top) & set(relevant))
    recall = hit / len(relevant)
    mrr = 0.0
    for i, cid in enumerate(ranked, start=1):
        if cid in relevant:
            mrr = 1.0 / i
            break
    return recall, mrr


def _warmup(name):
    """索引/模型构建不计入计时(只构建该配置需要的部分)。"""
    if name == "naive":
        rag.naive_rank("绿萝", 1)
    elif name == "bm25":
        rag._ensure_bm25().search("绿萝", 1)
    elif name == "dense":
        rag._ensure_dense().search("绿萝", 1)
    elif name == "hybrid":
        rag._ensure_bm25().search("绿萝", 1)
        rag._ensure_dense().search("绿萝", 1)
    else:
        rag.retrieve_debug("绿萝", 1, None)


def run_config(name, cases):
    _warmup(name)
    rows = []
    for case in cases:
        q = case["query"]
        hint = case.get("plant_hint") or None
        if name == "adaptive":
            ids, route, retried, cov, cos, lat = _rank(name, q, hint)
        else:
            ids, route, retried, cov, cos, lat = _rank(name, q, None)
        recall, mrr = _recall_mrr(ids, case.get("relevant", []))
        rows.append({
            "case_id": case["id"], "type": case.get("type"),
            "route": route, "retried": bool(retried),
            "recall5": recall, "mrr": mrr, "latency_ms": lat,
            "top1_id": ids[0] if ids else None,
            "top1_score": (cos if cos is not None else None),
            "coverage": cov, "top1_cos": cos, "ranked_ids": ids,
        })
    return rows


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def _p95(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return s[idx]


def summarize(rows, cases):
    """按配置汇总;rows 为 {config: [row]}。"""
    out = {}
    for name, rs in rows.items():
        recalls = [r["recall5"] for r in rs if r["recall5"] is not None]
        mrrs = [r["mrr"] for r in rs if r["mrr"] is not None]
        lat = [r["latency_ms"] for r in rs]
        by_type = {}
        for r in rs:
            if r["recall5"] is None:
                continue
            by_type.setdefault(r["type"], []).append(r["recall5"])
        out[name] = {
            "recall5": _mean(recalls), "mrr": _mean(mrrs),
            "lat_med": statistics.median(lat) if lat else 0.0,
            "lat_p95": _p95(lat),
            "by_type": {t: _mean(v) for t, v in by_type.items()},
        }
    # 路由准确率(以每条 case 的最优配置为上限)
    if "adaptive" in rows:
        best = {}
        for case in cases:
            vals = [summarize._case_recall(rows[c], case["id"])
                    for c in CONFIGS if c in rows]
            vals = [v for v in vals if v is not None]
            best[case["id"]] = max(vals) if vals else None
        ok = []
        for r in rows["adaptive"]:
            if r["recall5"] is None:
                continue
            b = best.get(r["case_id"])
            if b is None:
                continue
            ok.append(1.0 if r["recall5"] >= b - 1e-9 else 0.0)
        out["adaptive"]["route_acc"] = _mean(ok)
        out["adaptive"]["route_dist"] = {}
        out["adaptive"]["retried"] = 0
        for r in rows["adaptive"]:
            out["adaptive"]["route_dist"][r["route"]] = \
                out["adaptive"]["route_dist"].get(r["route"], 0) + 1
            out["adaptive"]["retried"] += 1 if r["retried"] else 0
    return out


def _case_recall(rs, case_id):
    for r in rs:
        if r["case_id"] == case_id:
            return r["recall5"]
    return None


summarize._case_recall = _case_recall


def _unanswerable_stats(rows):
    res = {}
    for name, rs in rows.items():
        cov = [r["coverage"] for r in rs
               if r["type"] == "unanswerable" and r["coverage"] is not None]
        cos = [r["top1_cos"] for r in rs
               if r["type"] == "unanswerable" and r["top1_cos"] is not None]
        res[name] = {"coverage": _mean(cov) if cov else None,
                     "cos": _mean(cos) if cos else None}
    return res


def _fmt_pct(x):
    return f"{x:.3f}" if x is not None else "-"


def write_tables(summ, rows, cases, out_path, err_path):
    lines = ["# 检索对比表", "",
             f"生成时间:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"题量:{len(cases)}(entity/symptom/alias/multi/unanswerable)", "",
             "## 总表", "",
             "| 配置 | Recall@5 | MRR | 延迟中位数(ms) | P95(ms) |",
             "|---|---|---|---|---|"]
    for name in CONFIGS:
        if name not in summ:
            continue
        s = summ[name]
        lines.append(f"| {name} | {_fmt_pct(s['recall5'])} | {_fmt_pct(s['mrr'])} "
                     f"| {s['lat_med']:.2f} | {s['lat_p95']:.2f} |")

    lines += ["", "## 分类型 Recall@5", ""]
    present = [c for c in CONFIGS if c in summ]
    lines += ["| 类型 | " + " | ".join(present) + " |",
              "|---|" + "---|" * len(present)]
    types = ["entity", "symptom", "alias", "multi"]
    for t in types:
        cells = [f"{summ[c]['by_type'].get(t, 0.0):.3f}" for c in present]
        lines.append(f"| {t} | " + " | ".join(cells) + " |")

    if "adaptive" in summ:
        a = summ["adaptive"]
        dist = a.get("route_dist", {})
        lines += ["", "## 路由分布(adaptive)", "",
                  f"- 路线次数:{json.dumps(dist, ensure_ascii=False)}",
                  f"- 回退次数:{a.get('retried', 0)}",
                  f"- 路由准确率:{_fmt_pct(a.get('route_acc'))}"]

    una = _unanswerable_stats(rows)
    lines += ["", "## 库外问题(unanswerable)", "",
              "不进入 Recall/MRR,仅看系统置信度是否更低。", "",
              "| 配置 | 平均 Top-1 覆盖度 | 平均 Top-1 余弦 |", "|---|---|---|"]
    for name in CONFIGS:
        if name not in una:
            continue
        u = una[name]
        cov = f"{u['coverage']:.3f}" if u["coverage"] is not None else "-"
        cos = f"{u['cos']:.3f}" if u["cos"] is not None else "-"
        lines.append(f"| {name} | {cov} | {cos} |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # 错误案例
    errs = [r for r in rows.get("adaptive", [])
            if r["recall5"] is not None and r["recall5"] < 1.0]
    case_map = {c["id"]: c for c in cases}
    errs.sort(key=lambda r: r["recall5"])
    el = ["# 检索错误案例(adaptive Recall@5 < 1, Top 20)", ""]
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
    for name in CONFIGS:
        if name not in rows:
            continue
        s = summ[name]
        cur.execute("SELECT COALESCE(MAX(run_id),0)+1 FROM eval_run")
        run_id = cur.fetchone()[0]
        cur.execute("INSERT INTO eval_run VALUES (?,?,?,?,?,?,?)",
                    (run_id, name, datetime.now().isoformat(timespec="seconds"),
                     s["recall5"], s["mrr"], s["lat_med"], s["lat_p95"]))
        for r in rows[name]:
            cur.execute("INSERT INTO case_result(run_id,case_id,type,route,retried,"
                        "recall5,mrr,latency_ms,top1_id,top1_score) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, r["case_id"], r["type"], r["route"],
                         1 if r["retried"] else 0, r["recall5"], r["mrr"],
                         r["latency_ms"], r["top1_id"], r["top1_score"]))
            if name == "adaptive":
                cur.execute("INSERT INTO route_trace(run_id,case_id,route,retried,"
                            "coverage,top1_cos) VALUES (?,?,?,?,?,?)",
                            (run_id, r["case_id"], r["route"],
                             1 if r["retried"] else 0, r["coverage"], r["top1_cos"]))
    conn.commit()
    conn.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--cases", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join("指标", "检索对比表.md"))
    ap.add_argument("--err", default=os.path.join("指标", "检索错误案例.md"))
    ap.add_argument("--db", default=cfg.EVAL_DB)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    cases = load_cases(args.cases, args.limit or None)
    print(f"[eval] 题量 {len(cases)},配置 {configs}")

    rows = {}
    for name in configs:
        print(f"[eval] 运行 {name} …")
        rows[name] = run_config(name, cases)

    summ = summarize(rows, cases)
    for name, s in summ.items():
        print(f"[eval] {name:9s} Recall@5={s['recall5']:.3f} "
              f"MRR={s['mrr']:.3f} 中位={s['lat_med']:.2f}ms "
              f"P95={s['lat_p95']:.2f}ms")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_tables(summ, rows, cases, args.out, args.err)
    write_db(args.db, rows, summ)
    print(f"[eval] 已写入 {args.out} / {args.err} / {args.db}")


if __name__ == "__main__":
    main()
