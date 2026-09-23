"""对比评测:同一测试集上,本地模型 vs DeepSeek 大模型,输出 指标/对比表.md。

用法:先跑 python train.py,再 python compare.py。
测试集文件名即真实标签(如 向日葵.jpg → 真实=向日葵),仅评估类别名一致的图片。
指标诚实拆分:Top-1(预测第一名)与 Top-3(前三名命中)分开报。
"""

import glob
import json
import os

import config as cfg
from tools import classify_local, classify_plant


def is_image(p):
    return p.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp"))


def gt_stem(p, classes=None):
    """从文件名取真实标签:支持 '向日葵.jpg' 或 '向日葵_1.jpg' 两种。"""
    stem = os.path.splitext(os.path.basename(p))[0]
    if classes:
        for c in classes:
            if stem == c or stem.startswith(c + "_"):
                return c
    return stem


def main():
    with open(os.path.join("models", "classes.json"), encoding="utf-8") as f:
        classes = json.load(f)

    paths = sorted(p for p in glob.glob(os.path.join(cfg.TEST_DIR, "*"))
                   if is_image(p) and gt_stem(p, classes) in classes)
    if not paths:
        print(f"没有可评估的图片:test_set 中文件名需等于 {classes} 之一")
        return

    # 累计:local_top1, local_top3, api_top1, api_top3
    acc = {"local_top1": 0, "local_top3": 0, "api_top1": 0, "api_top3": 0}
    rows = []
    for p in paths:
        gt = gt_stem(p, classes)
        loc = classify_local(p)
        api = classify_plant(p)

        loc_top1 = gt in loc["class"] or loc["class"] in gt
        loc_top3 = loc_top1 or any(gt in c or c in gt for c, _ in loc["top3"])
        api_top1 = gt in api["plant_name"] or api["plant_name"] in gt
        api_top3 = api_top1 or any(gt in a or a in gt for a in api["alternatives"])

        for k, v in (("local_top1", loc_top1), ("local_top3", loc_top3),
                     ("api_top1", api_top1), ("api_top3", api_top3)):
            acc[k] += int(v)

        local_top3_str = "、".join(f"{c}({pr:.2f})" for c, pr in loc["top3"])
        rows.append(
            f"| {os.path.basename(p)} | {gt} | "
            f"{loc['class']}({'对' if loc_top1 else '错'}) | "
            f"{local_top3_str} | "
            f"{api['plant_name']}({'对' if api_top1 else '错'}) | "
            f"{'命中' if api_top3 else '未中'} |"
        )

    n = len(paths)
    os.makedirs("指标", exist_ok=True)
    out = [
        "# 本地模型 vs 大模型 对比评测",
        "",
        f"测试集图片数: {n}(样本较小,指标仅供参考)",
        f"- 本地模型 Top-1 准确率: {acc['local_top1']}/{n} ({acc['local_top1']/n:.0%})",
        f"- 本地模型 Top-3 命中率: {acc['local_top3']}/{n} ({acc['local_top3']/n:.0%})",
        f"- 大模型 Top-1 准确率: {acc['api_top1']}/{n} ({acc['api_top1']/n:.0%})",
        f"- 大模型 Top-3 命中率: {acc['api_top3']}/{n} ({acc['api_top3']/n:.0%})",
        "",
        "| 图片 | 真实 | 本地Top1 | 本地Top3 | 大模型Top1 | 大模型Top3 |",
        "|---|---|---|---|---|---|", *rows,
        "",
        "> 注:本地模型为大模型(DeepSeek)对具体品种的命名可能比训练类名更细",
        ">(如 '草球' vs '仙人掌'),按 Top-1/Top-3 分别统计以保持诚实可比。",
    ]
    with open(os.path.join("指标", "对比表.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print("\n".join(out))
    print(f"\n对比表已保存 -> 指标/对比表.md")


if __name__ == "__main__":
    main()
