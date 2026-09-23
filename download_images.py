"""批量下载某类植物图片到 data/<类名>/,并清理无效/过小的图。

用法:python download_images.py "<搜索词>" [张数]
示例:python download_images.py "绿萝 盆栽" 40
"""

import os
import sys

from PIL import Image

MIN_SIZE = 3 * 1024   # 小于 3KB 视为无效
MIN_W = MIN_H = 200   # 小于 200px 视为无效


def download(keyword, target_dir, n=40):
    os.makedirs(target_dir, exist_ok=True)
    # 延迟导入,避免无网络时脚本报错
    from icrawler.builtin import BaiduImageCrawler
    crawler = BaiduImageCrawler(storage={"root_dir": target_dir})
    crawler.crawl(keyword=keyword, max_num=n)


def clean_dir(target_dir):
    """删除损坏、过小、过模糊的图片,返回删除数量。"""
    bad = 0
    for fn in os.listdir(target_dir):
        p = os.path.join(target_dir, fn)
        try:
            im = Image.open(p)
            im.verify()
            w, h = im.size
            if os.path.getsize(p) < MIN_SIZE or w < MIN_W or h < MIN_H:
                os.remove(p)
                bad += 1
        except Exception:
            os.remove(p)
            bad += 1
    return bad


if __name__ == "__main__":
    keyword = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    target = os.path.join("data", keyword.replace(" ", "_"))
    print(f"下载「{keyword}」-> {target}(目标 {n} 张)")
    download(keyword, target, n)
    cleaned = clean_dir(target)
    print(f"清理 {cleaned} 张无效图,剩余 {len(os.listdir(target))} 张")
