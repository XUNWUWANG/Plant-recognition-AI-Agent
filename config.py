"""全局配置:API key、模型、路径。所有路径都以本文件所在目录为准。"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_env(path):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_env(os.path.join(BASE_DIR, ".env"))

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-flash-vision-exp"  # 多模态,识别与生成共用

KB_DIR = os.path.join(BASE_DIR, "kb")            # 植物知识库
DATA_DIR = os.path.join(BASE_DIR, "data")        # 本地训练集(6-8 类)
TEST_DIR = os.path.join(BASE_DIR, "test_set")    # 对比测试集
CKPT = os.path.join(BASE_DIR, "models", "best.pt")  # 本地模型保存位置
NUM_CLASSES = 8  # 与 data/ 类别数一致

if not DEEPSEEK_API_KEY:
    raise RuntimeError("缺少 DEEPSEEK_API_KEY,请检查 code/.env 文件")
