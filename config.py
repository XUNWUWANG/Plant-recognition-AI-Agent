"""全局配置:API key、模型、CV 路径。所有路径都以本文件所在目录为准。

检索子系统自己的配置见 retrieval/settings.py。
"""

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

DATA_DIR = os.path.join(BASE_DIR, "data")        # 本地训练集
TEST_DIR = os.path.join(BASE_DIR, "test_set")    # 对比测试集
CKPT = os.path.join(BASE_DIR, "models", "best.pt")  # 本地模型保存位置
NUM_CLASSES = 20  # 与 models/classes.json 一致


def require_api_key():
    """懒校验 API Key;检索/评测等离线功能不应在 import 期失败。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY,请检查 .env 文件")
    return DEEPSEEK_API_KEY
