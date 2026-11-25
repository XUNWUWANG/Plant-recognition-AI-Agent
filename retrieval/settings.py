"""检索子系统配置:嵌入模型、缓存与知识库路径、检索超参。

只依赖根配置的 BASE_DIR;产品层不应直接读取这里的参数。
"""

import os

from config import BASE_DIR

# 嵌入模型
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBED_FALLBACK = "BAAI/bge-small-zh-v1.5"

# 缓存(可重建,不提交)
EMBED_CACHE = os.path.join(BASE_DIR, "models", "cache", "emb_qwen3_0.6b.npz")
CHUNKS_CACHE = os.path.join(BASE_DIR, "models", "cache", "chunks.json")

# 知识库与评测
KB_DIR = os.path.join(BASE_DIR, "kb")
EVAL_DIR = os.path.join(BASE_DIR, "data", "eval")
EVAL_DB = os.path.join(BASE_DIR, "指标", "retrieval", "eval.db")

# 检索超参
TOP_K = 5
FUSE_TOP_N = 10          # 每路参与融合的条数
RRF_K = 60
COVERAGE_THRESHOLD = 0.5  # BM25 回退阈值
DENSE_COS_THRESHOLD = 0.4
ANSWER_COS_THRESHOLD = 0.55  # 库外问题判定:覆盖度低且余弦低于此值视为未命中
CHUNK_MIN_LEN = 20
