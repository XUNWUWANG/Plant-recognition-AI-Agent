"""三路检索器:BM25、稠密向量、RRF 融合。重依赖懒加载。"""

import hashlib
import json
import os
import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

import config as cfg

QUERY_INSTRUCTION = "给定一个植物养护问题，检索能回答它的知识片段。"

STOPWORDS = {
    "的", "了", "吗", "呢", "啊", "呀", "吧", "么", "是", "我", "你", "他", "她",
    "它", "在", "和", "与", "或", "也", "都", "就", "还", "很", "太", "有",
    "怎么", "怎样", "如何", "什么", "多少", "多久", "请问", "一下", "可以",
    "需要", "应该", "能", "会", "要", "想", "这", "那", "个", "种", "该",
    "一次", "一个", "一直", "一般", "平时", "大概", "左右", "比较", "非常",
}

_TOKEN_RE = re.compile(r"^[\w]+$")


def tokenize(text):
    """jieba 分词 → 小写 → 过滤空白、标点与停用词。"""
    out = []
    for w in jieba.lcut(text or ""):
        w = w.strip().lower()
        if not w or w in STOPWORDS:
            continue
        if not _TOKEN_RE.match(w):
            continue
        out.append(w)
    return out


def _norm_rows(mat):
    """按行做 L2 归一化(零向量保持为零)。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class BM25Index:
    """jieba 分词 + BM25Okapi 稀疏检索。"""

    def __init__(self, chunks):
        self.chunks = chunks
        self.ids = [c["id"] for c in chunks]
        corpus = []
        for c in chunks:
            toks = tokenize(c["plant"] + " " + c["section"] + " " + c["text"])
            # 汉字级补偿:中文词组切分(晒焦/直晒)后,单字问法(晒)也能命中
            toks += [ch for ch in c["section"] + c["text"] if "一" <= ch <= "鿿"]
            corpus.append(toks)
        self._corpus = corpus
        self._bm25 = BM25Okapi(corpus, k1=1.5, b=0.75)

    def search(self, query, topk=10):
        """返回 [(chunk_id, score)],按分数降序。"""
        toks = tokenize(query)
        if not toks:
            return []
        scores = self._bm25.get_scores(toks)
        order = np.argsort(-np.asarray(scores), kind="stable")
        out = []
        for i in order[:topk]:
            s = float(scores[i])
            if s <= 0:
                continue
            out.append((self.ids[i], s))
        return out


def _chunks_hash(chunks):
    h = hashlib.sha1()
    for c in chunks:
        h.update(c["id"].encode("utf-8"))
        h.update(b"\0")
        h.update(c["text"].encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


class DenseIndex:
    """稠密向量检索:优先 Sentence-Transformers,失败时用 transformers 直载。

    结果缓存在 npz;kb 或模型变化时自动重建。
    """

    def __init__(self, chunks, model_name=None, cache_path=None):
        self.chunks = chunks
        self.ids = [c["id"] for c in chunks]
        self.model_name = model_name or cfg.EMBED_MODEL
        self.cache_path = cache_path or cfg.EMBED_CACHE
        self.kb_hash = _chunks_hash(chunks)
        self._model = None
        self._backend = None
        self._used_model = None
        self._from_cache = False
        self._matrix = None
        self._load_or_build()

    # ---------- 模型 ----------
    def _load_model(self):
        if self._model is not None:
            return self._model
        candidate_models = [self.model_name]
        if not self._from_cache and self.model_name != cfg.EMBED_FALLBACK:
            candidate_models.append(cfg.EMBED_FALLBACK)

        st_cls = None
        try:
            from sentence_transformers import SentenceTransformer
            st_cls = SentenceTransformer
        except Exception as e:  # noqa: BLE001
            print(f"[retrievers] sentence-transformers 不可用,改用 transformers:{e}")

        last_err = None
        for name in candidate_models:
            if st_cls is not None:
                model = self._try_load(st_cls, name)
                if model is not None:
                    self._backend, self._model, self._used_model = "st", model, name
                    return model
            model = self._load_hf(name)
            if model is not None:
                self._backend, self._model, self._used_model = "hf", model, name
                return model
            last_err = name
        raise RuntimeError(
            f"嵌入模型加载失败({self.model_name} 与 fallback 均失败),"
            f"请检查网络或设置 HF_ENDPOINT。最后尝试:{last_err}")

    def _try_load(self, cls, name):
        try:
            return cls(name, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"[retrievers] Sentence-Transformers 加载 {name} 失败:{e}")
            if not os.environ.get("HF_ENDPOINT"):
                os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
                try:
                    return cls(name, trust_remote_code=True)
                except Exception as e2:  # noqa: BLE001
                    print(f"[retrievers] 镜像加载 {name} 仍失败:{e2}")
        return None

    def _load_hf(self, name):
        try:
            from transformers import AutoModel, AutoTokenizer
            print(f"[retrievers] 使用 transformers 直载 {name} …")
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
            return {"tok": tok, "model": model}
        except Exception as e:  # noqa: BLE001
            print(f"[retrievers] transformers 直载 {name} 失败:{e}")
            return None

    # ---------- 编码 ----------
    def _encode_docs(self, texts):
        model = self._load_model()
        print(f"[retrievers] 编码 {len(texts)} 个知识片段(首次较慢)…")
        if self._backend == "st":
            return model.encode(texts, normalize_embeddings=True,
                                batch_size=32, show_progress_bar=False)
        return self._hf_encode(texts)

    def _encode_query(self, query):
        self._load_model()
        q = query
        if "qwen" in self._used_model.lower():
            q = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        if self._backend == "st":
            emb = self._model.encode([q], normalize_embeddings=True,
                                     show_progress_bar=False)
            return np.asarray(emb[0], dtype=np.float32)
        return self._hf_encode([q])[0]

    def _hf_encode(self, texts):
        import torch
        from torch.nn import functional as F

        tok, model = self._model["tok"], self._model["model"]
        outs = []
        for i in range(0, len(texts), 16):
            batch = texts[i:i + 16]
            enc = tok(batch, padding=True, truncation=True, max_length=512,
                      return_tensors="pt")
            with torch.no_grad():
                hidden = model(**enc).last_hidden_state
            if "qwen" in self._used_model.lower():
                mask = enc["attention_mask"]
                left_pad = bool((mask[:, -1].sum() == mask.shape[0]).item())
                if left_pad:
                    emb = hidden[:, -1]
                else:
                    idx = mask.sum(dim=1) - 1
                    emb = hidden[torch.arange(hidden.size(0)), idx]
            else:
                emb = hidden[:, 0]
            outs.append(F.normalize(emb, p=2, dim=1).cpu().numpy())
        return np.vstack(outs)

    # ---------- 缓存 ----------
    def _load_or_build(self):
        if self._load_cache():
            return
        texts = [c["text"] for c in self.chunks]
        emb = self._encode_docs(texts)
        np.savez(self.cache_path,
                 emb=np.asarray(emb, dtype=np.float32),
                 ids=np.array(self.ids, dtype=object),
                 kb_hash=np.array(self.kb_hash),
                 model=np.array(self._used_model or self.model_name))
        self._matrix = _norm_rows(np.asarray(emb, dtype=np.float32))
        print(f"[retrievers] 向量已缓存:{self.cache_path}")

    def _load_cache(self):
        if not os.path.exists(self.cache_path):
            return False
        try:
            data = np.load(self.cache_path, allow_pickle=True)
            if str(data["kb_hash"]) != self.kb_hash:
                print("[retrievers] kb 已变化,重建向量缓存")
                return False
            if list(data["ids"]) != self.ids:
                return False
            self._matrix = _norm_rows(np.asarray(data["emb"], dtype=np.float32))
            self._used_model = str(data["model"]) if "model" in data else self.model_name
            self.model_name = self._used_model
            self._from_cache = True
            print(f"[retrievers] 命中向量缓存:{self.cache_path}")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[retrievers] 向量缓存读取失败,重建:{e}")
            return False

    def search(self, query, topk=10):
        """返回 [(chunk_id, cosine)],按相似度降序。"""
        if not query.strip():
            return []
        q = self._encode_query(query)
        scores = self._matrix @ q
        order = np.argsort(-scores, kind="stable")
        return [(self.ids[i], float(scores[i])) for i in order[:topk]]


def rrf_fuse(ranked_lists, k=None, topk=5):
    """score = Σ 1/(k + rank),rank 从 1 开始;同分按 chunk_id 稳定排序。"""
    k = cfg.RRF_K if k is None else k
    acc = {}
    for ranked in ranked_lists:
        for rank, (cid, _score) in enumerate(ranked, start=1):
            acc[cid] = acc.get(cid, 0.0) + 1.0 / (k + rank)
    items = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:topk]
