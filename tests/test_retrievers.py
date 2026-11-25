"""检索器与分块测试(不涉及嵌入模型,可离线快速运行)。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import rag, retrievers  # noqa: E402


class TestChunks(unittest.TestCase):
    def test_chunk_count_and_unique_ids(self):
        chunks = rag._parse_kb()
        self.assertGreaterEqual(len(chunks), 130)
        self.assertLessEqual(len(chunks), 160)
        ids = [c["id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_alias_parsed(self):
        chunks = rag._parse_kb()
        green = [c for c in chunks if c["plant"] == "绿萝"][0]
        self.assertIn("黄金葛", green["aliases"])


class TestRetrievers(unittest.TestCase):
    def test_tokenize_filters_stopwords(self):
        toks = retrievers.tokenize("绿萝的叶子怎么发黄了")
        self.assertIn("绿萝", toks)
        self.assertIn("叶子", toks)
        self.assertNotIn("的", toks)
        self.assertNotIn("怎么", toks)
        self.assertNotIn("了", toks)

    def test_rrf_fuse_order_and_tie(self):
        a = [("x", 9.0), ("y", 8.0)]
        b = [("y", 0.9), ("z", 0.8)]
        fused = retrievers.rrf_fuse([a, b], k=60, topk=3)
        self.assertEqual(fused[0][0], "y")   # 两路都出现,分数最高
        self.assertEqual({fused[1][0], fused[2][0]}, {"x", "z"})
        self.assertEqual([c for c, _ in fused[1:]], ["x", "z"])  # 同分按 id 稳定

    def test_bm25_finds_section(self):
        idx = retrievers.BM25Index(rag.get_chunks())
        hits = [cid for cid, _ in idx.search("绿萝多久浇水", topk=10)]
        self.assertIn("绿萝#浇水", hits)


if __name__ == "__main__":
    unittest.main()
