"""向后兼容测试:原调用链仍可用,且离线检索不需要 API Key。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import rag  # noqa: E402
import tools  # noqa: E402


class TestCompat(unittest.TestCase):
    def test_retrieve_by_name(self):
        res = rag.retrieve("绿萝")
        self.assertTrue(res)
        self.assertTrue(all(r.startswith("## ") for r in res))
        self.assertIn("绿萝", res[0])

    def test_offline_advice(self):
        adv = tools.plant_advice_offline("绿萝")
        self.assertIn("source", adv)
        self.assertEqual(adv["source"], "kb")
        self.assertTrue(adv["content"].strip())

    def test_offline_advice_with_question(self):
        adv = tools.plant_advice_offline("绿萝", "叶子发黄怎么办")
        self.assertEqual(adv["source"], "kb")

    def test_naive_retrieve(self):
        self.assertTrue(rag.naive_retrieve("绿萝"))

    def test_retrieve_debug_shape(self):
        dbg = rag.retrieve_debug("绿萝怎么养")
        for key in ("route", "retried", "latency_ms", "coverage",
                    "top1_cos", "results"):
            self.assertIn(key, dbg)
        self.assertTrue(dbg["results"])

    def test_entity_filter(self):
        dbg = rag.retrieve_debug("绿萝多久浇一次水")
        self.assertTrue(dbg["results"])
        self.assertEqual(dbg["results"][0]["plant"], "绿萝")

    def test_unanswerable_gate(self):
        _ctx, hit = rag.retrieve_hit("我的飞船坏了")
        self.assertFalse(hit)
        _ctx, hit = rag.retrieve_hit("绿萝怎么养")
        self.assertTrue(hit)


if __name__ == "__main__":
    unittest.main()
