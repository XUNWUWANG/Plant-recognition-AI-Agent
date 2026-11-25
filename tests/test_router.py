"""路由决策与回退逻辑测试。"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import rag, router  # noqa: E402


class TestRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names, cls.primary = router.collect_names(rag.get_chunks())

    def test_decision_table(self):
        d = router.decide
        self.assertEqual(d("", "绿萝", self.names, self.primary), router.ROUTE_BM25)
        self.assertEqual(d("绿萝叶子发黄", None, self.names, self.primary),
                         router.ROUTE_HYBRID)
        self.assertEqual(d("绿萝怎么养", None, self.names, self.primary),
                         router.ROUTE_BM25)
        self.assertEqual(d("黄金葛怎么养", None, self.names, self.primary),
                         router.ROUTE_HYBRID)
        self.assertEqual(d("叶子发黄", None, self.names, self.primary),
                         router.ROUTE_DENSE)
        self.assertEqual(d("随便问问", None, self.names, self.primary),
                         router.ROUTE_HYBRID)

    def test_coverage(self):
        self.assertEqual(router.coverage("", "任意文本"), 1.0)
        self.assertGreater(router.coverage("叶子发黄", "叶子发黄怎么办"), 0.9)
        self.assertEqual(router.coverage("月季黑斑", "绿萝浇水"), 0.0)

    def test_should_retry(self):
        self.assertEqual(
            router.should_retry(router.ROUTE_BM25, "完全不相关", "叶子发黄", None),
            router.ROUTE_HYBRID)
        self.assertIsNone(
            router.should_retry(router.ROUTE_BM25, "叶子发黄怎么办", "叶子发黄", None))
        self.assertEqual(
            router.should_retry(router.ROUTE_DENSE, "", "叶子发黄", 0.1),
            router.ROUTE_HYBRID)
        self.assertIsNone(
            router.should_retry(router.ROUTE_DENSE, "", "叶子发黄", 0.9))
        self.assertIsNone(
            router.should_retry(router.ROUTE_HYBRID, "", "叶子发黄", 0.1))


if __name__ == "__main__":
    unittest.main()
