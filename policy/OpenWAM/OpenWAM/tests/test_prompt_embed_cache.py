"""Regression tests for prompt-cache eviction in long-running policy servers."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openwam.deploy.engine import _BoundedPromptEmbedCache


class PromptEmbedCacheTests(unittest.TestCase):
    def test_more_than_32_distinct_instructions(self):
        cache = _BoundedPromptEmbedCache(32)
        for i in range(256):
            cache[f"instruction {i}"] = (i, i + 1)
            self.assertEqual(cache[f"instruction {i}"], (i, i + 1))
            self.assertEqual(len(cache), min(i + 1, 32))
        self.assertEqual(list(cache), [f"instruction {i}" for i in range(224, 256)])

    def test_hit_protects_least_recently_used_entry(self):
        cache = _BoundedPromptEmbedCache(2)
        cache['a'] = 1
        cache['b'] = 2
        self.assertEqual(cache['a'], 1)
        cache['c'] = 3
        self.assertEqual(list(cache), ['a', 'c'])
        self.assertNotIn('b', cache)

    def test_overwrite_preserves_capacity_and_refreshes_recency(self):
        cache = _BoundedPromptEmbedCache(2)
        cache['a'] = 1
        cache['b'] = 2
        cache['a'] = 10
        cache['c'] = 3
        self.assertEqual(cache['a'], 10)
        self.assertEqual(len(cache), 2)
        self.assertNotIn('b', cache)

    def test_single_entry_repeated_evictions(self):
        cache = _BoundedPromptEmbedCache(1)
        for i in range(100):
            cache[i] = i
            self.assertEqual(list(cache), [i])
            self.assertEqual(cache[i], i)


if __name__ == '__main__':
    unittest.main()
