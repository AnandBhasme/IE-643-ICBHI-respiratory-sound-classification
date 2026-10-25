"""Small metric checks required before classifier training."""

import unittest

import numpy as np

from util.metrics import evaluate_confusion


class ICBHIMetricsTest(unittest.TestCase):
    def test_always_normal_predictor(self):
        cm = np.array([
            [3, 0, 0, 0],
            [4, 0, 0, 0],
            [2, 0, 0, 0],
            [1, 0, 0, 0],
        ])
        result = evaluate_confusion(cm)
        self.assertEqual(result["Sp"], 1.0)
        self.assertEqual(result["Se"], 0.0)
        self.assertEqual(result["Score"], 0.5)

    def test_perfect_predictor(self):
        result = evaluate_confusion(np.diag([3, 4, 2, 1]))
        self.assertEqual(result["Sp"], 1.0)
        self.assertEqual(result["Se"], 1.0)
        self.assertEqual(result["Score"], 1.0)


if __name__ == "__main__":
    unittest.main()
