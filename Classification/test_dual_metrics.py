import unittest

import numpy as np

from audit_metric_families import metrics_from_accuracy, reference_rows
from util.metrics import evaluate_confusion


class DualMetricReferenceTests(unittest.TestCase):
    def test_petmezas_published_rows_fit_micro_identities(self):
        official = metrics_from_accuracy(0.4737)
        self.assertAlmostEqual(official["Sp"], 0.8246, places=3)
        self.assertAlmostEqual(official["Acc"], 0.7369, places=3)
        self.assertAlmostEqual(official["Score"], 0.6492, places=3)
        tenfold = metrics_from_accuracy(0.5278)
        self.assertAlmostEqual(tenfold["Sp"], 0.8426, places=3)
        self.assertAlmostEqual(tenfold["Acc"], 0.7639, places=3)
        self.assertAlmostEqual(tenfold["Score"], 0.6852, places=3)

    def test_petmezas_figure5_matrix_official_metrics(self):
        cm = np.array([[2688,714,197,43],[1169,575,74,46],[570,169,95,52],[277,117,54,58]])
        metrics = evaluate_confusion(cm)
        self.assertAlmostEqual(metrics["Se"], 0.2236, places=3)
        self.assertAlmostEqual(metrics["Sp"], 0.7381, places=3)
        self.assertAlmostEqual(metrics["Score"], 0.4808, places=3)

    def test_always_normal_official_test_reference(self):
        cm = np.zeros((4, 4), dtype=np.int64)
        cm[:, 0] = [1579,649,385,143]
        metrics = evaluate_confusion(cm)
        self.assertAlmostEqual(metrics["Se"], 0.0)
        self.assertAlmostEqual(metrics["Sp"], 1.0)
        self.assertAlmostEqual(metrics["Score"], 0.5)
        self.assertAlmostEqual(metrics["Se_micro"], 0.57293, places=4)
        self.assertAlmostEqual(metrics["Sp_micro"], 0.85764, places=4)
        self.assertAlmostEqual(metrics["Score_micro"], 0.71528, places=4)

    def test_reference_rows_are_present_and_match_reported_values(self):
        rows = {row["run_id"]: row for row in reference_rows()}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows["REF_PETMEZAS_OFFICIAL_PUBLISHED"]["best_micro_Score"], 64.92)
        self.assertEqual(rows["REF_PETMEZAS_10FOLD_PUBLISHED"]["best_micro_Score"], 68.52)
        figure5 = rows["REF_PETMEZAS_10FOLD_FIG5_MATRIX"]
        self.assertAlmostEqual(float(figure5["best_Se"]), 22.36, places=2)
        self.assertAlmostEqual(float(figure5["best_Sp"]), 73.81, places=2)
        self.assertAlmostEqual(float(figure5["best_Score"]), 48.08, places=2)
        baseline = rows["REF_ALWAYS_NORMAL_OFFICIAL_TEST"]
        self.assertAlmostEqual(float(baseline["best_micro_Score"]), 71.53, places=2)


if __name__ == "__main__":
    unittest.main()
