import unittest
from utils.metrics_calculator import calculate_metrics

class TestMetricsCalculator(unittest.TestCase):

    def test_classification_metrics(self):
        y_train_true = [0, 1, 1, 0, 1]
        y_train_pred = [0, 1, 0, 0, 1]
        y_test_true = [1, 0, 1, 1]
        y_test_pred = [1, 0, 0, 1]
        y_train_proba = [0.1, 0.9, 0.4, 0.2, 0.8]
        y_test_proba = [0.7, 0.3, 0.2, 0.6]

        metrics = calculate_metrics(
            y_train_true, y_train_pred, y_test_true, y_test_pred,
            problem_type="classification",
            y_train_proba=y_train_proba,
            y_test_proba=y_test_proba
        )

        self.assertIn("train_accuracy", metrics)
        self.assertIn("test_accuracy", metrics)
        self.assertIn("train_precision", metrics)
        self.assertIn("test_precision", metrics)
        self.assertIn("train_recall", metrics)
        self.assertIn("test_recall", metrics)
        self.assertIn("train_f1_score", metrics)
        self.assertIn("test_f1_score", metrics)
        self.assertIn("train_roc_auc", metrics)
        self.assertIn("test_roc_auc", metrics)
        self.assertAlmostEqual(metrics["train_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["test_accuracy"], 0.75)

    def test_regression_metrics(self):
        y_train_true = [2.0, 3.0, 4.0, 5.0]
        y_train_pred = [2.1, 2.9, 4.2, 4.8]
        y_test_true = [1.0, 2.0, 3.0]
        y_test_pred = [1.2, 1.8, 2.9]

        metrics = calculate_metrics(
            y_train_true, y_train_pred, y_test_true, y_test_pred,
            problem_type="regression"
        )

        self.assertIn("train_mse", metrics)
        self.assertIn("test_mse", metrics)
        self.assertIn("train_mae", metrics)
        self.assertIn("test_mae", metrics)
        self.assertIn("train_r2", metrics)
        self.assertIn("test_r2", metrics)
        self.assertTrue(isinstance(metrics["train_mse"], float))
        self.assertTrue(isinstance(metrics["test_r2"], float))

    def test_invalid_problem_type(self):
        with self.assertRaises(ValueError):
            calculate_metrics([1], [1], [1], [1], problem_type="unknown")

if __name__ == "__main__":
    unittest.main()