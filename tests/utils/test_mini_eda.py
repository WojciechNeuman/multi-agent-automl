import unittest
import pandas as pd
from utils.mini_eda import compute_basic_stats

class TestMiniEDA(unittest.TestCase):

    def setUp(self):
        self.df_numeric = pd.DataFrame({
            "Age": [22, 38, 26, 35, 28],
            "Fare": [7.25, 71.2833, 7.925, 53.1, 8.05],
            "Pclass": [3, 1, 3, 1, 3],
            "Survived": [0, 1, 1, 0, 1]
        })

        self.df_categorical = pd.DataFrame({
            "Sex": ["male", "female", "female", "male", "female"],
            "Embarked": ["S", "C", "S", "C", "Q"]
        })

        self.df_text = pd.DataFrame({
            "Name": ["John Doe", "Jane Doe", "Alice Smith", "Bob Brown", "Charlie Wilson"],
            "Review": ["Excellent", "Good", "Average", "Bad", "Terrible"]
        })

        self.df_datetime = pd.DataFrame({
            "JoinDate": pd.to_datetime(["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"]),
            "LastLogin": pd.to_datetime(["2022-01-15", "2023-02-01", "2024-03-01", "2025-04-01", "2026-05-01"])
        })

    def test_numeric_stats(self):
        stats = compute_basic_stats(self.df_numeric, target_name="Survived")
        self.assertEqual(stats["Age"].dtype, "numeric")
        self.assertAlmostEqual(stats["Age"].mean, 29.8, places=1)
        self.assertAlmostEqual(stats["Age"].std, 6.6, places=1)
        self.assertAlmostEqual(stats["Age"].corr_target, 0.18, places=1)

    def test_categorical_stats(self):
        stats = compute_basic_stats(self.df_categorical)
        self.assertEqual(stats["Sex"].dtype, "categorical")
        self.assertEqual(stats["Sex"].cardinality, 2)
        self.assertAlmostEqual(stats["Sex"].top_freq, 0.6, places=1)

    def test_text_stats(self):
        stats = compute_basic_stats(self.df_text)
        self.assertEqual(stats["Name"].dtype, "categorical")

    def test_datetime_stats(self):
        stats = compute_basic_stats(self.df_datetime)
        self.assertEqual(stats["JoinDate"].dtype, "datetime")
        self.assertEqual(stats["JoinDate"].span_days, 1461)

    def test_empty_dataframe(self):
        stats = compute_basic_stats(pd.DataFrame())
        self.assertEqual(len(stats), 0)

    def test_missing_values(self):
        df_missing = self.df_numeric.copy()
        df_missing.loc[0, "Age"] = None
        stats = compute_basic_stats(df_missing, target_name="Survived")
        self.assertAlmostEqual(stats["Age"].missing_pct, 0.2, places=1)

    def test_target_correlation(self):
        df_corr = pd.DataFrame({
            "Feature": [1, 2, 3, 4, 5],
            "Target": [1, 0, 1, 0, 1]
        })
        stats = compute_basic_stats(df_corr, target_name="Target")
        self.assertAlmostEqual(stats["Feature"].corr_target, 0, places=1)


if __name__ == "__main__":
    unittest.main()
