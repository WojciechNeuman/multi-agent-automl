from unittest import TestCase
import unittest
from pydantic import ValidationError
from schemas.feature import FeatureSelectionRequest, FeatureOverview, Metadata, LLMConfig, FeatureSelectionResponse, FeatureSpec


class TestFeatureSelectionRequest(TestCase):
    
    def setUp(self):
        self.metadata = Metadata(
            dataset_name="Titanic",
            problem_type="classification",
            target_column="Survived"
        )

        self.basic_stats = {
            "Age": FeatureOverview(
                dtype="numeric",
                missing_pct=0.0,
                cardinality=80,
                mean=29.7,
                median=28.0,
                std=14.5,
                skewness=0.5,
                kurtosis=0.2,
                min_val=0.4,
                max_val=80.0,
                corr_target=0.11
            ),
            "Sex": FeatureOverview(
                dtype="categorical",
                missing_pct=0.0,
                cardinality=2,
                top_freq=0.65,
                rare_pct=0.0
            ),
            "Name": FeatureOverview(
                dtype="text",
                missing_pct=0.0,
                cardinality=500,
                avg_length=23.6
            ),
            "JoinDate": FeatureOverview(
                dtype="datetime",
                missing_pct=0.0,
                cardinality=500,
                span_days=1461
            ),
        }

        self.data_sample = {
            "Age": [22, 38, 26, 35, 28],
            "Sex": ["male", "female", "female", "male", "female"],
            "Name": ["John Doe", "Jane Doe", "Alice Smith", "Bob Brown", "Charlie Wilson"],
            "JoinDate": ["2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01", "2026-01-01"],
            "Survived": [0, 1, 1, 0, 1]
        }

    def test_valid_feature_selection_request(self):
        req = FeatureSelectionRequest(
            metadata=self.metadata,
            basic_stats=self.basic_stats,
            data_sample=self.data_sample,
            max_features=4,
            llm_config=LLMConfig(model="gpt-3.5-turbo")
        )

        self.assertEqual(req.metadata.dataset_name, "Titanic")
        self.assertEqual(req.metadata.problem_type, "classification")
        self.assertIn("Age", req.basic_stats)
        self.assertEqual(req.basic_stats["Age"].mean, 29.7)
        self.assertEqual(req.basic_stats["Sex"].top_freq, 0.65)
        self.assertEqual(req.basic_stats["JoinDate"].span_days, 1461)

    def test_invalid_feature_selection_request_missing_data_sample(self):
        with self.assertRaises(ValidationError):
            FeatureSelectionRequest(
                metadata=self.metadata,
                basic_stats=self.basic_stats,
                data_sample=None  # Use None to trigger the validation error
            )

    def test_invalid_basic_stats(self):
        with self.assertRaises(ValidationError):
            FeatureSelectionRequest(
                metadata=self.metadata,
                basic_stats=[{
                    "Age": {
                        "dtype": "numeric",
                        "missing_pct": 0.0,
                        "cardinality": 80
                    }
                }],
                data_sample=self.data_sample
            )

    def test_invalid_problem_type(self):
        with self.assertRaises(ValidationError):
            Metadata(
                dataset_name="Insurance",
                problem_type="clustering",
                target_column="charges"
            )


class TestFeatureSelectionResponse(TestCase):

    def test_valid_feature_selection_response(self):
        response = FeatureSelectionResponse(
            selected_features=[
                FeatureSpec(
                    name="Age",
                    dtype="numeric",
                    origin="raw",
                    transformer="original",
                    params={"strategy": "median"},
                    importance=0.75
                ),
                FeatureSpec(
                    name="Sex",
                    dtype="categorical",
                    origin="raw",
                    transformer="original",
                    params={"strategy": "most_frequent"},
                    importance=0.65
                ),
            ],
            preprocessing_code="dummy_pipeline_blob",
            reasoning="Selected features based on mutual information scores."
        )

        self.assertEqual(len(response.selected_features), 2)
        self.assertEqual(response.selected_features[0].name, "Age")
        self.assertEqual(response.selected_features[1].dtype, "categorical")
        self.assertEqual(response.preprocessing_code, "dummy_pipeline_blob")
        self.assertIn("mutual information", response.reasoning)

if __name__ == "__main__":
    unittest.main()
