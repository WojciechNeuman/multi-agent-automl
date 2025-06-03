import unittest
from unittest.mock import patch
from agents.feature_agent import run_feature_agent, FeatureSelectionRequest, FeatureSelectionResponse
from schemas.feature import FeatureSpec


class TestFeatureAgent(unittest.TestCase):

    def setUp(self):
        self.mock_request = FeatureSelectionRequest(
            metadata={
                "dataset_name": "mock_dataset",
                "problem_type": "classification",
                "target_column": "target",
            },
            data_sample={
                "feature1": [1, 2, 3, 4, 5],
                "feature2": ["A", "B", "A", "B", "C"],
                "target": [0, 1, 0, 1, 0],
            },
            selection_goal="maximize_accuracy",
            max_features=5,
            basic_stats={},  
            llm_config={
                "model": "mock_model",
                "temperature": 0.5,
                "max_tokens": 128,
            },
        )

        self.mock_response = FeatureSelectionResponse(
            selected_features=[
                FeatureSpec(name="feature1", dtype="numeric", origin="raw", transformer="none"),
                FeatureSpec(name="feature2", dtype="categorical", origin="raw", transformer="none"),
            ],
            preprocessing_code="mock_pipeline_blob",
            reasoning="Mock reasoning for feature selection.",
        )

    @patch("agents.feature_agent._build_pipeline_blob", return_value="mock_pipeline_blob")
    @patch("agents.feature_agent._call_llm")
    def test_run_feature_agent(self, mock_call_llm, mock_build_pipeline_blob):
        """Test the run_feature_agent function with mocked LLM and pipeline blob."""
        mock_call_llm.return_value = self.mock_response

        response = run_feature_agent(self.mock_request)

        self.assertEqual(response.selected_features, self.mock_response.selected_features)
        self.assertEqual(response.preprocessing_code, self.mock_response.preprocessing_code)
        self.assertEqual(response.reasoning, self.mock_response.reasoning)

        mock_call_llm.assert_called_once()
        mock_build_pipeline_blob.assert_called_once()


if __name__ == "__main__":
    unittest.main()
    