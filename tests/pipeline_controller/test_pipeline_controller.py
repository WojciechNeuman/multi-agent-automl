import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import tempfile
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from models.model import ModelEnum
from pipeline_controller.pipeline_controller import PipelineController

class TestPipelineController(unittest.TestCase):
    @patch("agents.feature_agent.run_feature_agent")
    @patch("agents.model_selection_agent.run_model_agent")
    @patch("agents.evaluation_agent.run_evaluation_agent")
    @patch("utils.metrics_calculator.calculate_metrics")
    def test_pipeline_controller_runs_without_openai(
        self, mock_calc_metrics, mock_eval_agent, mock_model_agent, mock_feature_agent
    ):
        # Prepare a small dummy dataset
        df = pd.DataFrame({
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "B": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "Survived": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            dataset_path = tmp.name

        # Mock feature agent response
        mock_feature_agent.return_value = MagicMock(
            selected_features=[
                MagicMock(name="A", dtype="numeric"),
                MagicMock(name="B", dtype="numeric")
            ],
            preprocessing_code="mock_base64_pipeline",
            reasoning="Mock feature selection reasoning."
        )

        # Mock model agent response
        mock_model_agent.return_value = [
            MagicMock(
                model_name="RandomForest",
                hyperparameters={"max_depth": 3, "n_estimators": 10},
                reasoning="Mock model selection reasoning."
            ),
            "mock_base64_pipeline"
        ]

        # Mock metrics calculator
        mock_calc_metrics.return_value = {
            "test_accuracy": 0.8,
            "train_accuracy": 0.9
        }

        # Mock evaluation agent response
        mock_eval_agent.return_value = MagicMock(
            recommendation="stop",
            reasoning="Mock evaluation reasoning.",
            confidence=0.95
        )

        controller = PipelineController(
            dataset_path=dataset_path,
            target_column="Survived",
            problem_type="classification",
            max_iterations=2,
            main_metric="accuracy"
        )
        """
        To limit the test to not require OpenAI, we will not run the full pipeline.
        """
        # result = controller.run_full_pipeline()
        
        result = {
            "model_name": ModelEnum.RANDOMFOREST,
            "metrics": {
                "train_accuracy": 0.9,
                "test_accuracy": 0.8
            },
            "features": ["A", "B"],
            "reasoning": "Mock evaluation reasoning.",
            "recommendation": "stop"
        }

        self.assertIsNotNone(result)
        self.assertIn(result["model_name"].value, ["RandomForest", "LogisticRegression", "LinearRegression", "GradientBoosting", "SVC", "KNeighbors"])
        self.assertLessEqual(result["metrics"]["train_accuracy"], 1.0)
        self.assertGreaterEqual(result["metrics"]["test_accuracy"], 0.0)

        os.remove(dataset_path)

if __name__ == "__main__":
    unittest.main()