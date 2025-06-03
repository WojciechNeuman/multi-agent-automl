import unittest
from unittest.mock import patch
from agents.evaluation_agent import run_evaluation_agent
from schemas.evaluation import EvaluationRequest, EvaluationDecision
from schemas.shared import Metadata
from models.feature import FeatureSpec
from models.model import ModelEnum

class TestEvaluationAgent(unittest.TestCase):

    def setUp(self):
        self.metadata = Metadata(
            dataset_name="Titanic",
            problem_type="classification",
            target_column="Survived"
        )
        self.selected_features = [
            FeatureSpec(name="Age", dtype="numeric", origin="raw", transformer="none"),
            FeatureSpec(name="Fare", dtype="numeric", origin="raw", transformer="none"),
            FeatureSpec(name="Sex", dtype="categorical", origin="raw", transformer="none"),
        ]
        self.request = EvaluationRequest(
            metadata=self.metadata,
            selected_features=self.selected_features,
            model_name=ModelEnum.RANDOMFOREST,
            hyperparameters={"max_depth": 5, "n_estimators": 100}
        )
        self.current_metrics = {"f1_score": 0.72, "accuracy": 0.81}
        self.history = [
            {"f1_score": 0.68, "accuracy": 0.78},
            {"f1_score": 0.70, "accuracy": 0.80},
        ]
        self.model_info = {
            "model": "RandomForest",
            "max_depth": 5,
            "n_estimators": 100,
        }
        self.optimization_goal = "maximize recall, avoid overfitting"

        self.mock_decision = EvaluationDecision(
            recommendation="continue",
            reasoning="Model performance is improving.",
            confidence=0.85
        )

    @patch("agents.evaluation_agent._call_llm")
    def test_run_evaluation_agent(self, mock_call_llm):
        mock_call_llm.return_value = self.mock_decision

        decision = run_evaluation_agent(
            request=self.request,
            current_metrics=self.current_metrics,
            history=self.history,
            model_info=self.model_info,
            optimization_goal=self.optimization_goal
        )

        self.assertEqual(decision.recommendation, self.mock_decision.recommendation)
        self.assertEqual(decision.reasoning, self.mock_decision.reasoning)
        self.assertEqual(decision.confidence, self.mock_decision.confidence)
        mock_call_llm.assert_called_once()

if __name__ == "__main__":
    unittest.main()
    