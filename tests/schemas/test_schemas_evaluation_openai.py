from unittest import TestCase
from unittest.mock import patch
from schemas.evaluation import EvaluationRequest, EvaluationResponse
from schemas.shared import Metadata
from models.feature import FeatureSpec
from models.model import ModelEnum
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class TestEvaluationAgentWithMock(TestCase):
    def setUp(self):
        self.metadata = Metadata(
            dataset_name="Titanic",
            problem_type="classification",
            target_column="Survived"
        )
        self.features = [
            FeatureSpec(name="age", dtype="numeric", origin="raw", transformer="none"),
            FeatureSpec(name="fare", dtype="numeric", origin="raw", transformer="none"),
            FeatureSpec(name="sex_male", dtype="categorical", origin="raw", transformer="none"),
        ]

        self.request = EvaluationRequest(
            metadata=self.metadata,
            selected_features=self.features,
            model_name=ModelEnum.RANDOMFOREST,
            hyperparameters={"n_estimators": 100, "max_depth": 5}
        )

        self.mock_response = EvaluationResponse(
            reasoning="The model achieved strong results with high precision.",
            metrics={"accuracy": 0.92, "precision": 0.93, "recall": 0.89},
            visualizations={"confusion_matrix": "base64img"}
        )

    def test_mocked_evaluation_agent_response(self):
        from openai import OpenAI
        from instructor import from_openai

        client = from_openai(OpenAI(api_key="sk-FAKE-KEY"))

        with patch.object(client.chat.completions, 'create', return_value=self.mock_response) as mock_create:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an evaluation assistant for ML models."},
                    {"role": "user", "content": f"Evaluate this model: {self.request.model_dump_json()}"}
                ],
                response_model=EvaluationResponse
            )

            self.assertIn("precision", response.metrics)
            self.assertEqual(response.reasoning[:9], "The model")
            self.assertEqual(response.visualizations["confusion_matrix"], "base64img")
            mock_create.assert_called_once()
            