import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from unittest import TestCase
from pydantic import ValidationError
from schemas.evaluation import EvaluationRequest, EvaluationResponse
from schemas.shared import Metadata
from models.feature import FeatureSpec
from models.model import ModelEnum

class TestEvaluationSchemas(TestCase):
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

    def test_valid_evaluation_request(self):
        req = EvaluationRequest(
            metadata=self.metadata,
            selected_features=self.features,
            model_name=ModelEnum.RANDOMFOREST,
            hyperparameters={"n_estimators": 100, "max_depth": 5}
        )
        self.assertEqual(req.model_name, ModelEnum.RANDOMFOREST)
        self.assertIn("n_estimators", req.hyperparameters)

    def test_invalid_evaluation_request_missing_fields(self):
        with self.assertRaises(ValidationError):
            EvaluationRequest(
                metadata=self.metadata,
                selected_features=[],
                model_name=None,
                hyperparameters=None
            )

    def test_valid_evaluation_response(self):
        res = EvaluationResponse(
            reasoning="Good model accuracy on test set.",
            metrics={"accuracy": 0.91, "f1": 0.88},
            visualizations={"roc_curve": "base64string_here"}
        )
        self.assertGreater(res.metrics["accuracy"], 0.9)

    def test_invalid_evaluation_response_missing_reasoning(self):
        with self.assertRaises(ValidationError):
            EvaluationResponse(
                metrics={"accuracy": 0.9}
            )
            