from unittest import TestCase
from pydantic import ValidationError
from schemas.model_selection import ModelSelectionRequest, ModelSelectionResponse
from schemas.shared import Metadata
import pandas as pd

class TestModelSchemas(TestCase):

    def setUp(self):
        self.metadata = Metadata(
            dataset_name="Titanic",
            problem_type="classification",
            target_column="Survived"
        )
        self.sample_data = pd.DataFrame([
            {"age": 22, "fare": 7.25, "Survived": 0},
            {"age": 38, "fare": 71.2833, "Survived": 1}
        ]).to_dict()

    def test_valid_model_selection_request(self):
        request = ModelSelectionRequest(
            metadata=self.metadata,
            selected_features=["age", "fare"],
            data=self.sample_data
        )

        self.assertEqual(request.metadata.dataset_name, "Titanic")
        self.assertIn("fare", request.selected_features)

    def test_invalid_model_selection_request_missing_features(self):
        with self.assertRaises(ValidationError):
            ModelSelectionRequest(
                metadata=self.metadata,
                selected_features=None,
                data=self.sample_data
            )

    def test_valid_model_selection_response(self):
        response = ModelSelectionResponse(
            model_name="RandomForestClassifier",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            reasoning="Chosen due to good performance on tabular data."
        )

        self.assertEqual(response.model_name, "RandomForestClassifier")
        self.assertIn("n_estimators", response.hyperparameters)

    def test_invalid_model_selection_response_missing_model_name(self):
        with self.assertRaises(ValidationError):
            ModelSelectionResponse(
                hyperparameters={"max_depth": 3},
                reasoning="Fallback"
            )
