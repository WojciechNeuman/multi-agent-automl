from unittest import TestCase
from unittest.mock import patch

import sys
import os
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from schemas.model_selection import ModelEnum, ModelSelectionRequest, ModelSelectionResponse
from schemas.shared import Metadata


class TestModelAgentWithMock(TestCase):

    def setUp(self):
        self.metadata = Metadata(
            dataset_name="TinyTitanic",
            problem_type="classification",
            target_column="Survived"
        )

        self.sample_data = pd.DataFrame([
            {"age": 22, "fare": 7.25, "Survived": 0},
            {"age": 38, "fare": 71.28, "Survived": 1}
        ]).to_dict()

        self.request = ModelSelectionRequest(
            metadata=self.metadata,
            selected_features=["age", "fare"],
            data=self.sample_data
        )

        self.mock_response = ModelSelectionResponse(
            model_name="RandomForest",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            reasoning="Selected for its robustness on tabular data."
        )

    def test_mocked_model_agent_response(self):
        from openai import OpenAI
        from instructor import from_openai

        client = from_openai(OpenAI(api_key="sk-FAKE-KEY"))

        with patch.object(client.chat.completions, 'create', return_value=self.mock_response) as mock_create:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful ML assistant."
                    },
                    {
                        "role": "user",
                        "content": f"Choose a model for: {self.request.model_dump_json()}"
                    }
                ],
                response_model=ModelSelectionResponse
            )

            self.assertEqual(response.model_name, ModelEnum.RANDOMFOREST)
            self.assertIn("n_estimators", response.hyperparameters)
            self.assertEqual(response.reasoning, "Selected for its robustness on tabular data.")
            mock_create.assert_called_once()
            print("✅ Mocked response:", response)
