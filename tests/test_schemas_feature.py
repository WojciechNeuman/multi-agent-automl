import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from unittest import TestCase
from pydantic import ValidationError
from schemas.feature import FeatureSelectionRequest
from schemas.shared import Metadata


class TestFeatureSelectionRequest(TestCase):
    def test_valid_feature_selection_request(self):
        request = FeatureSelectionRequest(
            metadata=Metadata(
                dataset_name="titanic",
                problem_type="classification",
                target_column="survived",
            ),
            data={"age": [22, 34], "survived": [0, 1]},
        )

        self.assertEqual(request.metadata.dataset_name, "titanic")
        self.assertEqual(request.metadata.problem_type, "classification")
        self.assertIn("age", request.data)

    def test_invalid_problem_type_in_metadata(self):
        with self.assertRaises(ValidationError) as context:
            Metadata(
                dataset_name="insurance",
                problem_type="clustering",
                target_column="charges"
            )
        self.assertIn("problem_type", str(context.exception))

    def test_missing_required_field(self):
        with self.assertRaises(ValidationError):
            FeatureSelectionRequest(
                metadata=Metadata(
                    dataset_name="demo",
                    problem_type="regression",
                    target_column="price"
                ),
                data=None
            )
