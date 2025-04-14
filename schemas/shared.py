from typing import Literal, Optional
from pydantic import BaseModel, Field

class Metadata(BaseModel):
    dataset_name: str = Field(
        ..., description="The name of the dataset being processed."
    )
    problem_type: Literal["classification", "regression"] = Field(
        ..., description="The type of ML task (classification or regression)."
    )
    target_column: str = Field(
        ..., description="The name of the target variable for prediction."
    )
    additional_notes: Optional[str] = Field(
        None, description="Optional notes or comments about the dataset or task."
    )
