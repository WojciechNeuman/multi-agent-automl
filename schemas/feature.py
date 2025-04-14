from typing import List, Optional
from pydantic import BaseModel, Field
from .shared import Metadata

class FeatureSelectionRequest(BaseModel):
    metadata: Metadata = Field(
        ..., description="Metadata including dataset name, task type, and target column."
    )
    data: dict = Field(
        ..., description="Serialized input dataset (e.g., result of DataFrame.to_dict())."
    )

class FeatureSelectionResponse(BaseModel):
    selected_features: List[str] = Field(
        ..., description="List of selected feature names deemed most relevant for the task."
    )
    reasoning: Optional[str] = Field(
        None, description="Optional explanation or reasoning behind the feature selection."
    )
