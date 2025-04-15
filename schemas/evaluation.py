from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .shared import Metadata

class EvaluationRequest(BaseModel):
    metadata: Metadata = Field(
        ..., description="Metadata about the ML task and dataset."
    )
    selected_features: List[str] = Field(
        ..., description="Features used by the trained model."
    )
    model_name: str = Field(
        ..., description="Name of the trained model to evaluate."
    )
    hyperparameters: Dict[str, Any] = Field(
        ..., description="Hyperparameters used for the selected model."
    )

class EvaluationResponse(BaseModel):
    reasoning: str = Field(
        ..., description="Summary of the model's performance and interpretation of the results."
    )
    metrics: Dict[str, float] = Field(
        ..., description="Computed evaluation metrics such as accuracy, F1 score, etc."
    )
    visualizations: Optional[Dict[str, str]] = Field(
        None, description="Optional visualizations encoded as base64 images or description strings."
    )
