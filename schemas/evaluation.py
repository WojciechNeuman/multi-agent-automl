from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from .shared import Metadata
from models.model import ModelEnum
from models.feature import FeatureSpec
from schemas.shared import LLMConfig

class EvaluationRequest(BaseModel):
    metadata: Metadata = Field(
        ..., description="Metadata about the ML task and dataset."
    )
    selected_features: List[FeatureSpec] = Field(
        ..., description="List of features selected for model evaluation."
    )
    model_name: ModelEnum = Field(
        ..., description="The name of the selected ML model (e.g., 'RandomForest', 'XGBoost')."
    )
    hyperparameters: Dict[str, Any] = Field(
        ..., description="Hyperparameter configuration for the selected model."
    )
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="Parameters used for the chat call."
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

class EvaluationDecision(BaseModel):
    recommendation: Literal['continue', 'switch_model', 'switch_features', 'stop']
    reasoning: str = Field(..., max_length=500)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    