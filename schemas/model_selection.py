from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from schemas.shared import LLMConfig, Metadata
from models.model import ModelEnum

class ModelSelectionRequest(BaseModel):
    metadata: Metadata = Field(
        ..., description="Metadata including dataset name, task type, and target column."
    )
    selected_features: List[str] = Field(
        ..., description="List of features selected by the FeatureAgent."
    )
    data: dict = Field(
        ..., description="Serialized dataset to be used for model selection and training."
    )
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="Parameters used for the chat call."
    )

class ModelSelectionResponse(BaseModel):
    model_name: ModelEnum = Field(
        ..., description="The name of the selected ML model (e.g., 'RandomForest', 'XGBoost')."
    )
    hyperparameters: Dict[str, Any] = Field(
        ..., description="Hyperparameter configuration for the selected model."
    )
    reasoning: Optional[str] = Field(
        None, description="Explanation for why this model and configuration were chosen."
    )
