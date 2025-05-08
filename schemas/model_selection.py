from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from schemas.shared import LLMConfig, Metadata

class ModelSelectionRequest(BaseModel):
    metadata: Metadata = Field(
        ..., description="Metadata including dataset name, task type, and target column."
    )
    data_sample: Dict[str, list] = Field(
        ...,
        description="Small row sample (e.g. df.head().to_dict()) - never the full dataset.",
    )
    selected_features: List[str] = Field(
        ..., description="List of features selected by the FeatureAgent."
    )
    feature_pipeline: str = Field(
        ...,
        description="Serialized pipeline code (e.g., from FeatureAgent) to be used for model training.",
    )

    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="Parameters used for the chat call."
    )

class ModelSelectionResponse(BaseModel):
    model_name: str = Field(
        ..., description="The name of the selected ML model (e.g., 'RandomForest', 'XGBoost')."
    )
    model_declaration: str = Field(
        ...,
        description="Python code to instantiate the model with the chosen hyperparameters."
    )
    reasoning: Optional[str] = Field(
        ...,
        max_length=512,
        description="≤512-word Explanation for why this model and configuration were chosen."
    )
