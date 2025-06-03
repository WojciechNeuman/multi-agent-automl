import os
from typing import Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()
model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
model_temperature = float(os.getenv("MODEL_TEMPERATURE", 0.8))
model_max_tokens = int(os.getenv("MAX_TOKENS", 1024))

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

class LLMConfig(BaseModel):
    """Parameters controlling the underlying chat completion request."""

    model: str = Field(
        default=model_name, description="OpenAI-compatible model name."
    )
    temperature: float = Field(
        default=model_temperature, ge=0, le=2, description="Sampling temperature."
    )
    max_tokens: int = Field(
        default=model_max_tokens, ge=128, le=8192, description="Maximum tokens in the response."
    )
