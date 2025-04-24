from typing import List, Optional, Literal, Dict, Any
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

class FeatureOverview(BaseModel):
    dtype: Literal["numeric", "categorical", "text", "datetime"] = Field(
        ...,
        description="Logical data type of the feature after preprocessing."
    )
    missing_pct: float = Field(
        ...,
        description="Percentage of missing values in the feature."
    )
    cardinality: int = Field(
        ...,
        description="Number of unique values in the feature."
    )
    skewness: Optional[float] = Field(
        ...,
        description="Skewness of the feature distribution. (if applicable)"
    )
    mean: Optional[float] = Field(
        ...,
        description="Mean of the feature values. (if applicable)"
    )
    median: Optional[float] = Field(
        ...,
        description="Median of the feature values. (if applicable)"
    )
    std: Optional[float] = Field(
        ...,
        description="Standard deviation of the feature values. (if applicable)"
    )

class FeatureSelectionRequest(BaseModel):
    metadata: Metadata
    basic_stats: Dict[str, FeatureOverview]
    data_sample: Dict[str, list] = Field(
        ...,
        description="Sample of the input dataset (e.g., result of DataFrame.head().to_dict())."
    )

class FeatureSpec(BaseModel):
    name: str = Field(
        ...,
        description="Original column name or alias of the feature as it will appear in the ML pipeline.",
    )
    dtype: Literal["numeric", "categorical", "text", "datetime"]= Field(
        ...,
        description="Logical data type after preprocessing (e.g. 'numeric', 'categorical', 'text', 'datetime').",
    )
    origin: Literal["raw", "derived"] = Field(
        ...,
        description="Indicates whether the feature comes directly from the source dataset ('raw') or was generated during preprocessing ('derived').",
    )
    transformer: str = Field(
        ...,
        description="Named step or dotted-path reference to the transformer inside the preprocessing Pipeline that produces this feature.",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary of key hyper-parameters for the transformer/generator that produced the feature.",
    )
    importance: Optional[float] = Field(
        None,
        description="Optional numerical estimate of the feature's predictive importance (e.g. mutual information, gain, permutation score).",
    )


class FeatureSelectionResponse(BaseModel):
    selected_features: List[FeatureSpec] = Field(
        ...,
        description="Ordered collection of FeatureSpec objects describing the final feature set provided to ModelAgent.",
    )
    preprocessing_code: str = Field(
        ...,
        description="Base64-encoded or UTF-8 string produced by `skops.io.dumps` that reconstructs the preprocessing sklearn.Pipeline when loaded.",
    )
    reasoning: str = Field(
        ...,
        description="Short natural-language justification (≤ 100 words) summarising why these features were selected.",
        max_length=600,
    )
