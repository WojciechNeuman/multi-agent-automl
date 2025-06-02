from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from schemas.shared import LLMConfig

from models.feature import FeatureSpec

class Metadata(BaseModel):
    """Minimal information required to identify the dataset and task."""

    dataset_name: str = Field(..., description="Human-readable dataset identifier.")
    problem_type: Literal["classification", "regression"] = Field(
        ..., description="Down-stream ML problem type."
    )
    target_column: str = Field(..., description="Name of the target/label column.")
    additional_notes: Optional[str] = Field(
        None, description="Optional user-supplied context or constraints."
    )


class FeatureOverview(BaseModel):
    """
    Compact representation of per-column statistics.
    Extended with skewness, kurtosis, min/max, and top frequency.
    """

    dtype: Literal["numeric", "categorical", "text", "datetime"]
    missing_pct: float
    cardinality: int

    # Numeric-only
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    corr_target: Optional[float] = None

    # Categorical-only
    top_freq: Optional[float] = None
    rare_pct: Optional[float] = None

    # Text-only
    avg_length: Optional[float] = None
    lang_detected: Optional[str] = None

    # Date-time only
    span_days: Optional[int] = None


class FeatureSelectionRequest(BaseModel):
    """Input consumed by FeatureAgent."""

    metadata: Metadata
    basic_stats: Dict[str, FeatureOverview]
    data_sample: Dict[str, list] = Field(
        ...,
        description="Small row sample (e.g. df.head().to_dict()) - never the full dataset.",
    )
    selection_goal: str = Field(
        default="maximise predictive power with the fewest features",
        description="Natural-language description of what the LLM should optimise for.",
    )
    max_features: int = Field(
        default=30,
        ge=1,
        description="Upper bound on the number of features the LLM may return.",
    )
    llm_config: LLMConfig = Field(
        default_factory=LLMConfig, description="Parameters used for the chat call."
    )
    evaluation_conclusions: Optional[str] = Field(
        None,
        description=(
            "Optional summary of previous evaluation results, if available. "
            "Helps the LLM avoid repeating past mistakes."
        ),
    )

class FeatureSelectionResponse(BaseModel):
    """Output returned by FeatureAgent."""

    selected_features: List[FeatureSpec]
    preprocessing_code: str = Field(
        ...,
        description=(
            "UTF-8 / Base64 string produced by `skops.io.dumps`,"
            " sufficient to rebuild the sklearn.Pipeline."
        ),
    )
    reasoning: str = Field(
        ...,
        max_length=1000,
        description="Rationale summarised from the LLM answer.",
    )
