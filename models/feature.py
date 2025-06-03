from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal, Any

class FeatureSpec(BaseModel):
    """Full lineage of a single feature after selection / engineering."""

    name: str
    dtype: Literal["numeric", "categorical", "text", "datetime"]
    origin: Literal["raw", "derived"]
    transformer: str
    params: Dict[str, Any] = Field(default_factory=dict)
    importance: Optional[float] = None
    