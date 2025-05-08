"""
FeatureAgent - extended.
* Filters zero-variance columns before LLM.
* Adds optional Mutual-Information ranking (top_k_for_llm).
* Prompt now includes - missing %, cardinality, corr_target (if present).
* Still keeps the final choice with the LLM.
"""
from __future__ import annotations

import json
from typing import List
import os
import instructor
import base64

import numpy as np
import pandas as pd
import skops.io as sio
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schemas.feature import (
    FeatureSelectionRequest,
    FeatureSelectionResponse,
    FeatureSpec,
)
from utils.mini_eda import compute_basic_stats

load_dotenv()
API_KEY = os.getenv("API_KEY")

# --------------------------------------------------------------------------- #
# Pre-filters                                                                  #
# --------------------------------------------------------------------------- #
def _drop_constant_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with a single unique value (after NA drop)."""
    return df.loc[:, df.nunique(dropna=True) > 1]


def _mutual_info_topk(
    df: pd.DataFrame, target_col: str, problem: str, k: int = 15
) -> list[str]:
    """Return names of k features with highest MI to the target (quick heuristic)."""
    x = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_mask = x.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    x_num = x.loc[:, numeric_mask]
    if x_num.empty:
        return []

    func = mutual_info_classif if problem == "classification" else mutual_info_regression
    mi = func(x_num.fillna(0), y)
    top_idx = np.argsort(mi)[::-1][:k]
    return x_num.columns[top_idx].tolist()


# --------------------------------------------------------------------------- #
# LLM helpers                                                                  #
# --------------------------------------------------------------------------- #
_SYSTEM_ROLE = (
    "You are a senior Feature Engineering Assistant. "
    "You will receive a compact description of a dataset's columns, "
    "including numeric, categorical, text, and datetime features. "
    "Select the best features for a predictive model, aiming for "
    "accuracy and interpretability. "
)

def _build_prompt(
    req: FeatureSelectionRequest,
    stats: dict,
    recommended: list[str],
) -> str:
    """
    Compose a compact, LLM-friendly prompt with extended stats.
    Now includes numeric, categorical, text, and datetime-specific fields.
    """
    lines = [
        f"Dataset: {req.metadata.dataset_name}",
        f"Problem type: {req.metadata.problem_type}",
        f"Target: {req.metadata.target_column}",
        f"Goal: {req.selection_goal}",
        f"Max features: {req.max_features}",
        "",
        "### Candidate features",
    ]

    for col, meta in stats.items():
        # Shared fields
        line = f"- {col}: type={meta.dtype}, miss={meta.missing_pct:.1%}, card={meta.cardinality}"
        
        # Numeric extensions
        if meta.dtype == "numeric":
            line += (
                f", mean={meta.mean:.2f}, median={meta.median:.2f}, "
                f"std={meta.std:.2f}, skew={meta.skewness:.2f}, kurt={meta.kurtosis:.2f}, "
                f"min={meta.min_val:.2f}, max={meta.max_val:.2f}"
            )
        
        # Categorical extensions
        elif meta.dtype == "categorical":
            line += (
                f", top_freq={meta.top_freq:.1%}, rare_pct={meta.rare_pct:.1%}"
            )
        
        # Text extensions
        elif meta.dtype == "text":
            line += (
                f", avg_length={meta.avg_length:.1f}"
            )
            if meta.lang_detected:
                line += f", lang={meta.lang_detected}"
        
        # Date-time extensions
        elif meta.dtype == "datetime":
            if meta.span_days is not None:
                line += f", span_days={meta.span_days}"

        # Correlation to target (if available)
        if meta.corr_target is not None:
            line += f", corr≈{meta.corr_target:.2f}"
        
        # Add to output
        lines.append(line)

    df_full = pd.DataFrame.from_dict(req.data_sample)

    if df_full.shape[0] > 5:
        df_full = df_full.sample(5, random_state=42).reset_index(drop=True)

    sample_json_lines = [json.dumps(row) for row in df_full.to_dict(orient="records")]
    lines.append("\n### Sample data (5 rows):")
    lines.extend(sample_json_lines)

    if recommended:
        lines.append(
            f"\n### Recommended by Mutual-Information: {', '.join(recommended)}"
        )

    return "\n".join(lines)



def _call_llm(prompt: str, cfg) -> str:
    client = instructor.from_openai(OpenAI(api_key=API_KEY))    
    rsp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[{"role": "system", "content": _SYSTEM_ROLE},
                  {"role": "user", "content": prompt}],
        response_model=FeatureSelectionResponse
    )
    return rsp


def _parse_llm(raw: str) -> list[FeatureSpec]:
    try:
        return [FeatureSpec(**d) for d in json.loads(raw)]
    except (json.JSONDecodeError, ValidationError) as err:  # noqa: TRY003
        raise ValueError("LLM response could not be parsed into FeatureSpec list.") from err


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def run_feature_agent(req: FeatureSelectionRequest) -> FeatureSelectionResponse:
    # 1. Load and pre-filter sample
    df = pd.DataFrame.from_dict(req.data_sample)
    df = _drop_constant_cols(df)

    # 2. Stats (compute if missing)
    stats = req.basic_stats or compute_basic_stats(
        df, target_name=req.metadata.target_column
    )

    # 3. Recommend top-MI features (quick heuristic – can be empty)
    recommended = _mutual_info_topk(
        df, req.metadata.target_column, req.metadata.problem_type, k=10
    )

    # 4. Build prompt + call LLM
    prompt = _build_prompt(req, stats, recommended)

    print("LLM prompt:\n", prompt, "\n")

    response = _call_llm(prompt, req.llm_config)

    print("LLM response:\n", response, "\n")
    print(type(response))

    # 5. Build pipeline (simple baseline)
    pipe_blob = _build_pipeline_blob(df, response.selected_features)

    print("Pipeline blob:\n", pipe_blob, "\n")

    # 6. Done
    return FeatureSelectionResponse(
        selected_features=response.selected_features,
        preprocessing_code=pipe_blob,
        reasoning=response.reasoning,
    )


# --------------------------------------------------------------------------- #
# Helpers – pipeline, reasoning                                               #
# --------------------------------------------------------------------------- #
def _build_pipeline_blob(df: pd.DataFrame, specs: List[FeatureSpec]) -> str:
    num_cols = [s.name for s in specs if s.dtype == "numeric" and s.name in df.columns]
    cat_cols = [s.name for s in specs if s.dtype == "categorical" and s.name in df.columns]

    num_pipe = Pipeline(
        steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    cat_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    pipe = ColumnTransformer(
        transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
        remainder="drop",
    )

    # Serialize as a binary blob, then base64-encode for safe JSON transport
    binary_blob = sio.dumps(pipe)
    base64_blob = base64.b64encode(binary_blob).decode("ascii")
    return base64_blob

def _trim_reasoning(raw: str, limit: int = 600) -> str:
    return raw if len(raw) <= limit else raw[: limit - 3] + "..."
