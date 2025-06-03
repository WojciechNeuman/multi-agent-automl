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
import time

import numpy as np
import pandas as pd
import skops.io as sio
from loguru import logger
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

logger.remove()

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

_SYSTEM_ROLE = (
    "You are a senior Feature Engineering Assistant. "
    "You will receive a compact description of a dataset's columns, "
    "including numeric, categorical, text, and datetime features. "
    "Your goal is to select the most useful features for predictive modeling, balancing:\n"
    "- accuracy and generalization,\n"
    "- interpretability,\n"
    "- robustness to missing values or noise,\n"
    "- potential interactions between variables.\n\n"

    "⚠️ Important:\n"
    "- You are encouraged to **experiment** and sometimes include features that may not appear useful at first glance "
    "(e.g., weak correlation, high cardinality) if they might interact well with others.\n"
    "- Do not limit yourself to only obviously strong predictors — explore feature diversity and potential synergy.\n"
    "- However, always justify your selection clearly and concisely.\n\n"

    "Return your answer as a JSON object with the following fields:\n"
    "- selected_features: a list of objects, each with fields: "
    "name (str), dtype (one of 'numeric', 'categorical', 'text', 'datetime'), "
    "origin ('raw' or 'derived'), transformer (str), params (dict), importance (float or null)\n"
    "- preprocessing_code: a string (base64-encoded pipeline)\n"
    "- reasoning: a string (≤500 words, summarizing your rationale and why the features were selected)\n\n"

    "Example:\n"
    "{\n"
    "  \"selected_features\": [\n"
    "    {\"name\": \"age\", \"dtype\": \"numeric\", \"origin\": \"raw\", \"transformer\": \"none\", \"params\": {}, \"importance\": 0.42},\n"
    "    {\"name\": \"sex\", \"dtype\": \"categorical\", \"origin\": \"raw\", \"transformer\": \"onehot\", \"params\": {}, \"importance\": 0.31},\n"
    "    {\"name\": \"country\", \"dtype\": \"categorical\", \"origin\": \"raw\", \"transformer\": \"onehot\", \"params\": {}, \"importance\": 0.11}\n"
    "  ],\n"
    "  \"preprocessing_code\": \"<base64 string>\",\n"
    "  \"reasoning\": \"Selected based on mutual information and data diversity. While 'country' has weak direct correlation, it may provide interaction signals with age and income.\"\n"
    "}"
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
        line = f"- {col}: type={meta.dtype}, miss={meta.missing_pct:.1%}, card={meta.cardinality}"
        
        if meta.dtype == "numeric":
            line += (
                f", mean={meta.mean:.2f}, median={meta.median:.2f}, "
                f"std={meta.std:.2f}, skew={meta.skewness:.2f}, kurt={meta.kurtosis:.2f}, "
                f"min={meta.min_val:.2f}, max={meta.max_val:.2f}"
            )
        
        elif meta.dtype == "categorical":
            line += (
                f", top_freq={meta.top_freq:.1%}, rare_pct={meta.rare_pct:.1%}"
            )
        
        elif meta.dtype == "text":
            line += (
                f", avg_length={meta.avg_length:.1f}"
            )
            if meta.lang_detected:
                line += f", lang={meta.lang_detected}"
        
        elif meta.dtype == "datetime":
            if meta.span_days is not None:
                line += f", span_days={meta.span_days}"

        if meta.corr_target is not None:
            line += f", corr≈{meta.corr_target:.2f}"
        
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

    if req.evaluation_conclusions:
        lines.append(
            f"\n### (THE MOST IMPORTANT PART!!!) Previous evaluation conclusions: {req.evaluation_conclusions}"
        )

    return "\n".join(lines)

def _call_llm(req: FeatureSelectionRequest, prompt: str, retries: int = 3) -> FeatureSelectionResponse:
    """
    Call the LLM with the given prompt and return the response.
    Retries up to `retries` times on ValidationError or Exception.
    """
    for attempt in range(1, retries + 1):
        try:
            client = instructor.from_openai(OpenAI(api_key=API_KEY))
            response = client.chat.completions.create(
                model=req.llm_config.model,
                temperature=req.llm_config.temperature,
                max_tokens=req.llm_config.max_tokens,
                messages=[{"role": "system", "content": _SYSTEM_ROLE},
                          {"role": "user", "content": prompt}],
                response_model=FeatureSelectionResponse,
                max_retries=3
            )
            return response
        except ValidationError as e:
            logger.error(f"LLM response validation error (attempt {attempt+1}/3): {e}")
            if attempt == retries:
                raise
        except Exception as e:
            logger.error(f"LLM call failed (attempt {attempt+1}/3): {e}")
            if attempt == retries:
                raise

def run_feature_agent(req: FeatureSelectionRequest) -> FeatureSelectionResponse:
    logger.info("[FeatureAgent] Starting feature selection process")
    logger.info(
        "[FeatureAgent] Feature Agent will try to choose the best possible parameters, it will choose "
        f"maximum {req.max_features} features. Its goal is: {getattr(req, 'selection_goal', None)}"
    )
    start_time = time.time()

    logger.info(f"[FeatureAgent] Processing request for dataset '{req.metadata.dataset_name.split('_')[-1]}'")

    df = pd.DataFrame.from_dict(req.data_sample)
    df = _drop_constant_cols(df)
    logger.debug(f"Dataframe shape after constant column drop: {df.shape}")

    if not req.basic_stats:
        logger.debug("Computing basic stats...")
        stats = compute_basic_stats(df, target_name=req.metadata.target_column)
        logger.debug(f"Computed stats: {len(stats)} features")
    else:
        stats = req.basic_stats

    recommended = _mutual_info_topk(
        df, req.metadata.target_column, req.metadata.problem_type, k=10
    )
    logger.debug(f"Top MI-recommended features: {recommended}")

    prompt = _build_prompt(req, stats, recommended)
    logger.debug(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Prompt content:\n{prompt}")

    response = _call_llm(req, prompt)
    logger.info("[FeatureAgent] Feature Agent received response successfully")
    logger.debug(f"LLM response:\n{response}")

    try:
        pipe_blob = _build_pipeline_blob(df, response.selected_features)
        logger.debug(f"Pipeline serialized to {len(pipe_blob)} bytes")
    except Exception as e:
        logger.exception(f"Pipeline serialization failed with error: {e}")
        raise

    elapsed = time.time() - start_time
    logger.info(f"[FeatureAgent] Feature Agent finished after {elapsed:.2f} seconds. "
                f"Selected features: {[f.name for f in response.selected_features]}")
    logger.info(f"[FeatureAgent] Feature Agent chose the above-mentioned features because: {response.reasoning}")
    return FeatureSelectionResponse(
        selected_features=response.selected_features,
        preprocessing_code=pipe_blob,
        reasoning=response.reasoning,
    )

def _build_pipeline_blob(df: pd.DataFrame, specs: List[FeatureSpec]) -> str:
    try:
        num_cols = [s.name for s in specs if s.dtype == "numeric" and s.name in df.columns]
        cat_cols = [s.name for s in specs if s.dtype == "categorical" and s.name in df.columns]

        logger.debug(f"Numeric features: {num_cols}")
        logger.debug(f"Categorical features: {cat_cols}")

        num_pipe = Pipeline(
            steps=[("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        cat_pipe = Pipeline(
            steps=[("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
        )

        pipe = ColumnTransformer(
            transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)],
            remainder="drop",
        )

        binary_blob = sio.dumps(pipe)
        base64_blob = base64.b64encode(binary_blob).decode("ascii")
        logger.debug(f"Pipeline blob generated with {len(binary_blob)} bytes")
        return base64_blob

    except Exception as e:
        logger.exception(f"Failed to build pipeline blob with error: {e}")
        raise
