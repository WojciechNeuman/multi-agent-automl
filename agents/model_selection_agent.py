from __future__ import annotations

import os
import sys
import json
import base64
import instructor
from sklearn.pipeline import Pipeline
import skops.io as sio
from typing import Dict, Any, Tuple
from loguru import logger
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError
import time

from schemas.model_selection import (
    ModelSelectionRequest,
    ModelSelectionResponse,
    ModelEnum,
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

load_dotenv()
API_KEY = os.getenv("API_KEY")

logger.remove()

_SYSTEM_ROLE = (
    "You are a senior Model Selection Assistant. "
    "You will receive a compact description of a dataset's columns, "
    "including numeric, categorical, text, and datetime features, "
    "as well as previous evaluation results if available.\n\n"

    "Your goal is to choose the most suitable ML model and its hyperparameters "
    "for a predictive task, with emphasis on:\n"
    "- accuracy and generalization,\n"
    "- interpretability where relevant,\n"
    "- robustness to noise or missing data,\n"
    "- and potential synergies with the selected features.\n\n"

    "Important guidelines:\n"
    "- If past evaluation results are available, analyze them carefully.\n"
    "    - Identify which models performed best (e.g. highest test accuracy or lowest generalization gap).\n"
    "    - Prefer improving or fine-tuning models that already performed well.\n"
    "    - Consider whether simpler models (e.g. LogisticRegression) outperformed complex ones (e.g. GradientBoosting) — "
    "this may indicate overfitting in complex models or strong signal structure.\n"
    "- You are encouraged to **experiment** with different model types if justified by data shape or past results.\n"
    "- Also experiment with **non-default hyperparameters** that suit the dataset's size, sparsity, noise level, or feature count.\n"
    "- NEVER blindly default to a model — every decision must be **rational and explained** based on metadata and history.\n\n"
    "- NEVER choose the same model (model and hyperparameters) with the same features as before (information about previous evaluations is provided).\n"

    "Return your answer as a JSON object with the following fields:\n"
    "- model_name: one of ['RandomForest', 'LogisticRegression', 'LinearRegression', 'GradientBoosting', 'SVC', 'KNeighbors']\n"
    "- hyperparameters: a dictionary of hyperparameter names and values for the selected model\n"
    "- reasoning: a string (≤500 characters) explaining why this model and configuration were chosen. "
    "Your explanation must be concise, data-aware, and reflect performance history.\n\n"

    "Example:\n"
    "{\n"
    "  \"model_name\": \"GradientBoosting\",\n"
    "  \"hyperparameters\": {\"n_estimators\": 150, \"learning_rate\": 0.05},\n"
    "  \"reasoning\": \"GradientBoosting outperformed others in past iterations. Using more trees and lower learning rate to reduce overfitting.\"\n"
    "}"
)


def _build_prompt(req: ModelSelectionRequest) -> str:
    """
    Compose a compact, LLM-friendly prompt for model selection.
    """
    logger.debug("Building the prompt for model selection.")

    lines = [
        f"## Dataset Metadata",
        f"- Dataset Name: {req.metadata.dataset_name}",
        f"- Problem Type: {req.metadata.problem_type}",
        f"- Target Column: {req.metadata.target_column}",
        "",
        "## Selected Features",
    ]

    for col in req.selected_features:
        lines.append(f"- {col}")

    if req.evaluation_conclusions:
        lines.append(
            f"\n## (THE MOST IMPORTANT PART!!!) Previous Evaluation Conclusions: {req.evaluation_conclusions}"
        )

    return "\n".join(lines)

def _call_llm(req: ModelSelectionRequest, prompt: str, retries: int = 3) -> ModelSelectionResponse:
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
                response_model=ModelSelectionResponse,
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

def run_model_agent(req: ModelSelectionRequest, preprocessing_code: str) -> Tuple[ModelSelectionResponse, str]:
    """
    Run the model selection agent to choose the best model and hyperparameters
    based on the provided dataset and selected features.
    """
    logger.info("[ModelAgent] Starting model selection process.")
    start_time = time.time()

    prompt = _build_prompt(req)
    logger.debug(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Prompt content:\n{prompt}")

    response = _call_llm(req, prompt)
    logger.info("[ModelAgent] Model Agent received response from LLM successfully.")
    logger.debug(f"LLM response:\n{response}")

    try:
        pipe_blob = _build_training_pipeline(response, preprocessing_code)
        logger.debug("Pipeline serialized to {len(pipe_blob)} bytes")
    except Exception as e:
        logger.error(f"Failed to build pipeline: {e}")
        raise e

    elapsed = time.time() - start_time
    logger.info(f"[ModelAgent] Model Selection Agent finished after {elapsed:.2f} seconds. "
                f"Selected model: {response.model_name}, hyperparameters: {response.hyperparameters}")
    logger.info(f"[ModelAgent] Reasoning: {response.reasoning}")
    return response, pipe_blob

def _build_training_pipeline(response: ModelSelectionResponse, preprocessing_code: str) -> str:
    """
    Serialize the model selection response and preprocessing code into a Base64 string.
    This is a simplified example; in practice, you would use a library like `skops` to
    serialize the sklearn pipeline.
    """
    try:
        preprocessing_blob = base64.b64decode(preprocessing_code)
        preprocessing_pipeline = sio.loads(preprocessing_blob)

        model = instantiate_model(response.model_name, response.hyperparameters)

        full_pipeline = Pipeline([
            ("preprocessing", preprocessing_pipeline),
            ("model", model)
        ])

        serialized_pipeline = sio.dumps(full_pipeline)
        base64_pipeline = base64.b64encode(serialized_pipeline).decode("utf-8")

        return base64_pipeline

    except Exception as e:
        logger.error(f"Failed to serialize pipeline: {e}")
        raise e
    

MODEL_FACTORY = {
    ModelEnum.RANDOMFOREST: RandomForestClassifier,
    ModelEnum.LOGISTICREGRESSION: LogisticRegression,
    ModelEnum.LINEARREGRESSION: LinearRegression,
    ModelEnum.GRADIENTBOOSTING: GradientBoostingClassifier,
    ModelEnum.SVC: SVC,
    ModelEnum.KNEIGHBORS: KNeighborsClassifier,
}

def instantiate_model(model_enum: ModelEnum, hyperparams: Dict[str, Any]):
    """
    Given a ModelEnum and its hyperparameters, return an instantiated sklearn model.
    """
    if model_enum not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {model_enum}")
    
    model_class = MODEL_FACTORY[model_enum]
    return model_class(**hyperparams)
