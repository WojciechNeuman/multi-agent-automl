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
from schemas.shared import LLMConfig

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

load_dotenv()
API_KEY = os.getenv("API_KEY")

# Configure loguru (basic console + rotating file)
logger.remove()
logger.add("logs/model_selection_agent.log", rotation="10 MB", retention="7 days", level="DEBUG")
logger.add(sys.stderr, level="INFO")

# --------------------------------------------------------------------------- #
# LLM helpers                                                                 #
# --------------------------------------------------------------------------- #
_SYSTEM_ROLE = (
    "You are a senior Model Selection Assistant. "
    "You will receive a compact description of a dataset's columns, "
    "including numeric, categorical, text, and datetime features. "
    "Select the best model and hyperparameters for a predictive model, aiming for "
    "accuracy and interpretability. "
)


def _build_prompt(req: ModelSelectionRequest) -> str:
    """
    Compose a compact, LLM-friendly prompt for model selection.
    """
    logger.info("Building the prompt for model selection.")

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

    return "\n".join(lines)

def _call_llm(req: ModelSelectionRequest, prompt: str) -> ModelSelectionResponse:
    """
    Call the LLM with the given prompt and return the response.
    """
    try:
        client = instructor.from_openai(OpenAI(api_key=API_KEY))
        response = client.chat.completions.create(
            model=req.llm_config.model,
            temperature=req.llm_config.temperature,
            max_tokens=req.llm_config.max_tokens,
            messages=[{"role": "system", "content": _SYSTEM_ROLE},
                      {"role": "user", "content": prompt}],
            response_model=ModelSelectionResponse
        )
        return response
    except ValidationError as e:
        logger.error(f"LLM response validation error: {e}")
        raise e
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise e

# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #
def run_model_agent(req: ModelSelectionRequest, preprocessing_code: str) -> Tuple[ModelSelectionResponse, str]:
    """
    Run the model selection agent to choose the best model and hyperparameters
    based on the provided dataset and selected features.
    """
    logger.info("Model Selection Agent started with default parameters.")
    start_time = time.time()
    logger.info(f"Processing request for dataset '{req.metadata.dataset_name}'")

    # 1. Build prompt
    prompt = _build_prompt(req)
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Prompt content:\n{prompt}")

    # 2. Call LLM``
    response = _call_llm(req, prompt)
    logger.info("LLM response received successfully")
    logger.debug(f"LLM response:\n{response}")

    # 3. Build pipeline
    try:
        pipe_blob = _build_training_pipeline(response, preprocessing_code)
        logger.info("Pipeline serialized to {len(pipe_blob)} bytes")
    except Exception as e:
        logger.error(f"Failed to build pipeline: {e}")
        raise e

    # 4. Return response
    elapsed = time.time() - start_time
    logger.info(f"Model Selection Agent finished after {elapsed:.2f} seconds. "
                f"Selected model: {response.model_name}, hyperparameters: {response.hyperparameters}")
    logger.info(f"Reasoning: {response.reasoning}")
    return response, pipe_blob

def _build_training_pipeline(response: ModelSelectionResponse, preprocessing_code: str) -> str:
    """
    Serialize the model selection response and preprocessing code into a Base64 string.
    This is a simplified example; in practice, you would use a library like `skops` to
    serialize the sklearn pipeline.
    """
    try:
        # Step 1: Decode and load the preprocessing pipeline
        preprocessing_blob = base64.b64decode(preprocessing_code)
        preprocessing_pipeline = sio.loads(preprocessing_blob)

        model = instantiate_model(response.model_name, response.hyperparameters)

        # Step 3: Build full sklearn pipeline
        full_pipeline = Pipeline([
            ("preprocessing", preprocessing_pipeline),
            ("model", model)
        ])

        # Step 4: Serialize and encode as base64 for transport
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
