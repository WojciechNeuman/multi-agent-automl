from __future__ import annotations

from typing import List, Dict, Any
from pydantic import ValidationError
import os
import sys
import instructor
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
import time

from schemas.evaluation import EvaluationRequest, EvaluationDecision
from models.feature import FeatureSpec

load_dotenv()
API_KEY = os.getenv("API_KEY")

logger.remove()
logger.add("logs/evaluation_agent.log", rotation="10 MB", retention="7 days", level="DEBUG")

_SYSTEM_ROLE = (
    "You are an advanced AutoML Evaluation Agent. "
    "You receive the current model's metrics, history of previous results, "
    "task type (classification or regression), model details, selected features, "
    "and optimization goal. "
    "Analyze the situation: check if the model is improving, stagnating, or degrading, "
    "look for signs of overfitting/underfitting, assess feature quality, "
    "and understand how changes affected results. "
    "Return your answer as a JSON object with the following fields:\n"
    "- recommendation: one of ['continue', 'switch_model', 'switch_features', 'stop']\n"
    "- reasoning: a string (max 500 characters) explaining your decision\n"
    "- confidence: a number between 0 and 1 (optional, can be null)\n"
    "Example:\n"
    "{\n"
    "  \"recommendation\": \"continue\",\n"
    "  \"reasoning\": \"Model performance is improving over the last 2 iterations. No signs of overfitting detected.\",\n"
    "  \"confidence\": 0.87\n"
    "}\n"
)

class LLMRunContext:
    def __init__(
        self,
        current_metrics: Dict[str, float],
        history: List[Dict[str, float]],
        task_type: str,
        model_info: Dict[str, Any],
        selected_features: List[FeatureSpec],
        optimization_goal: str
    ):
        self.current_metrics = current_metrics
        self.history = history
        self.task_type = task_type
        self.model_info = model_info
        self.selected_features = selected_features
        self.optimization_goal = optimization_goal

def _build_prompt(ctx: LLMRunContext) -> str:
    lines = [
        "## Modeling Context",
        f"- Task type: {ctx.task_type}",
        f"- Optimization goal: {ctx.optimization_goal}",
        "",
        "## Model Info",
        *(f"- {k}: {v}" for k, v in ctx.model_info.items()),
        "",
        "## Selected Features",
        *(f"- {f.name} ({f.dtype})" for f in ctx.selected_features),
        "",
        "## Current Metrics",
        *(f"- {k}: {v:.4f}" for k, v in ctx.current_metrics.items()),
        "",
        "## History (previous iterations)",
    ]
    if ctx.history:
        for i, metrics in enumerate(ctx.history, 1):
            hist_line = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            lines.append(f"- Iteration {i}: {hist_line}")
    else:
        lines.append("- No previous iterations.")
    return "\n".join(lines)

def _call_llm(prompt: str, retries: int = 3) -> EvaluationDecision:
    """
    Call the LLM with the given prompt and return the response.
    Retries up to 3 times on ValidationError or Exception.
    """
    for attempt in range(1, retries + 1):
        try:
            client = instructor.from_openai(OpenAI(api_key=API_KEY))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=512,
                messages=[
                    {"role": "system", "content": _SYSTEM_ROLE},
                    {"role": "user", "content": prompt}
                ],
                response_model=EvaluationDecision,
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

def run_evaluation_agent(
    request: EvaluationRequest,
    current_metrics: Dict[str, float],
    history: List[Dict[str, float]],
    model_info: Dict[str, Any],
    optimization_goal: str
) -> EvaluationDecision:
    """
    Main entry point for the EvaluationAgent.
    Builds the context, sends it to the LLM, and returns the decision.
    """
    logger.info("Evaluation Agent started processing.")
    logger.info(f"Running evaluation agent for dataset '{request.metadata.dataset_name}'")
    start_time = time.time()

    ctx = LLMRunContext(
        current_metrics=current_metrics,
        history=history,
        task_type=request.metadata.problem_type,
        model_info=model_info,
        selected_features=request.selected_features,
        optimization_goal=optimization_goal
    )
    prompt = _build_prompt(ctx)
    logger.info(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Prompt content:\n{prompt}")

    decision = _call_llm(prompt)
    elapsed = time.time() - start_time
    logger.info(f"Evaluation Agent finished after {elapsed:.2f} seconds. "
                f"Recommendation: {decision.recommendation}")
    logger.info(f"Reasoning: {decision.reasoning}")

    return decision

def build_evaluation_conclusions(
    selected_features: List[FeatureSpec],
    model_name: str,
    hyperparameters: Dict[str, Any],
    evaluation_decision: EvaluationDecision
) -> str:
    features_str = ", ".join([f.name for f in selected_features])
    hyperparams_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
    summary = (
        f"Features selected: {features_str}\n"
        f"Model: {model_name}\n"
        f"Hyperparameters: {hyperparams_str}\n"
        f"Evaluation conclusion: {evaluation_decision.reasoning}\n"
        f"Recommendation: {evaluation_decision.recommendation}\n"
        f"Confidence: {evaluation_decision.confidence}"
    )
    return summary
