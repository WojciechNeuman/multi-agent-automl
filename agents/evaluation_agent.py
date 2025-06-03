from __future__ import annotations

from typing import List, Dict, Any
from pydantic import ValidationError
import os
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

_SYSTEM_ROLE = (
    "You are a critical AutoML Evaluation Agent responsible for iterative improvement "
    "of ML pipelines. You will receive:\n"
    "- current model metrics (train/test),\n"
    "- history of previous test results,\n"
    "- task type (classification or regression),\n"
    "- selected features and model info,\n"
    "- optimization goal (e.g. maximize recall, avoid overfitting).\n\n"

    "You must critically analyze the situation using **relative** and **absolute** evaluation. "
    "The following expectations apply:\n"
    "1. Treat any significant gap between train and test performance — regardless of absolute values — as a sign of overfitting. "
    "For example, train = 0.07 and test = 0.04 **is overfitting**, as is train = 0.95 and test = 0.85.\n"
    "2. Always be skeptical of `continue`. Recommend it **only** when:\n"
    "   - performance is clearly improving **and**\n"
    "   - the train/test gap is stable and ≤ 0.10 **and**\n"
    "   - current test performance is above average (e.g. ≥ 0.75).\n"
    "3. If performance is poor (e.g. F1, accuracy, recall < 0.70 or R² < 0.4), and no major overfitting is present, suggest trying **more complex models**.\n"
    "4. If overfitting is consistent (train much higher than test), suggest `switch_model` or `switch_features`.\n"
    "5. If the model has achieved consistently high performance (e.g. test ≥ 0.85) and shows diminishing improvements across iterations, `stop` may be justified.\n"
    "6. Do not default to the safest or simplest option. Favor decisive, justified recommendations. `continue` is acceptable **only when strongly warranted**.\n\n"

    "Metric direction assumptions:\n"
    "- For classification: maximize accuracy, F1, recall, precision (higher is better)\n"
    "- For regression: maximize R²; minimize MAE, RMSE (lower is better)\n\n"

    "Your answer must be a JSON object with the following fields:\n"
    "- recommendation: one of ['continue', 'switch_model', 'switch_features', 'stop']\n"
    "- reasoning: a string (max 1000 characters) justifying your decision\n"
    "- confidence: a number between 0 and 1 (optional, can be null)\n\n"

    "Example output:\n"
    "{\n"
    "  \"recommendation\": \"switch_model\",\n"
    "  \"reasoning\": \"Test accuracy is low (0.64) and train-test gap (~0.3) suggests overfitting. A more robust model is needed.\",\n"
    "  \"confidence\": 0.85\n"
    "}"
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

def _call_llm(req: EvaluationRequest, prompt: str, retries: int = 3) -> EvaluationDecision:
    """
    Call the LLM with the given prompt and return the response.
    Retries up to 3 times on ValidationError or Exception.
    """
    for attempt in range(1, retries + 1):
        try:
            client = instructor.from_openai(OpenAI(api_key=API_KEY))
            response = client.chat.completions.create(
                model=req.llm_config.model,
                temperature=req.llm_config.temperature,
                max_tokens=req.llm_config.max_tokens,
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
    logger.info("[EvaluationAgent] Starting evaluation process.")
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
    logger.debug(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Prompt content:\n{prompt}")

    decision = _call_llm(request, prompt)
    logger.info('[EvaluationAgent] Evaluation Agent received response successfully.')
    logger.debug(f"LLM response:\n{decision}")
    elapsed = time.time() - start_time
    logger.info(f"[EvaluationAgent] Evaluation Agent finished after {elapsed:.2f} seconds. "
                f"Recommendation: {decision.recommendation}")
    logger.info(f"[EvaluationAgent] Reasoning: {decision.reasoning}")

    return decision

def build_evaluation_conclusions(
    selected_features: List[FeatureSpec],
    model_name: str,
    hyperparameters: Dict[str, Any],
    evaluation_decision: EvaluationDecision,
    iteration: int = None
) -> str:
    features_str = ", ".join([f.name for f in selected_features])
    hyperparams_str = ", ".join(f"{k}={v}" for k, v in hyperparameters.items())
    if iteration is not None:
        iteration_str = f"Iteration {iteration} conclusions:\n"
    else:
        iteration_str = ""
    summary = (
        f"{iteration_str}"
        f"Features selected: {features_str}\n"
        f"Model: {model_name}\n"
        f"Hyperparameters: {hyperparams_str}\n"
        f"Evaluation conclusion: {evaluation_decision.reasoning}\n"
        f"Recommendation: {evaluation_decision.recommendation}\n"
        f"Confidence: {evaluation_decision.confidence}"
    )
    return summary
