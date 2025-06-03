import os
import sys
import base64
import skops.io as sio
import pandas as pd
from typing import Dict, List
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from schemas.feature import FeatureSelectionRequest
from agents.feature_agent import run_feature_agent
from schemas.model_selection import ModelSelectionRequest
from agents.model_selection_agent import run_model_agent
from schemas.evaluation import EvaluationRequest
from agents.evaluation_agent import run_evaluation_agent, build_evaluation_conclusions
from utils.metrics_calculator import calculate_metrics

from sklearn.model_selection import train_test_split

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.info(" === Logging initialized for PipelineController. ===")
logger.add("logs/pipeline_controller_{time:YYYY-MM-DD_HH-mm-ss}.log", rotation="10 MB", retention="7 days", level="DEBUG")

class PipelineController:
    """
    Orchestrates the full AutoML pipeline using the agent modules.
    """

    def __init__(self, dataset_path: str, target_column: str, problem_type: str, main_metric: str, optimization_goal: str = "Avoid underfitting and overfitting.", max_features: int = 10, max_iterations: int = 5):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.problem_type = problem_type
        self.max_features = max_features
        self.df = pd.read_csv(dataset_path)
        self.selected_features = None
        self.features_preprocessing_code = None
        self.model_name = None
        self.model_hyperparams = None
        self.model_preprocessing_code = None
        self.pipeline = None
        self.metrics_history: List[Dict[str, float]] = []
        self.optimization_goal = optimization_goal
        self.max_iterations = max_iterations
        self.main_metric = main_metric
        self.models_results = []

    def run_feature_selection(self, evaluation_conclusions: str = None):
        sample = self.df.head(500).to_dict(orient='list')
        feature_req = FeatureSelectionRequest(
            metadata={
                "dataset_name": os.path.basename(self.dataset_path),
                "problem_type": self.problem_type,
                "target_column": self.target_column
            },
            basic_stats={},
            data_sample=sample,
            max_features=self.max_features,
            evaluation_conclusions=evaluation_conclusions
        )
        feature_resp = run_feature_agent(feature_req)
        self.selected_features = feature_resp.selected_features
        self.features_preprocessing_code = feature_resp.preprocessing_code

    def run_model_selection(self, evaluation_conclusions: str = None):
        sample = self.df.head(10).to_dict(orient='list')
        model_req = ModelSelectionRequest(
            metadata={
                "dataset_name": os.path.basename(self.dataset_path),
                "problem_type": self.problem_type,
                "target_column": self.target_column
            },
            selected_features=[f.name for f in self.selected_features],
            data=sample,
            evaluation_conclusions=evaluation_conclusions
        )
        model_resp = run_model_agent(model_req, self.features_preprocessing_code)
        self.model_name = model_resp[0].model_name
        self.model_hyperparams = model_resp[0].hyperparameters
        self.model_preprocessing_code = model_resp[1]

    def decode_pipeline(self, base64_blob):
        binary_blob = base64.b64decode(base64_blob)
        pipe = sio.loads(binary_blob)
        return pipe

    def train_and_evaluate(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.pipeline = self.decode_pipeline(self.model_preprocessing_code)
        self.pipeline.fit(X_train, y_train)
        y_train_pred = self.pipeline.predict(X_train)
        y_test_pred = self.pipeline.predict(X_test)
        current_metrics = calculate_metrics(
            y_train_true=y_train,
            y_train_pred=y_train_pred,
            y_test_true=y_test,
            y_test_pred=y_test_pred,
            problem_type=self.problem_type
        )
        self.metrics_history.append(current_metrics)
        return current_metrics

    def run_evaluation(self, current_metrics):
        eval_req = EvaluationRequest(
            metadata={
                "dataset_name": os.path.basename(self.dataset_path),
                "problem_type": self.problem_type,
                "target_column": self.target_column
            },
            selected_features=self.selected_features,
            model_name=self.model_name,
            hyperparameters=self.model_hyperparams
        )
        decision = run_evaluation_agent(
            request=eval_req,
            current_metrics=current_metrics,
            history=self.metrics_history[:-1],
            model_info={
                "model_name": self.model_name,
                "hyperparameters": self.model_hyperparams
            },
            optimization_goal=self.optimization_goal
        )
        return decision
    
    def is_better(self, new_value: float, best_value: float) -> bool:
        minimize_metrics = {"mae", "mse", "rmse"}
        maximize_metrics = {"accuracy", "f1", "precision", "recall", "r2"}

        metric = self.main_metric.lower()
        if metric in minimize_metrics:
            return new_value < best_value
        elif metric in maximize_metrics:
            return new_value > best_value
        else:
            logger.warning(f"Unknown metric direction for '{self.main_metric}', defaulting to maximize.")
            return new_value > best_value

    def run_full_pipeline(self):
        best_result = None
        best_metric = -float("inf")
        best_reasoning = ""
        best_model_info = None
        evaluation_conclusions = None
        logger.info("=== [PIPELINE STARTED] ===")
        logger.debug(f"Dataset: {self.dataset_path}")
        logger.debug(f"Target column: {self.target_column}")
        logger.debug(f"Problem type: {self.problem_type}")
        logger.debug(f"Max iterations: {self.max_iterations}")
        logger.debug(f"Main metric: {self.main_metric}")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"=== [ITERATION {iteration}] ===")

            self.run_feature_selection(evaluation_conclusions)

            self.run_model_selection(evaluation_conclusions)

            logger.info("[Training] Training and evaluating model...")
            current_metrics = self.train_and_evaluate()
            logger.info(f"[Training] Metrics: {current_metrics}")

            logger.info("[EvaluationAgent] Starting evaluation...")
            decision = self.run_evaluation(current_metrics)

            iteration_evaluation_conclusions = build_evaluation_conclusions(
                selected_features=self.selected_features,
                model_name=self.model_name,
                hyperparameters=self.model_hyperparams,
                evaluation_decision=decision,
                iteration=iteration
            )
            
            if evaluation_conclusions is None:
                evaluation_conclusions = iteration_evaluation_conclusions
            else:
                evaluation_conclusions += "\n\n" + iteration_evaluation_conclusions

            metric_value = current_metrics.get(f"test_{self.main_metric}", None)
            self.models_results.append({
                "metrics": current_metrics,
                "model_name": self.model_name,
                "hyperparameters": self.model_hyperparams,
                "features": [f.name for f in self.selected_features],
                "reasoning": decision.reasoning,
                "recommendation": decision.recommendation
            })

            if metric_value is not None and (best_result is None or self.is_better(metric_value, best_metric)):
                best_metric = metric_value
                best_result = self.models_results[-1]
                best_reasoning = decision.reasoning
                best_model_info = {
                    "model_name": self.model_name,
                    "hyperparameters": self.model_hyperparams,
                    "features": [f.name for f in self.selected_features]
                }

            if decision.recommendation == "stop":
                logger.info("Agent recommended to stop. Ending iterations.")
                break

        logger.info("=== [PIPELINE ENDED] ===")
        if best_result:
            logger.debug(f"Best model: {best_model_info['model_name']}")
            logger.debug(f"Hyperparameters: {best_model_info['hyperparameters']}")
            logger.debug(f"Features: {best_model_info['features']}")
            logger.debug(f"Test {self.main_metric}: {best_metric:.4f}")
            logger.debug(f"Reasoning: {best_reasoning}")
        else:
            logger.warning("No valid model found.")

        return best_result

def main():
    controller = PipelineController(
        dataset_path="../datasets/titanic.csv",
        target_column="Survived",
        problem_type="classification",
        max_iterations=3,
        main_metric="accuracy"
    )
    controller.run_full_pipeline()

if __name__ == "__main__":
    main()
