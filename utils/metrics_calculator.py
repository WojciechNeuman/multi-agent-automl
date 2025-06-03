from typing import Dict, Any, Literal
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, root_mean_squared_error, mean_absolute_error, r2_score
)

def calculate_metrics(
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    problem_type: Literal["classification", "regression"],
    y_train_proba=None,
    y_test_proba=None
) -> Dict[str, Any]:
    """
    Calculates standard metrics for classification or regression.
    Returns a dictionary with train and test metrics.
    """
    metrics = {}

    if problem_type == "classification":
        metrics["train_accuracy"] = accuracy_score(y_train_true, y_train_pred)
        metrics["test_accuracy"] = accuracy_score(y_test_true, y_test_pred)
        metrics["train_precision"] = precision_score(y_train_true, y_train_pred, average="binary", zero_division=0)
        metrics["test_precision"] = precision_score(y_test_true, y_test_pred, average="binary", zero_division=0)
        metrics["train_recall"] = recall_score(y_train_true, y_train_pred, average="binary", zero_division=0)
        metrics["test_recall"] = recall_score(y_test_true, y_test_pred, average="binary", zero_division=0)
        metrics["train_f1_score"] = f1_score(y_train_true, y_train_pred, average="binary", zero_division=0)
        metrics["test_f1_score"] = f1_score(y_test_true, y_test_pred, average="binary", zero_division=0)
        if y_train_proba is not None and y_test_proba is not None:
            try:
                metrics["train_roc_auc"] = roc_auc_score(y_train_true, y_train_proba)
                metrics["test_roc_auc"] = roc_auc_score(y_test_true, y_test_proba)
            except Exception:
                pass
    elif problem_type == "regression":
        metrics["train_rmse"] = root_mean_squared_error(y_train_true, y_train_pred)
        metrics["test_rmse"] = root_mean_squared_error(y_test_true, y_test_pred)
        metrics["train_mae"] = mean_absolute_error(y_train_true, y_train_pred)
        metrics["test_mae"] = mean_absolute_error(y_test_true, y_test_pred)
        metrics["train_r2"] = r2_score(y_train_true, y_train_pred)
        metrics["test_r2"] = r2_score(y_test_true, y_test_pred)
    else:
        raise ValueError(f"Unknown problem_type: {problem_type}")

    return metrics