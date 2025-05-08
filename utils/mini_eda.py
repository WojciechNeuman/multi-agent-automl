from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from schemas.feature import FeatureOverview


def _quick_corr(series: pd.Series, target: pd.Series) -> float | None:
    """Fast absolute correlation against a numeric / binary target."""
    try:
        if pd.api.types.is_numeric_dtype(series) and pd.api.types.is_numeric_dtype(
            target
        ):
            return float(abs(series.corr(target)))
        if series.dtype == "object" and target.nunique() <= 2:
            le_s = LabelEncoder().fit_transform(series.astype(str))
            le_t = LabelEncoder().fit_transform(target.astype(str))
            return float(abs(np.corrcoef(le_s, le_t)[0, 1]))
    except Exception:
        pass
    return None


def compute_basic_stats(
    df: pd.DataFrame, target_name: str | None = None
) -> Dict[str, FeatureOverview]:
    """
    Extracts rich feature statistics for LLM prompt building.
    Handles numeric, categorical, text, and datetime columns.
    """
    stats = {}
    target = df[target_name] if target_name in df.columns else None

    for col in df.columns:
        s = df[col]
        dtype = (
            "numeric"
            if pd.api.types.is_numeric_dtype(s)
            else "datetime"
            if pd.api.types.is_datetime64_any_dtype(s)
            else "text"
            if pd.api.types.is_string_dtype(s) and s.str.len().mean() > 40
            else "categorical"
        )

        # Shared stats
        kw = dict(
            dtype=dtype,
            missing_pct=float(s.isna().mean()),
            cardinality=int(s.nunique(dropna=True)),
        )

        # Numeric extensions
        if dtype == "numeric":
            kw.update(
                mean=float(s.mean()),
                median=float(s.median()),
                std=float(s.std()),
                skewness=float(s.skew()),
                kurtosis=float(s.kurtosis()),
                min_val=float(s.min()),
                max_val=float(s.max()),
            )

        # Categorical extensions
        elif dtype == "categorical":
            vc = s.value_counts(normalize=True)
            if not vc.empty:
                kw.update(
                    top_freq=float(vc.max()),
                    rare_pct=float((vc < 0.01).mean()),
                )

        # Text extensions
        elif dtype == "text":
            lengths = s.dropna().str.len()
            kw.update(
                avg_length=float(lengths.mean()),
            )

        # Date-time extensions
        elif dtype == "datetime":
            kw.update(
                span_days=(s.max() - s.min()).days if not s.empty else None
            )

        # Correlation (if target is provided and valid)
        if target is not None:
            corr = _quick_corr(s, target)
            if corr is not None:
                kw["corr_target"] = corr


        stats[col] = FeatureOverview(**kw)

    return stats
