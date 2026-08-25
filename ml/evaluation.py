"""Shared model evaluation helpers for retraining and experiments.

The performance gate must compare production and candidate models on the
same validation rows.  This module keeps that protocol explicit: validation
rows are split away from candidate-training rows first, then both models are
scored with the same feature ordering and target values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "RUL"
NON_FEATURE_COLUMNS = [TARGET_COLUMN, "unit", "cycle"]


@dataclass(frozen=True)
class EvaluationResult:
    """Regression metrics calculated on one concrete validation set."""

    mae: float
    rmse: float
    n_samples: int
    row_indices: tuple[int, ...]


def split_training_and_validation(
    df: pd.DataFrame,
    validation_fraction: float = 0.25,
    random_state: int = 42,
    min_validation_rows: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a deterministic validation split separated from training data.

    If at least two units are available, the split is unit-based so examples
    from the same engine trajectory do not leak across train and validation.
    For a single unit, the latest cycles are held out to preserve temporal
    order within that trajectory.
    """

    clean = df.dropna().copy()
    if clean.empty:
        return clean, clean

    if "unit" in clean.columns and clean["unit"].nunique() >= 2:
        units = clean["unit"].dropna().unique()
        train_units, validation_units = train_test_split(
            units, test_size=validation_fraction, random_state=random_state
        )
        train_df = clean[clean["unit"].isin(train_units)].copy()
        validation_df = clean[clean["unit"].isin(validation_units)].copy()
    else:
        sort_cols = [c for c in ["unit", "cycle"] if c in clean.columns]
        ordered = clean.sort_values(sort_cols).copy() if sort_cols else clean.copy()
        validation_rows = max(min_validation_rows, int(round(len(ordered) * validation_fraction)))
        validation_rows = min(max(1, validation_rows), max(1, len(ordered) - 1))
        split_at = len(ordered) - validation_rows
        train_df = ordered.iloc[:split_at].copy()
        validation_df = ordered.iloc[split_at:].copy()

    if len(validation_df) < min_validation_rows and len(clean) > min_validation_rows:
        sort_cols = [c for c in ["unit", "cycle"] if c in clean.columns]
        ordered = clean.sort_values(sort_cols).copy() if sort_cols else clean.copy()
        validation_df = ordered.tail(min_validation_rows).copy()
        train_df = ordered.drop(index=validation_df.index).copy()

    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def feature_frame_for_model(model, scaler, df: pd.DataFrame) -> pd.DataFrame:
    """Return feature columns in the exact order expected by the scaler/model."""

    features = df.drop(columns=NON_FEATURE_COLUMNS, errors="ignore")
    expected_features = getattr(scaler, "feature_names_in_", None)
    if expected_features is not None:
        missing = [column for column in expected_features if column not in features.columns]
        if missing:
            raise ValueError(f"Missing required validation features: {missing}")
        features = features.loc[:, list(expected_features)]
    return features


def evaluate_regressor(model, scaler, validation_df: pd.DataFrame) -> EvaluationResult | None:
    """Evaluate a fitted regressor on a validation frame with known RUL targets."""

    if model is None or scaler is None or validation_df is None:
        return None
    if TARGET_COLUMN not in validation_df.columns or len(validation_df) == 0:
        return None

    X = feature_frame_for_model(model, scaler, validation_df)
    y_true = validation_df[TARGET_COLUMN].astype(float).to_numpy()
    if len(y_true) == 0:
        return None

    X_scaled = scaler.transform(X)
    y_pred = np.asarray(model.predict(X_scaled), dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return EvaluationResult(
        mae=mae,
        rmse=rmse,
        n_samples=int(len(y_true)),
        row_indices=tuple(int(i) for i in validation_df.index.tolist()),
    )
