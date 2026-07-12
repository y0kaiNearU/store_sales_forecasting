from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import TARGET_COL
from .metrics import metric_report, wmae


def make_time_folds(
    train_df: pd.DataFrame,
    n_folds: int = 3,
    validation_weeks: int = 8,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Create chronological expanding-window folds.

    Returns tuples: train_start, train_end, valid_start, valid_end.
    """
    df = train_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    dates = sorted(df["Date"].unique())
    if len(dates) < (n_folds + 1) * validation_weeks:
        raise ValueError("Not enough dates for requested folds.")

    folds = []
    for fold_idx in range(n_folds):
        val_start_pos = len(dates) - validation_weeks * (n_folds - fold_idx)
        val_end_pos = val_start_pos + validation_weeks - 1
        train_start = dates[0]
        train_end = dates[val_start_pos - 1]
        valid_start = dates[val_start_pos]
        valid_end = dates[val_end_pos]
        folds.append((pd.Timestamp(train_start), pd.Timestamp(train_end), pd.Timestamp(valid_start), pd.Timestamp(valid_end)))
    return folds


def evaluate_time_folds(
    forecaster_factory: Callable[[], object],
    train_df: pd.DataFrame,
    features_df: pd.DataFrame,
    stores_df: pd.DataFrame,
    n_folds: int = 3,
    validation_weeks: int = 8,
    log_callback: Callable[[dict], None] | None = None,
) -> pd.DataFrame:
    """Evaluate a forecaster using chronological folds."""
    print("USING FIXED evaluate_time_folds v2")

    df = train_df.copy().reset_index(drop=True)
    features_df = features_df.copy().reset_index(drop=True)
    stores_df = stores_df.copy().reset_index(drop=True)

    df["Date"] = pd.to_datetime(df["Date"])

    rows = []
    folds = make_time_folds(
        df,
        n_folds=n_folds,
        validation_weeks=validation_weeks,
    )

    for i, (train_start, train_end, valid_start, valid_end) in enumerate(folds, start=1):
        train_mask = (
            (df["Date"] >= train_start)
            & (df["Date"] <= train_end)
        ).to_numpy()

        valid_mask = (
            (df["Date"] >= valid_start)
            & (df["Date"] <= valid_end)
        ).to_numpy()

        fold_train = df.iloc[train_mask].copy().reset_index(drop=True)
        fold_valid = df.iloc[valid_mask].copy().reset_index(drop=True)

        model = forecaster_factory()
        model.fit(fold_train, features_df, stores_df)

        predict_input = fold_valid.drop(columns=[TARGET_COL], errors="ignore").copy().reset_index(drop=True)
        preds = model.predict(predict_input)

        y_true = fold_valid[TARGET_COL].to_numpy()
        is_holiday = fold_valid["IsHoliday"].to_numpy()

        if isinstance(preds, (pd.Series, pd.DataFrame)):
            preds = preds.to_numpy()

        report = metric_report(y_true, preds, is_holiday)

        row = {
            "fold": i,
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "valid_start": valid_start.date().isoformat(),
            "valid_end": valid_end.date().isoformat(),
            "train_rows": len(fold_train),
            "valid_rows": len(fold_valid),
            **report,
        }

        rows.append(row)

        if log_callback is not None:
            log_callback({
                f"fold_{i}_{k}": v
                for k, v in row.items()
                if isinstance(v, (int, float, np.number))
            })

    return pd.DataFrame(rows)


def summarize_cv(cv_results: pd.DataFrame) -> dict:
    return {
        "cv_wmae_mean": float(cv_results["wmae"].mean()),
        "cv_wmae_std": float(cv_results["wmae"].std(ddof=0)),
        "cv_mae_mean": float(cv_results["mae"].mean()),
    }
