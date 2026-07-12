import numpy as np
import pandas as pd


def wmae(y_true, y_pred, is_holiday) -> float:
    """Weighted Mean Absolute Error used by the Walmart Kaggle competition.

    Holiday weeks receive weight 5, non-holiday weeks receive weight 1.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    is_holiday = np.asarray(is_holiday).astype(bool)
    weights = np.where(is_holiday, 5.0, 1.0)
    return float(np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def metric_report(y_true, y_pred, is_holiday) -> dict:
    return {
        "wmae": wmae(y_true, y_pred, is_holiday),
        "mae": mae(y_true, y_pred),
        "holiday_wmae_part": wmae(
            pd.Series(y_true)[pd.Series(is_holiday).astype(bool)],
            pd.Series(y_pred)[pd.Series(is_holiday).astype(bool)],
            np.ones(pd.Series(is_holiday).astype(bool).sum(), dtype=bool),
        ) if pd.Series(is_holiday).astype(bool).sum() > 0 else np.nan,
    }
