from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor

from .config import LAGS, ROLLING_WINDOWS, TARGET_COL
from .features import (
    add_aggregate_features,
    add_train_lag_features,
    fill_model_feature_missing_values,
    prepare_base_features,
)


@dataclass
class ModelInfo:
    name: str
    params: dict[str, Any]


class WalmartSalesForecaster(BaseEstimator, RegressorMixin):
    """End-to-end forecaster that accepts raw Kaggle train/test frames.

    It stores train history internally and creates lag/rolling features recursively during
    prediction, so `predict(raw_test)` works without manually preprocessing the test set.
    """

    def __init__(
        self,
        model: Any | None = None,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
        clip_negative: bool = True,
        use_sample_weight: bool = True,
        model_name: str = "tree_model",
    ):
        self.model = model
        self.lags = lags if lags is not None else LAGS
        self.rolling_windows = rolling_windows if rolling_windows is not None else ROLLING_WINDOWS
        self.clip_negative = clip_negative
        self.use_sample_weight = use_sample_weight
        self.model_name = model_name

    def fit(self, train_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame):
        if TARGET_COL not in train_df.columns:
            raise ValueError(f"train_df must contain {TARGET_COL}")

        self.features_df_ = features_df.copy()
        self.stores_df_ = stores_df.copy()
        train_df = train_df.copy()
        train_df["Date"] = pd.to_datetime(train_df["Date"])

        self.history_ = train_df[["Store", "Dept", "Date", TARGET_COL]].copy()
        self.stats_ = self._compute_stats(train_df)

        train_features = prepare_base_features(train_df, self.features_df_, self.stores_df_)
        train_features = add_train_lag_features(train_features, self.lags, self.rolling_windows)
        train_features = add_aggregate_features(train_features, self.stats_)

        drop_cols = {TARGET_COL, "Date", "Id"}
        numeric_cols = train_features.select_dtypes(include=["number", "bool"]).columns.tolist()
        self.feature_columns_ = [c for c in numeric_cols if c not in drop_cols]

        train_features = fill_model_feature_missing_values(train_features)
        self.feature_medians_ = train_features[self.feature_columns_].median(numeric_only=True).to_dict()

        X = train_features[self.feature_columns_]
        y = train_features[TARGET_COL].astype(float)

        if self.model is None:
            self.model_ = RandomForestRegressor(
                n_estimators=80,
                max_depth=18,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        else:
            try:
                self.model_ = clone(self.model)
            except Exception:
                self.model_ = self.model

        fit_kwargs = {}
        if self.use_sample_weight:
            fit_kwargs["sample_weight"] = np.where(train_features["IsHoliday"].astype(bool), 5.0, 1.0)

        try:
            self.model_.fit(X, y, **fit_kwargs)
        except TypeError:
            self.model_.fit(X, y)
        return self

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        raw = raw_df.copy()
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw["_original_index"] = np.arange(len(raw))
        raw = raw.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

        history = self.history_.copy()
        predictions_by_original_index: dict[int, float] = {}

        for date, current_rows in raw.groupby("Date", sort=True):
            current_rows = current_rows.copy()
            feature_frame = self._make_future_features(current_rows, history)
            X = feature_frame[self.feature_columns_]
            preds = np.asarray(self.model_.predict(X), dtype=float)
            if self.clip_negative:
                preds = np.maximum(preds, 0.0)

            for idx, pred in zip(current_rows["_original_index"].tolist(), preds.tolist()):
                predictions_by_original_index[int(idx)] = float(pred)

            new_history = current_rows[["Store", "Dept", "Date"]].copy()
            new_history[TARGET_COL] = preds
            history = pd.concat([history, new_history], ignore_index=True)

        ordered = [predictions_by_original_index[i] for i in range(len(raw_df))]
        return np.asarray(ordered, dtype=float)

    def _make_future_features(self, current_rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        base = prepare_base_features(current_rows.drop(columns=["_original_index"], errors="ignore"), self.features_df_, self.stores_df_)
        base = add_aggregate_features(base, self.stats_)

        hist = history.copy()
        hist["Date"] = pd.to_datetime(hist["Date"])
        for lag in self.lags:
            lag_source = hist[["Store", "Dept", "Date", TARGET_COL]].copy()
            lag_source["Date"] = lag_source["Date"] + pd.to_timedelta(7 * lag, unit="D")
            lag_source = lag_source.rename(columns={TARGET_COL: f"lag_{lag}"})
            base = base.merge(lag_source, on=["Store", "Dept", "Date"], how="left")

        hist_sorted = hist.sort_values(["Store", "Dept", "Date"])
        for window in self.rolling_windows:
            roll = (
                hist_sorted.groupby(["Store", "Dept"])[TARGET_COL]
                .apply(lambda s, w=window: float(s.tail(w).mean()) if len(s) else np.nan)
                .rename(f"rolling_mean_{window}")
                .reset_index()
            )
            roll_std = (
                hist_sorted.groupby(["Store", "Dept"])[TARGET_COL]
                .apply(lambda s, w=window: float(s.tail(w).std()) if len(s.tail(w)) > 1 else np.nan)
                .rename(f"rolling_std_{window}")
                .reset_index()
            )
            base = base.merge(roll, on=["Store", "Dept"], how="left")
            base = base.merge(roll_std, on=["Store", "Dept"], how="left")

        base = fill_model_feature_missing_values(base, self.feature_medians_)
        for col in self.feature_columns_:
            if col not in base.columns:
                base[col] = self.feature_medians_.get(col, 0.0)
        return base

    def _compute_stats(self, train_df: pd.DataFrame) -> dict[str, Any]:
        y = train_df[TARGET_COL].astype(float)
        key_group = train_df.groupby(["Store", "Dept"])[TARGET_COL]
        return {
            "store_dept_mean": key_group.mean().to_dict(),
            "store_dept_median": key_group.median().to_dict(),
            "store_mean": train_df.groupby("Store")[TARGET_COL].mean().to_dict(),
            "store_median": train_df.groupby("Store")[TARGET_COL].median().to_dict(),
            "dept_mean": train_df.groupby("Dept")[TARGET_COL].mean().to_dict(),
            "dept_median": train_df.groupby("Dept")[TARGET_COL].median().to_dict(),
            "global_mean": float(y.mean()),
            "global_median": float(y.median()),
        }

    def _check_is_fitted(self):
        required = ["model_", "feature_columns_", "history_", "stats_", "features_df_", "stores_df_"]
        missing = [name for name in required if not hasattr(self, name)]
        if missing:
            raise RuntimeError(f"Model is not fitted. Missing attributes: {missing}")


def make_lightgbm_model(random_state: int = 42, **overrides):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("Install LightGBM first: pip install lightgbm") from exc

    params = dict(
        objective="regression_l1",
        n_estimators=1200,
        learning_rate=0.035,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.2,
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(overrides)
    return LGBMRegressor(**params)


def make_xgboost_model(random_state: int = 42, **overrides):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("Install XGBoost first: pip install xgboost") from exc

    params = dict(
        objective="reg:absoluteerror",
        n_estimators=900,
        learning_rate=0.035,
        max_depth=8,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.5,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )
    params.update(overrides)
    return XGBRegressor(**params)
