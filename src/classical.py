from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .config import TARGET_COL
from .metrics import wmae


class SeasonalNaiveForecaster:
    """Fast classical baseline for weekly retail data.

    Main rule: predict the same Store+Dept sales from 52 weeks ago.
    Fallbacks: Store+Dept median -> Dept median -> Store median -> global median.
    """

    def fit(self, train_df: pd.DataFrame):
        train = train_df.copy()
        train["Date"] = pd.to_datetime(train["Date"])
        self.history_ = train[["Store", "Dept", "Date", TARGET_COL]].copy()
        self.store_dept_median_ = train.groupby(["Store", "Dept"])[TARGET_COL].median().to_dict()
        self.dept_median_ = train.groupby("Dept")[TARGET_COL].median().to_dict()
        self.store_median_ = train.groupby("Store")[TARGET_COL].median().to_dict()
        self.global_median_ = float(train[TARGET_COL].median())
        return self

    def predict(self, raw_df: pd.DataFrame) -> np.ndarray:
        raw = raw_df.copy()
        raw["Date"] = pd.to_datetime(raw["Date"])
        raw["lookup_date"] = raw["Date"] - pd.to_timedelta(52 * 7, unit="D")
        hist = self.history_.rename(columns={"Date": "lookup_date", TARGET_COL: "lag_52_sales"})
        out = raw.merge(hist, on=["Store", "Dept", "lookup_date"], how="left")
        preds = out["lag_52_sales"]
        key = list(zip(out["Store"], out["Dept"]))
        fallback = pd.Series(key).map(self.store_dept_median_)
        fallback = fallback.fillna(out["Dept"].map(self.dept_median_))
        fallback = fallback.fillna(out["Store"].map(self.store_median_))
        fallback = fallback.fillna(self.global_median_)
        preds = preds.fillna(fallback)
        return np.maximum(preds.to_numpy(dtype=float), 0.0)


@dataclass
class SeriesResult:
    store: int
    dept: int
    method: str
    wmae: float
    status: str


def evaluate_seasonal_naive(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> dict:
    model = SeasonalNaiveForecaster().fit(train_df)
    preds = model.predict(valid_df.drop(columns=[TARGET_COL], errors="ignore"))
    return {"wmae": wmae(valid_df[TARGET_COL], preds, valid_df["IsHoliday"])}


def top_series(train_df: pd.DataFrame, n: int = 5) -> list[tuple[int, int]]:
    totals = train_df.groupby(["Store", "Dept"])[TARGET_COL].sum().sort_values(ascending=False)
    return [(int(store), int(dept)) for store, dept in totals.head(n).index.tolist()]


def evaluate_sarima_selected_series(
    train_df: pd.DataFrame,
    series_keys: Iterable[tuple[int, int]] | None = None,
    validation_weeks: int = 8,
    seasonal_period: int = 52,
) -> pd.DataFrame:
    """Run SARIMA on a few selected series.

    This is intentionally limited because fitting SARIMA for every Store+Dept series is slow.
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError("Install statsmodels first: pip install statsmodels") from exc

    train = train_df.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    if series_keys is None:
        series_keys = top_series(train, n=5)

    rows: list[SeriesResult] = []
    for store, dept in series_keys:
        sdf = train[(train["Store"] == store) & (train["Dept"] == dept)].sort_values("Date")
        if len(sdf) <= validation_weeks + 60:
            rows.append(SeriesResult(store, dept, "SARIMA", np.nan, "skipped_too_short"))
            continue
        y_train = sdf[TARGET_COL].iloc[:-validation_weeks].astype(float)
        valid = sdf.iloc[-validation_weeks:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = SARIMAX(
                    y_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 0, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fitted = model.fit(disp=False, maxiter=80)
                preds = fitted.forecast(steps=validation_weeks)
            score = wmae(valid[TARGET_COL], np.maximum(preds, 0.0), valid["IsHoliday"])
            rows.append(SeriesResult(store, dept, "SARIMA", score, "ok"))
        except Exception as exc:
            rows.append(SeriesResult(store, dept, "SARIMA", np.nan, f"failed: {type(exc).__name__}"))
    return pd.DataFrame([r.__dict__ for r in rows])


def evaluate_prophet_selected_series(
    train_df: pd.DataFrame,
    series_keys: Iterable[tuple[int, int]] | None = None,
    validation_weeks: int = 8,
) -> pd.DataFrame:
    """Run Prophet on selected Store+Dept series."""
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("Install Prophet first: pip install prophet") from exc

    train = train_df.copy()
    train["Date"] = pd.to_datetime(train["Date"])
    if series_keys is None:
        series_keys = top_series(train, n=5)

    rows: list[SeriesResult] = []
    for store, dept in series_keys:
        sdf = train[(train["Store"] == store) & (train["Dept"] == dept)].sort_values("Date")
        if len(sdf) <= validation_weeks + 20:
            rows.append(SeriesResult(store, dept, "Prophet", np.nan, "skipped_too_short"))
            continue
        work = sdf[["Date", TARGET_COL, "IsHoliday"]].rename(columns={"Date": "ds", TARGET_COL: "y"})
        fit_df = work.iloc[:-validation_weeks]
        valid = sdf.iloc[-validation_weeks:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode="multiplicative",
                )
                model.add_country_holidays(country_name="US")
                model.fit(fit_df[["ds", "y"]])
                future = work[["ds"]].iloc[-validation_weeks:]
                forecast = model.predict(future)
            preds = np.maximum(forecast["yhat"].to_numpy(dtype=float), 0.0)
            score = wmae(valid[TARGET_COL], preds, valid["IsHoliday"])
            rows.append(SeriesResult(store, dept, "Prophet", score, "ok"))
        except Exception as exc:
            rows.append(SeriesResult(store, dept, "Prophet", np.nan, f"failed: {type(exc).__name__}"))
    return pd.DataFrame([r.__dict__ for r in rows])
