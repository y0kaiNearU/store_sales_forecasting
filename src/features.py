import numpy as np
import pandas as pd

MARKDOWN_COLS = [f"MarkDown{i}" for i in range(1, 6)]
NUMERIC_EXTERNAL_COLS = ["Temperature", "Fuel_Price", *MARKDOWN_COLS, "CPI", "Unemployment", "Size"]
TYPE_MAP = {"A": 0, "B": 1, "C": 2}


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    return out


def merge_calendar_store_features(raw_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    """Merge raw sales/test rows with Kaggle features and store metadata."""
    df = ensure_datetime(raw_df)
    features = ensure_datetime(features_df).drop(columns=["IsHoliday"], errors="ignore")
    stores = stores_df.copy()

    out = df.merge(features, on=["Store", "Date"], how="left")
    out = out.merge(stores, on="Store", how="left")
    return out


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    out = ensure_datetime(df)
    iso = out["Date"].dt.isocalendar()
    out["year"] = out["Date"].dt.year.astype(int)
    out["month"] = out["Date"].dt.month.astype(int)
    out["week"] = iso.week.astype(int)
    out["day"] = out["Date"].dt.day.astype(int)
    out["quarter"] = out["Date"].dt.quarter.astype(int)
    out["dayofyear"] = out["Date"].dt.dayofyear.astype(int)
    out["is_month_start"] = out["Date"].dt.is_month_start.astype(int)
    out["is_month_end"] = out["Date"].dt.is_month_end.astype(int)
    out["is_year_end"] = out["Date"].dt.is_year_end.astype(int)
    out["IsHoliday"] = out["IsHoliday"].astype(int)

    # cyclical seasonality features for weekly data
    out["week_sin"] = np.sin(2 * np.pi * out["week"] / 52.0)
    out["week_cos"] = np.cos(2 * np.pi * out["week"] / 52.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out


def add_holiday_window_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add manually useful retail holiday indicators.

    The competition weighs holiday weeks more, so tree models benefit from these flags.
    """
    out = ensure_datetime(df)
    out["is_super_bowl_window"] = out["week"].isin([5, 6, 7]).astype(int)
    out["is_labor_day_window"] = out["week"].isin([35, 36, 37]).astype(int)
    out["is_thanksgiving_window"] = out["week"].isin([46, 47, 48]).astype(int)
    out["is_christmas_window"] = out["week"].isin([50, 51, 52]).astype(int)
    out["is_black_friday_week"] = out["week"].isin([47, 48]).astype(int)
    return out


def fill_external_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in MARKDOWN_COLS:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
            out[col] = out[col].fillna(0.0)

    for col in ["Temperature", "Fuel_Price", "CPI", "Unemployment", "Size"]:
        if col in out.columns:
            store_median = out.groupby("Store")[col].transform("median")
            out[col] = out[col].fillna(store_median).fillna(out[col].median())

    if "Type" in out.columns:
        out["TypeOrdinal"] = out["Type"].map(TYPE_MAP).fillna(-1).astype(int)
        out = out.drop(columns=["Type"])
    return out


def prepare_base_features(raw_df: pd.DataFrame, features_df: pd.DataFrame, stores_df: pd.DataFrame) -> pd.DataFrame:
    out = merge_calendar_store_features(raw_df, features_df, stores_df)
    out = add_date_features(out)
    out = add_holiday_window_features(out)
    out = fill_external_missing_values(out)
    return out


def add_train_lag_features(df: pd.DataFrame, lags: list[int], windows: list[int]) -> pd.DataFrame:
    """Create leakage-safe lag and rolling features for training rows."""
    out = df.sort_values(["Store", "Dept", "Date"]).copy()
    group = out.groupby(["Store", "Dept"], group_keys=False)["Weekly_Sales"]

    for lag in lags:
        out[f"lag_{lag}"] = group.shift(lag)

    shifted = group.shift(1)
    for window in windows:
        out[f"rolling_mean_{window}"] = shifted.groupby([out["Store"], out["Dept"]]).transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        out[f"rolling_std_{window}"] = shifted.groupby([out["Store"], out["Dept"]]).transform(
            lambda s: s.rolling(window=window, min_periods=2).std()
        )
    return out


def add_aggregate_features(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Map precomputed train aggregates onto train/validation/test rows."""
    out = df.copy()
    key_index = list(zip(out["Store"], out["Dept"]))
    out["store_dept_mean"] = pd.Series(key_index, index=out.index).map(stats["store_dept_mean"])
    out["store_dept_median"] = pd.Series(key_index, index=out.index).map(stats["store_dept_median"])
    out["store_mean"] = out["Store"].map(stats["store_mean"])
    out["store_median"] = out["Store"].map(stats["store_median"])
    out["dept_mean"] = out["Dept"].map(stats["dept_mean"])
    out["dept_median"] = out["Dept"].map(stats["dept_median"])
    out["global_mean"] = stats["global_mean"]
    out["global_median"] = stats["global_median"]
    return out


def fill_model_feature_missing_values(df: pd.DataFrame, feature_medians: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number", "bool"]).columns
    if feature_medians is None:
        medians = out[numeric_cols].median(numeric_only=True).to_dict()
    else:
        medians = feature_medians
    for col in numeric_cols:
        value = medians.get(col, out[col].median())
        if pd.isna(value):
            value = 0.0
        out[col] = out[col].fillna(value)
    return out
