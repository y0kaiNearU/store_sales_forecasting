from pathlib import Path

import pandas as pd
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]


def load_raw():
    train = pd.read_csv(f"{RAW_DIR}/train.csv", parse_dates=["Date"])
    test = pd.read_csv(f"{RAW_DIR}/test.csv", parse_dates=["Date"])
    features = pd.read_csv(f"{RAW_DIR}/features.csv", parse_dates=["Date"])
    stores = pd.read_csv(f"{RAW_DIR}/stores.csv")
    return train, test, features, stores


def build_features(df, features_df, stores_df):
    df = df.merge(features_df, on=["Store", "Date", "IsHoliday"], how="left")
    df = df.merge(stores_df, on="Store", how="left")

    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    df[MARKDOWN_COLS] = df[MARKDOWN_COLS].fillna(0)
    df["CPI"] = df["CPI"].fillna(df["CPI"].median())
    df["Unemployment"] = df["Unemployment"].fillna(df["Unemployment"].median())

    return df


def add_unique_id(df):
    df = df.copy()
    df["unique_id"] = df["Store"].astype(str) + "_" + df["Dept"].astype(str)
    return df


def to_long_format(df, extra_cols=("IsHoliday",)):
    long_df = add_unique_id(df)
    long_df["ds"] = pd.to_datetime(long_df["Date"])
    long_df = long_df.rename(columns={"Weekly_Sales": "y"})
    cols = ["unique_id", "ds", "y"] + list(extra_cols)
    return long_df[cols].sort_values(["unique_id", "ds"]).reset_index(drop=True)


def add_lag_features(df, lags=(1, 2, 4, 8, 52), target="Weekly_Sales"):
    df = df.sort_values(["Store", "Dept", "Date"])
    for lag in lags:
        df[f"lag_{lag}w"] = df.groupby(["Store", "Dept"])[target].shift(lag)
    return df


def wmae(y_true, y_pred, is_holiday):
    weights = is_holiday.map({True: 5, False: 1})
    return (weights * np.abs(y_true - y_pred)).sum() / weights.sum()
