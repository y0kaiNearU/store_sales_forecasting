from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

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
