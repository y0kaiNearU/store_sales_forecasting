from __future__ import annotations

from pathlib import Path

import pandas as pd
from neuralforecast import NeuralForecast

from .preprocessing import add_unique_id, MARKDOWN_COLS


class NeuralForecastPipeline:

    def __init__(self, model_col: str):
        self.model_col = model_col
        self.nf: NeuralForecast | None = None

    @classmethod
    def load(cls, path: str | Path, model_col: str) -> "NeuralForecastPipeline":
        obj = cls(model_col)
        obj.nf = NeuralForecast.load(path=str(path))
        return obj

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = add_unique_id(raw_df.copy())
        df["ds"] = pd.to_datetime(df["Date"])

        preds = self.nf.predict()
        preds = preds.rename(columns={self.model_col: "Weekly_Sales"})

        out = df[["Store", "Dept", "Date", "unique_id", "ds"]].merge(
            preds, on=["unique_id", "ds"], how="left"
        )
        return out[["Store", "Dept", "Date", "Weekly_Sales"]]


class TFTForecastPipeline:

    FUTR_EXOG_COLS = [
        "IsHoliday", "CPI", "Unemployment", "Fuel_Price", "Temperature",
        "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    ]

    def __init__(self):
        self.nf: NeuralForecast | None = None
        self.features: pd.DataFrame | None = None
        self.stores: pd.DataFrame | None = None

    @classmethod
    def load(cls, nf_path: str | Path, features_path: str | Path, stores_path: str | Path) -> "TFTForecastPipeline":
        obj = cls()
        obj.nf = NeuralForecast.load(path=str(nf_path))
        obj.features = pd.read_csv(features_path, parse_dates=["Date"])
        obj.stores = pd.read_csv(stores_path)
        return obj

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        future_df = self.nf.make_future_dataframe()
        ids = future_df["unique_id"].str.split("_", n=1, expand=True)
        future_df["Store"] = ids[0].astype(int)
        future_df["Dept"] = ids[1].astype(int)
        future_df["Date"] = future_df["ds"]

        exog = future_df.merge(self.features, on=["Store", "Date"], how="left")
        exog = exog.merge(self.stores, on="Store", how="left")
        exog[MARKDOWN_COLS] = exog[MARKDOWN_COLS].fillna(0)
        exog["CPI"] = exog["CPI"].fillna(exog["CPI"].median())
        exog["Unemployment"] = exog["Unemployment"].fillna(exog["Unemployment"].median())

        futr_df = exog[["unique_id", "ds"] + self.FUTR_EXOG_COLS]
        preds = self.nf.predict(futr_df=futr_df)
        preds = preds.rename(columns={"TFT": "Weekly_Sales"})

        out = add_unique_id(raw_df.copy())
        out["ds"] = pd.to_datetime(out["Date"])
        out = out[["Store", "Dept", "Date", "unique_id", "ds"]].merge(
            preds, on=["unique_id", "ds"], how="left"
        )
        return out[["Store", "Dept", "Date", "Weekly_Sales"]]


class TimesFMForecastPipeline:

    REPO_ID = "google/timesfm-2.5-200m-pytorch"
    INPUT_SIZE = 52
    HORIZON = 39

    def __init__(self):
        self.model = None
        self.train: pd.DataFrame | None = None

    @classmethod
    def load(
        cls, train_csv_path: str | Path, checkpoint_path: str | Path | None = None
    ) -> "TimesFMForecastPipeline":
        """Loads the pretrained TimesFM weights, optionally overriding them with a
        fine-tuned checkpoint directory/file (as produced by model.save_pretrained())."""
        import timesfm

        obj = cls()
        obj.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(obj.REPO_ID)
        if checkpoint_path is not None:
            obj.model.load_checkpoint(str(checkpoint_path))
        obj.model.compile(timesfm.ForecastConfig(
            max_context=obj.INPUT_SIZE,
            max_horizon=obj.HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=False,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        ))
        train = pd.read_csv(train_csv_path, parse_dates=["Date"])
        obj.train = add_unique_id(train)
        return obj

    def predict(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = add_unique_id(raw_df.copy())
        df["ds"] = pd.to_datetime(df["Date"])

        contexts, valid_uids, last_dates = [], [], []
        for uid in df["unique_id"].unique():
            hist = self.train[self.train["unique_id"] == uid].sort_values("Date")
            if len(hist) >= self.INPUT_SIZE:
                contexts.append(hist["Weekly_Sales"].values[-self.INPUT_SIZE:].astype("float32"))
                valid_uids.append(uid)
                last_dates.append(hist["Date"].max())

        preds_df = pd.DataFrame(columns=["unique_id", "ds", "Weekly_Sales"])
        if contexts:
            point, _ = self.model.forecast(horizon=self.HORIZON, inputs=contexts)
            rows = []
            for uid, last_date, series_point in zip(valid_uids, last_dates, point):
                fcst_dates = pd.date_range(
                    last_date + pd.Timedelta(weeks=1), periods=self.HORIZON, freq="W-FRI"
                )
                rows.append(pd.DataFrame({
                    "unique_id": uid, "ds": fcst_dates, "Weekly_Sales": series_point,
                }))
            preds_df = pd.concat(rows, ignore_index=True)

        out = df[["Store", "Dept", "Date", "unique_id", "ds"]].merge(
            preds_df, on=["unique_id", "ds"], how="left"
        )
        return out[["Store", "Dept", "Date", "Weekly_Sales"]]
