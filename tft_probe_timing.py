import time
import numpy as np
from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from src.data_loading import load_raw_data
from src.features import build_features, to_long_format, MARKDOWN_COLS
from src.metrics import wmae

np.random.seed(42)
FUTR_EXOG_COLS = ["IsHoliday", "CPI", "Unemployment", "Fuel_Price", "Temperature"] + MARKDOWN_COLS

raw = load_raw_data()
train_full = build_features(raw["train"], raw["features"], raw["stores"])
long_df = to_long_format(train_full, extra_cols=FUTR_EXOG_COLS)

H, INPUT_SIZE, MIN_LEN = 39, 52, 91
series_len = long_df.groupby("unique_id").size()
keep_ids = series_len[series_len >= MIN_LEN].index
long_df = long_df[long_df["unique_id"].isin(keep_ids)].reset_index(drop=True)

for steps in [100, 300]:
    model = TFT(h=H, input_size=INPUT_SIZE, futr_exog_list=FUTR_EXOG_COLS, max_steps=steps,
                hidden_size=32, n_head=2, scaler_type="robust", random_seed=42,
                enable_progress_bar=False, logger=False)
    nf = NeuralForecast(models=[model], freq="W-FRI")
    t0 = time.time()
    cv = nf.cross_validation(df=long_df, n_windows=None, test_size=H)
    cv = cv.merge(long_df[["unique_id","ds","IsHoliday"]], on=["unique_id","ds"], how="left")
    w = wmae(cv["y"], cv["TFT"], cv["IsHoliday"])
    print(f"max_steps={steps:>4} | val_wmae={w:.2f} | took {time.time()-t0:.1f}s")
