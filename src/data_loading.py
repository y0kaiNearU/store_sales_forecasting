from pathlib import Path
import pandas as pd
from .config import DATA_DIR, TARGET_COL


def load_raw_data(data_dir: str | Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load Kaggle raw files from data/raw."""
    data_dir = Path(data_dir)
    files = {
        "train": "train.csv",
        "test": "test.csv",
        "features": "features.csv",
        "stores": "stores.csv",
        "sample_submission": "sampleSubmission.csv",
    }
    missing = [name for name in files.values() if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing files in {data_dir}: {missing}. Put the Kaggle CSV files in data/raw."
        )
    return {key: pd.read_csv(data_dir / filename) for key, filename in files.items()}


def make_submission_frame(test_df: pd.DataFrame, predictions, sample_submission: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a valid Kaggle submission dataframe."""
    if sample_submission is not None and "Id" in sample_submission.columns:
        submission = sample_submission.copy()
    else:
        submission = pd.DataFrame()
        submission["Id"] = (
            test_df["Store"].astype(str)
            + "_"
            + test_df["Dept"].astype(str)
            + "_"
            + test_df["Date"].astype(str)
        )
    submission[TARGET_COL] = predictions
    return submission


def describe_dataframes(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        rows.append({"name": name, "rows": len(df), "columns": len(df.columns), "column_names": ", ".join(df.columns)})
    return pd.DataFrame(rows)
