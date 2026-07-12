from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "models"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"

WANDB_ENTITY = os.getenv("WANDB_ENTITY", "gchal22-free-university-of-tbilisi-")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "store_sales_forecast")

RANDOM_STATE = 42
VALIDATION_FOLDS = 3
VALIDATION_WEEKS = 8

LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLLING_WINDOWS = [4, 8, 13, 26]

TARGET_COL = "Weekly_Sales"
ID_COL = "Id"
KEY_COLS = ["Store", "Dept"]
