from sklearn.tree import DecisionTreeRegressor
from src.data_loading import load_raw_data
from src.models import WalmartSalesForecaster
from src.validation import evaluate_time_folds, summarize_cv


def main():
    data = load_raw_data()
    # Small smoke test: use a few stores only to validate feature creation and recursive prediction.
    train = data["train"][(data["train"]["Store"] == 1) & (data["train"]["Dept"].isin([1, 2, 3]))].copy()
    model_factory = lambda: WalmartSalesForecaster(
        model=DecisionTreeRegressor(max_depth=6, random_state=42),
        lags=[1, 2, 52],
        rolling_windows=[4],
        model_name="smoke_decision_tree",
    )
    cv = evaluate_time_folds(model_factory, train, data["features"], data["stores"], n_folds=1, validation_weeks=4)
    print(cv)
    print(summarize_cv(cv))


if __name__ == "__main__":
    main()
