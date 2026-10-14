import gc

import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation


DATA_FILE = "data/processed/station_hour_features.csv"

TRAIN_END = "2026-02-28 23:00:00"
VALIDATION_START = "2026-03-01 00:00:00"
VALIDATION_END = "2026-05-31 23:00:00"
TEST_START = "2026-06-01 00:00:00"

COMMON_FEATURES = [
    "station_id",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature",
    "precipitation",
]


def calculate_metrics(actual, predicted):
    """Return MAE and WAPE."""

    absolute_error = (actual - predicted).abs()

    mae = absolute_error.mean()
    wape = absolute_error.sum() / actual.sum()

    return mae, wape


def evaluate_station_groups(
    station_average,
    station_ids,
    actual,
    predicted,
):
    """Compare test performance for quiet and busy stations."""

    low_threshold = station_average.quantile(0.25)
    high_threshold = station_average.quantile(0.75)

    test_average = station_ids.map(station_average)

    groups = {
        "LOW VOLUME": test_average <= low_threshold,
        "BUSY": test_average >= high_threshold,
    }

    print("\nSTATION VOLUME COMPARISON")

    for name, mask in groups.items():
        mae, wape = calculate_metrics(
            actual[mask],
            predicted[mask],
        )

        print(f"\n{name}")
        print(f"Rows: {mask.sum():,}")
        print(f"MAE: {mae:.3f}")
        print(f"WAPE: {wape:.3f}")


def run_model(target):
    """Train and evaluate one research model with recent-demand lags."""

    lag_features = (
        ["departures_lag_24h", "departures_lag_168h"]
        if target == "departures"
        else ["arrivals_lag_24h", "arrivals_lag_168h"]
    )

    features = COMMON_FEATURES + lag_features

    columns = [
        "datetime",
        target,
        *features,
    ]

    print(f"\nLoading {target} data...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=columns,
        parse_dates=["datetime"],
        dtype={
            "station_id": "category",
            "hour": "int8",
            "day_of_week": "int8",
            "month": "int8",
            "is_weekend": "int8",
            "is_holiday": "int8",
            "temperature": "float32",
            "precipitation": "float32",
            target: "float32",
            lag_features[0]: "float32",
            lag_features[1]: "float32",
        },
    )

    train_mask = df["datetime"] <= TRAIN_END

    validation_mask = (
        (df["datetime"] >= VALIDATION_START)
        & (df["datetime"] <= VALIDATION_END)
    )

    test_mask = df["datetime"] >= TEST_START

    print(f"Train rows: {train_mask.sum():,}")
    print(f"Validation rows: {validation_mask.sum():,}")
    print(f"Test rows: {test_mask.sum():,}")

    station_average = (
        df.loc[train_mask, ["station_id", target]]
        .groupby("station_id", observed=True)[target]
        .mean()
    )

    station_average.index = station_average.index.astype(str)

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, target]

    X_validation = df.loc[validation_mask, features]
    y_validation = df.loc[validation_mask, target]

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print(f"\nTraining {target} model...")

    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="mae",
        callbacks=[
            early_stopping(30),
            log_evaluation(50),
        ],
    )

    validation_predictions = pd.Series(
        model.predict(X_validation)
    ).clip(lower=0)

    validation_actual = y_validation.reset_index(drop=True)

    validation_mae, validation_wape = calculate_metrics(
        validation_actual,
        validation_predictions,
    )

    del X_train, y_train, X_validation, y_validation
    gc.collect()

    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, target].reset_index(drop=True)

    test_station_ids = (
        X_test["station_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    test_predictions = pd.Series(
        model.predict(X_test)
    ).clip(lower=0)

    test_mae, test_wape = calculate_metrics(
        y_test,
        test_predictions,
    )

    output = df.loc[
        test_mask,
        ["station_id", "datetime", target],
    ].copy()

    output[f"predicted_{target}"] = test_predictions.values

    output.to_csv(
        f"data/processed/test_{target}_predictions.csv",
        index=False,
    )

    print(f"\n{target.upper()}")

    print("\nVALIDATION")
    print(f"MAE: {validation_mae:.3f}")
    print(f"WAPE: {validation_wape:.3f}")

    print("\nTEST")
    print(f"MAE: {test_mae:.3f}")
    print(f"WAPE: {test_wape:.3f}")

    print(f"\nBest iteration: {model.best_iteration_}")

    evaluate_station_groups(
        station_average,
        test_station_ids,
        y_test,
        test_predictions,
    )

    del df, X_test, y_test, test_predictions, model
    gc.collect()


def main():
    run_model("departures")
    run_model("arrivals")


if __name__ == "__main__":
    main()
