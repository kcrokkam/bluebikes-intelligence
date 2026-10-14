import gc
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor


DATA_FILE = "data/processed/station_hour_features.csv"
STATION_PROFILE_FILE = "data/processed/station_demand_profile.csv"
NETWORK_PROFILE_FILE = "data/processed/network_demand_profile.csv"

TRAIN_START = "2025-09-01 00:00:00"
TRAIN_END = "2026-02-28 23:00:00"
VALIDATION_START = "2026-03-01 00:00:00"
VALIDATION_END = "2026-05-31 23:00:00"
TEST_START = "2026-06-01 00:00:00"

FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "temperature",
    "precipitation",
    "historical_departures",
    "historical_arrivals",
]


def calculate_metrics(actual, predicted):
    """Return MAE and WAPE."""

    absolute_error = (actual - predicted).abs()

    return (
        absolute_error.mean(),
        absolute_error.sum() / actual.sum(),
    )


def load_data():
    """Load model data and attach frozen historical demand profiles."""

    print("Loading forecasting data...")

    df = pd.read_csv(
        DATA_FILE,
        usecols=[
            "station_id",
            "datetime",
            "departures",
            "arrivals",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
            "is_holiday",
            "temperature",
            "precipitation",
        ],
        parse_dates=["datetime"],
    )

    df = df[
        df["datetime"] >= TRAIN_START
    ].copy()

    station_profile = pd.read_csv(
        STATION_PROFILE_FILE
    )

    network_profile = pd.read_csv(
        NETWORK_PROFILE_FILE
    )

    df = df.merge(
        station_profile,
        on=["station_id", "day_of_week", "hour"],
        how="left",
    )

    df = df.merge(
        network_profile,
        on=["day_of_week", "hour"],
        how="left",
    )

    df["historical_departures"] = (
        df["historical_departures"]
        .fillna(df["network_departures"])
    )

    df["historical_arrivals"] = (
        df["historical_arrivals"]
        .fillna(df["network_arrivals"])
    )

    return df


def run_model(target, train, validation, test):
    """Compare the frozen-profile baseline with LightGBM."""

    print(f"\nTraining production-feasible {target} model...")

    baseline_column = (
        "historical_departures"
        if target == "departures"
        else "historical_arrivals"
    )

    test_actual = test[target].reset_index(drop=True)

    baseline_predictions = (
        test[baseline_column]
        .reset_index(drop=True)
    )

    baseline_mae, baseline_wape = calculate_metrics(
        test_actual,
        baseline_predictions,
    )

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        train[FEATURES],
        train[target],
    )

    validation_predictions = pd.Series(
        model.predict(validation[FEATURES])
    ).clip(lower=0)

    validation_actual = (
        validation[target]
        .reset_index(drop=True)
    )

    validation_mae, validation_wape = calculate_metrics(
        validation_actual,
        validation_predictions,
    )

    test_predictions = pd.Series(
        model.predict(test[FEATURES])
    ).clip(lower=0)

    test_mae, test_wape = calculate_metrics(
        test_actual,
        test_predictions,
    )

    print("\nFROZEN PROFILE BASELINE")
    print(f"Test MAE: {baseline_mae:.3f}")
    print(f"Test WAPE: {baseline_wape:.3f}")

    print("\nLIGHTGBM")
    print(f"Validation MAE: {validation_mae:.3f}")
    print(f"Validation WAPE: {validation_wape:.3f}")
    print(f"Test MAE: {test_mae:.3f}")
    print(f"Test WAPE: {test_wape:.3f}")

    # Keep production and frozen-profile predictions separate from the
    # offline lag-feature research model's test prediction files.
    output = test[["station_id", "datetime", target]].reset_index(drop=True).copy()
    output[f"predicted_{target}"] = test_predictions.to_numpy()
    output[f"baseline_{target}"] = baseline_predictions.to_numpy()

    output_path = Path(f"data/processed/production_test_{target}_predictions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"Saved ranking predictions to: {output_path}")

    del model
    gc.collect()


def main():
    df = load_data()

    train = df[
        (df["datetime"] >= TRAIN_START)
        & (df["datetime"] <= TRAIN_END)
    ]

    validation = df[
        (df["datetime"] >= VALIDATION_START)
        & (df["datetime"] <= VALIDATION_END)
    ]

    test = df[
        df["datetime"] >= TEST_START
    ]

    print(f"Train rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    run_model(
        "departures",
        train,
        validation,
        test,
    )

    run_model(
        "arrivals",
        train,
        validation,
        test,
    )


if __name__ == "__main__":
    main()
