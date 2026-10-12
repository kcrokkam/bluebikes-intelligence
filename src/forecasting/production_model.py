import gc

import pandas as pd

from lightgbm import LGBMRegressor


DATA_FILE = "data/processed/station_hour_features.csv"

STATION_PROFILE_FILE = (
    "data/processed/station_demand_profile.csv"
)

NETWORK_PROFILE_FILE = (
    "data/processed/network_demand_profile.csv"
)


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

    absolute_error = (
        actual - predicted
    ).abs()

    mae = absolute_error.mean()

    wape = (
        absolute_error.sum()
        / actual.sum()
    )

    return mae, wape


def load_data():

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

    # We only need data after the
    # historical profile cutoff.
    df = df[
        df["datetime"] >= TRAIN_START
    ].copy()

    print("Loading station demand profiles...")

    station_profile = pd.read_csv(
        STATION_PROFILE_FILE
    )

    network_profile = pd.read_csv(
        NETWORK_PROFILE_FILE
    )

    # Add station-specific historical demand
    df = df.merge(
        station_profile,
        on=[
            "station_id",
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    # Add network fallback demand
    df = df.merge(
        network_profile,
        on=[
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    # New stations may not have their own history.
    # Use the network profile instead.
    df["historical_departures"] = (
        df["historical_departures"]
        .fillna(
            df["network_departures"]
        )
    )

    df["historical_arrivals"] = (
        df["historical_arrivals"]
        .fillna(
            df["network_arrivals"]
        )
    )

    return df


def run_model(
    target,
    df,
):

    print(
        f"\nTraining production-feasible "
        f"{target} model..."
    )

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

    print(
        f"Train rows: {len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Test rows: {len(test):,}"
    )

    # Simple frozen historical-profile baseline
    baseline_column = (
        "historical_departures"
        if target == "departures"
        else "historical_arrivals"
    )

    baseline_predictions = (
        test[baseline_column]
        .reset_index(drop=True)
    )

    test_actual = (
        test[target]
        .reset_index(drop=True)
    )

    baseline_mae, baseline_wape = (
        calculate_metrics(
            test_actual,
            baseline_predictions,
        )
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

    validation_predictions = (
        pd.Series(
            model.predict(
                validation[FEATURES]
            )
        )
        .clip(lower=0)
    )

    validation_actual = (
        validation[target]
        .reset_index(drop=True)
    )

    validation_mae, validation_wape = (
        calculate_metrics(
            validation_actual,
            validation_predictions,
        )
    )

    test_predictions = (
        pd.Series(
            model.predict(
                test[FEATURES]
            )
        )
        .clip(lower=0)
    )

    test_mae, test_wape = (
        calculate_metrics(
            test_actual,
            test_predictions,
        )
    )

    print("\nFROZEN PROFILE BASELINE")

    print(
        f"Test MAE: "
        f"{baseline_mae:.3f}"
    )

    print(
        f"Test WAPE: "
        f"{baseline_wape:.3f}"
    )

    print("\nLIGHTGBM")

    print(
        f"Validation MAE: "
        f"{validation_mae:.3f}"
    )

    print(
        f"Validation WAPE: "
        f"{validation_wape:.3f}"
    )

    print(
        f"Test MAE: "
        f"{test_mae:.3f}"
    )

    print(
        f"Test WAPE: "
        f"{test_wape:.3f}"
    )

    del model

    gc.collect()


def main():

    df = load_data()

    run_model(
        "departures",
        df,
    )

    run_model(
        "arrivals",
        df,
    )


if __name__ == "__main__":
    main()