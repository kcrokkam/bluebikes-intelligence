import gc

import pandas as pd

from lightgbm import (
    LGBMRegressor,
    early_stopping,
    log_evaluation,
)


DATA_FILE = "data/processed/station_hour_features.csv"


# --------------------------------------------------
# 1. CHRONOLOGICAL TRAIN / VALIDATION / TEST PERIODS
# --------------------------------------------------

TRAIN_END = "2026-02-28 23:00:00"

VALIDATION_START = "2026-03-01 00:00:00"
VALIDATION_END = "2026-05-31 23:00:00"

TEST_START = "2026-06-01 00:00:00"


# --------------------------------------------------
# 2. FEATURES SHARED BY BOTH MODELS
# --------------------------------------------------

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


# --------------------------------------------------
# 3. MAE + WAPE
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    """Calculate MAE and WAPE."""

    absolute_error = (
        actual - predicted
    ).abs()

    mae = absolute_error.mean()

    wape = (
        absolute_error.sum()
        / actual.sum()
    )

    return mae, wape


# --------------------------------------------------
# 4. BUSY VS LOW-VOLUME STATION EVALUATION
# --------------------------------------------------

def evaluate_station_groups(
    station_average,
    test_station_ids,
    actual,
    predicted,
):
    """
    Compare performance on low-volume and busy stations.

    Low-volume = bottom 25% of stations by training demand.
    Busy       = top 25% of stations by training demand.
    """

    low_threshold = station_average.quantile(
        0.25
    )

    high_threshold = station_average.quantile(
        0.75
    )

    test_station_average = (
        test_station_ids
        .map(station_average)
    )

    low_mask = (
        test_station_average
        <= low_threshold
    )

    busy_mask = (
        test_station_average
        >= high_threshold
    )

    print("\nSTATION VOLUME COMPARISON")

    for name, mask in [
        ("LOW VOLUME", low_mask),
        ("BUSY", busy_mask),
    ]:

        group_actual = actual[mask]
        group_predicted = predicted[mask]

        mae, wape = calculate_metrics(
            group_actual,
            group_predicted,
        )

        print(f"\n{name}")
        print(
            f"Rows: {len(group_actual):,}"
        )
        print(
            f"MAE: {mae:.3f}"
        )
        print(
            f"WAPE: {wape:.3f}"
        )


# --------------------------------------------------
# 5. TRAIN + EVALUATE ONE MODEL
# --------------------------------------------------

def run_model(target):
    """
    Train either the departures model
    or the arrivals model.
    """

    # Use the matching lag features
    if target == "departures":

        lag_features = [
            "departures_lag_24h",
            "departures_lag_168h",
        ]

    else:

        lag_features = [
            "arrivals_lag_24h",
            "arrivals_lag_168h",
        ]

    features = (
        COMMON_FEATURES
        + lag_features
    )

    columns = [
        "datetime",
        target,
        *features,
    ]

    # --------------------------------------------------
    # LOAD ONLY THE DATA NEEDED FOR THIS MODEL
    # --------------------------------------------------

    print(
        f"\nLoading {target} data..."
    )

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

    # --------------------------------------------------
    # CREATE TIME SPLITS
    # --------------------------------------------------

    train_mask = (
        df["datetime"]
        <= TRAIN_END
    )

    validation_mask = (
        (df["datetime"] >= VALIDATION_START)
        & (df["datetime"] <= VALIDATION_END)
    )

    test_mask = (
        df["datetime"]
        >= TEST_START
    )

    print(
        f"Train rows: "
        f"{train_mask.sum():,}"
    )

    print(
        f"Validation rows: "
        f"{validation_mask.sum():,}"
    )

    print(
        f"Test rows: "
        f"{test_mask.sum():,}"
    )

    # --------------------------------------------------
    # CALCULATE TRAINING DEMAND BY STATION
    #
    # We do this BEFORE deleting training data.
    # This lets us later identify busy and quiet stations.
    # --------------------------------------------------

    station_average = (
        df.loc[
            train_mask,
            ["station_id", target],
        ]
        .groupby(
            "station_id",
            observed=True,
        )[target]
        .mean()
    )

    # Convert station IDs to strings
    # so mapping is simple later.
    station_average.index = (
        station_average.index.astype(str)
    )

    # --------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------

    X_train = df.loc[
        train_mask,
        features,
    ]

    y_train = df.loc[
        train_mask,
        target,
    ]

    X_validation = df.loc[
        validation_mask,
        features,
    ]

    y_validation = df.loc[
        validation_mask,
        target,
    ]

    # --------------------------------------------------
    # LIGHTGBM MODEL
    # --------------------------------------------------

    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print(
        f"\nTraining {target} model..."
    )

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

    # --------------------------------------------------
    # VALIDATION PERFORMANCE
    # --------------------------------------------------

    validation_predictions = (
        model.predict(
            X_validation
        )
    )

    validation_predictions = (
        pd.Series(
            validation_predictions
        )
        .clip(lower=0)
    )

    validation_actual = (
        y_validation
        .reset_index(drop=True)
    )

    validation_mae, validation_wape = (
        calculate_metrics(
            validation_actual,
            validation_predictions,
        )
    )

    # --------------------------------------------------
    # FREE TRAINING MEMORY BEFORE TESTING
    # --------------------------------------------------

    del X_train
    del y_train
    del X_validation
    del y_validation

    gc.collect()

    # --------------------------------------------------
    # TEST DATA
    # --------------------------------------------------

    X_test = df.loc[
        test_mask,
        features,
    ]

    y_test = (
        df.loc[
            test_mask,
            target,
        ]
        .reset_index(drop=True)
    )

    test_station_ids = (
        X_test["station_id"]
        .astype(str)
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------

    test_predictions = (
        model.predict(
            X_test
        )
    )

    test_predictions = (
        pd.Series(
            test_predictions
        )
        .clip(lower=0)
    )

    output = df.loc[
        test_mask,
        ["station_id", "datetime", target],
    ].copy()

    output[f"predicted_{target}"] = (
        test_predictions.values
    )

    output.to_csv(
        f"data/processed/test_{target}_predictions.csv",
        index=False,
    )

    test_mae, test_wape = (
        calculate_metrics(
            y_test,
            test_predictions,
        )
    )

    # --------------------------------------------------
    # MAIN RESULTS
    # --------------------------------------------------

    print(
        f"\n{target.upper()}"
    )

    print("\nVALIDATION")

    print(
        f"MAE: "
        f"{validation_mae:.3f}"
    )

    print(
        f"WAPE: "
        f"{validation_wape:.3f}"
    )

    print("\nTEST")

    print(
        f"MAE: "
        f"{test_mae:.3f}"
    )

    print(
        f"WAPE: "
        f"{test_wape:.3f}"
    )

    print(
        f"\nBest iteration: "
        f"{model.best_iteration_}"
    )

    # --------------------------------------------------
    # BUSY VS LOW-VOLUME STATIONS
    # --------------------------------------------------

    evaluate_station_groups(
        station_average,
        test_station_ids,
        y_test,
        test_predictions,
    )

    # --------------------------------------------------
    # CLEAN MEMORY BEFORE NEXT MODEL
    # --------------------------------------------------

    del df
    del X_test
    del y_test
    del test_predictions
    del test_station_ids
    del station_average
    del model

    gc.collect()


# --------------------------------------------------
# 6. RUN DEPARTURES, THEN ARRIVALS
# --------------------------------------------------

def main():

    run_model(
        "departures"
    )

    run_model(
        "arrivals"
    )


if __name__ == "__main__":
    main()