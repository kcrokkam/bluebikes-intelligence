import pandas as pd


DATA_FILE = "data/processed/station_hour_features.csv"


# Chronological split
TRAIN_END = "2026-02-28 23:00:00"

VALIDATION_START = "2026-03-01 00:00:00"
VALIDATION_END = "2026-05-31 23:00:00"

TEST_START = "2026-06-01 00:00:00"


def load_data():
    """Load only the columns needed for the baseline."""

    columns = [
        "station_id",
        "datetime",
        "departures",
        "arrivals",
        "hour",
        "day_of_week",
    ]

    df = pd.read_csv(
        DATA_FILE,
        usecols=columns,
    )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    return df


def create_splits(df):
    """Split data chronologically into train, validation, and test."""

    train = df[
        df["datetime"] <= TRAIN_END
    ].copy()

    validation = df[
        (df["datetime"] >= VALIDATION_START)
        & (df["datetime"] <= VALIDATION_END)
    ].copy()

    test = df[
        df["datetime"] >= TEST_START
    ].copy()

    return train, validation, test


def build_historical_average(train):
    """Calculate average demand by station, weekday, and hour."""

    averages = (
        train
        .groupby(
            [
                "station_id",
                "day_of_week",
                "hour",
            ]
        )[
            [
                "departures",
                "arrivals",
            ]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "departures": "predicted_departures",
                "arrivals": "predicted_arrivals",
            }
        )
    )

    return averages


def make_predictions(df, averages, train):
    """Attach historical-average predictions."""

    predictions = df.merge(
        averages,
        on=[
            "station_id",
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    # Fallback for station/time combinations
    # that were not seen during training.
    mean_departures = train["departures"].mean()
    mean_arrivals = train["arrivals"].mean()

    predictions["predicted_departures"] = (
        predictions["predicted_departures"]
        .fillna(mean_departures)
    )

    predictions["predicted_arrivals"] = (
        predictions["predicted_arrivals"]
        .fillna(mean_arrivals)
    )

    return predictions


def calculate_metrics(df, actual, predicted):
    """Calculate MAE and WAPE."""

    absolute_error = (
        df[actual] - df[predicted]
    ).abs()

    mae = absolute_error.mean()

    wape = (
        absolute_error.sum()
        / df[actual].sum()
    )

    return mae, wape


def evaluate(name, df):
    """Print baseline metrics."""

    departure_mae, departure_wape = calculate_metrics(
        df,
        "departures",
        "predicted_departures",
    )

    arrival_mae, arrival_wape = calculate_metrics(
        df,
        "arrivals",
        "predicted_arrivals",
    )

    print(f"\n{name}")

    print(
        f"Departures MAE: {departure_mae:.3f}"
    )

    print(
        f"Departures WAPE: {departure_wape:.3f}"
    )

    print(
        f"Arrivals MAE: {arrival_mae:.3f}"
    )

    print(
        f"Arrivals WAPE: {arrival_wape:.3f}"
    )


def main():
    print("Loading feature dataset...")

    df = load_data()

    train, validation, test = create_splits(
        df
    )

    print(f"\nTrain rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    averages = build_historical_average(
        train
    )

    validation = make_predictions(
        validation,
        averages,
        train,
    )

    test = make_predictions(
        test,
        averages,
        train,
    )

    evaluate(
        "VALIDATION",
        validation,
    )

    evaluate(
        "TEST",
        test,
    )


if __name__ == "__main__":
    main()