import pandas as pd

from lightgbm import LGBMRegressor


HISTORY_FILE = (
    "data/processed/station_hour_features.csv"
)

STATION_PROFILE_FILE = (
    "data/processed/station_demand_profile.csv"
)

NETWORK_PROFILE_FILE = (
    "data/processed/network_demand_profile.csv"
)

TOMORROW_FILE = (
    "data/processed/tomorrow_features.csv"
)

OUTPUT_FILE = (
    "data/processed/tomorrow_predictions.csv"
)


TRAIN_START = "2025-09-01 00:00:00"


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


def load_training_data():
    """Prepare historical data using the same features as tomorrow."""

    print("Loading historical training data...")

    history = pd.read_csv(
        HISTORY_FILE,
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

    history = history[
        history["datetime"] >= TRAIN_START
    ].copy()

    station_profile = pd.read_csv(
        STATION_PROFILE_FILE
    )

    network_profile = pd.read_csv(
        NETWORK_PROFILE_FILE
    )

    history = history.merge(
        station_profile,
        on=[
            "station_id",
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    history = history.merge(
        network_profile,
        on=[
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    history["historical_departures"] = (
        history["historical_departures"]
        .fillna(history["network_departures"])
    )

    history["historical_arrivals"] = (
        history["historical_arrivals"]
        .fillna(history["network_arrivals"])
    )

    return history


def train_model(history, target):
    """Train one final demand model."""

    print(f"Training {target} model...")

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
        history[FEATURES],
        history[target],
    )

    return model


def main():

    history = load_training_data()

    print(
        f"Training rows: {len(history):,}"
    )

    tomorrow = pd.read_csv(
        TOMORROW_FILE,
        parse_dates=["datetime"],
    )

    departures_model = train_model(
        history,
        "departures",
    )

    arrivals_model = train_model(
        history,
        "arrivals",
    )

    tomorrow["predicted_departures"] = (
        departures_model
        .predict(tomorrow[FEATURES])
        .clip(min=0)
    )

    tomorrow["predicted_arrivals"] = (
        arrivals_model
        .predict(tomorrow[FEATURES])
        .clip(min=0)
    )

    tomorrow["predicted_net_flow"] = (
        tomorrow["predicted_arrivals"]
        - tomorrow["predicted_departures"]
    )

    tomorrow.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTomorrow predictions created.")

    print(
        f"Rows: {len(tomorrow):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()