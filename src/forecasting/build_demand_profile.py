import pandas as pd


INPUT_FILE = "data/processed/station_hour_features.csv"

STATION_OUTPUT_FILE = (
    "data/processed/station_demand_profile.csv"
)

NETWORK_OUTPUT_FILE = (
    "data/processed/network_demand_profile.csv"
)

TRAIN_END = "2025-08-31 23:00:00"


def main():

    print("Loading historical demand data...")

    df = pd.read_csv(
        INPUT_FILE,
        usecols=[
            "station_id",
            "datetime",
            "hour",
            "day_of_week",
            "departures",
            "arrivals",
        ],
        parse_dates=["datetime"],
    )

    # 1. Use only historical training data
    train = df[
        df["datetime"] <= TRAIN_END
    ].copy()

    # 2. Build station-specific demand profile
    station_profile = (
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
                "departures": "historical_departures",
                "arrivals": "historical_arrivals",
            }
        )
    )

    station_profile.to_csv(
        STATION_OUTPUT_FILE,
        index=False,
    )

    # 3. Build network-level fallback profile
    # Used for new stations with no historical record
    network_profile = (
        train
        .groupby(
            [
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
                "departures": "network_departures",
                "arrivals": "network_arrivals",
            }
        )
    )

    network_profile.to_csv(
        NETWORK_OUTPUT_FILE,
        index=False,
    )

    # 4. Print results
    print("\nDemand profiles created.")

    print(
        f"Station profile rows: "
        f"{len(station_profile):,}"
    )

    print(
        f"Station profile stations: "
        f"{station_profile['station_id'].nunique()}"
    )

    print(
        f"Saved to: "
        f"{STATION_OUTPUT_FILE}"
    )

    print(
        f"\nNetwork profile rows: "
        f"{len(network_profile):,}"
    )

    print(
        f"Saved to: "
        f"{NETWORK_OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()