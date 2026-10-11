from pathlib import Path
import json

import pandas as pd


def load_inventory_snapshot(file_path):
    """Load and clean one Bluebikes station-status JSON file."""

    # 1. Open the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # 2. Extract station records
    stations = data["data"]["stations"]

    # 3. Convert station records to a DataFrame
    stations_df = pd.DataFrame(stations)

    # 4. Convert the Bluebikes feed timestamp
    feed_time = pd.to_datetime(
        data["last_updated"],
        unit="s",
        utc=True,
    )

    stations_df["feed_last_updated"] = feed_time

    # 5. Convert each station's last reported timestamp
    stations_df["station_last_reported"] = pd.to_datetime(
        stations_df["last_reported"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    # 6. Calculate how old each station report is
    stations_df["report_age_minutes"] = (
        stations_df["feed_last_updated"]
        - stations_df["station_last_reported"]
    ).dt.total_seconds() / 60

    # 7. Keep only operational stations
    active_df = stations_df[
        (stations_df["is_installed"] == 1)
        & (stations_df["is_renting"] == 1)
    ].copy()

    active_df = active_df.reset_index(drop=True)

    return active_df


def build_inventory_dataset(snapshot_dir):
    """Load all snapshots and combine them into one dataset."""

    # 8. Find all station-status JSON files
    snapshot_dir = Path(snapshot_dir)

    snapshot_files = sorted(
        snapshot_dir.rglob("station_status_*.json")
    )

    print("Snapshots found:", len(snapshot_files))

    # 9. Load every snapshot
    snapshot_dfs = []

    for file_path in snapshot_files:
        snapshot_df = load_inventory_snapshot(file_path)
        snapshot_dfs.append(snapshot_df)

    # 10. Combine all snapshots
    inventory_df = pd.concat(
        snapshot_dfs,
        ignore_index=True,
    )

    return inventory_df


def add_stockout_labels(inventory_df):
    """
    Create stockout_next_60min.

    1 = station reaches 0 bikes within the next 60 minutes
    0 = station does not reach 0 bikes within the next 60 minutes
    NA = a full 60 minutes of future data is not available
    """

    # 11. Sort observations by station and time
    inventory_df = inventory_df.sort_values(
        ["station_id", "feed_last_updated"]
    ).copy()

    inventory_df["stockout_next_60min"] = pd.NA

    # 12. Process one station at a time
    for station_id, station_df in inventory_df.groupby("station_id"):

        station_end_time = station_df[
            "feed_last_updated"
        ].max()

        for index, row in station_df.iterrows():

            current_time = row["feed_last_updated"]
            current_bikes = row["num_bikes_available"]

            # Skip stations that are already stocked out
            if current_bikes == 0:
                continue

            future_end_time = (
                current_time
                + pd.Timedelta(minutes=60)
            )

            # Not enough future data to create a reliable label
            if station_end_time < future_end_time:
                continue

            # Look only at snapshots after the current time
            # and within the next 60 minutes
            future_rows = station_df[
                (station_df["feed_last_updated"] > current_time)
                & (
                    station_df["feed_last_updated"]
                    <= future_end_time
                )
            ]

            # Did the station hit 0 bikes?
            stockout_occurred = (
                future_rows["num_bikes_available"] == 0
            ).any()

            inventory_df.loc[
                index,
                "stockout_next_60min",
            ] = int(stockout_occurred)

    return inventory_df


def get_station_history(inventory_df):
    """Choose a changing station and inspect its inventory history."""

    # 13. Find stations whose inventory changed frequently
    station_variation = (
        inventory_df
        .groupby("station_id")["num_bikes_available"]
        .nunique()
        .sort_values(ascending=False)
    )

    example_station = station_variation.index[0]

    # 14. Keep one station
    station_history = inventory_df[
        inventory_df["station_id"] == example_station
    ].copy()

    station_history = station_history.sort_values(
        "feed_last_updated"
    )

    # 15. Calculate inventory change from previous snapshot
    station_history["inventory_change"] = (
        station_history["num_bikes_available"].diff()
    )

    return example_station, station_history


def validate_positive_label(inventory_df):
    """Inspect one positive label to confirm the station really hit 0."""

    # 16. Find one row labelled as a future stockout
    positive_rows = inventory_df[
        inventory_df["stockout_next_60min"] == 1
    ]

    if positive_rows.empty:
        print("\nNo positive stockout labels found.")
        return

    positive_row = positive_rows.iloc[0]

    station_id = positive_row["station_id"]
    prediction_time = positive_row["feed_last_updated"]

    future_end_time = (
        prediction_time
        + pd.Timedelta(minutes=60)
    )

    # 17. Show the station from prediction time
    # through the following 60 minutes
    validation_window = inventory_df[
        (inventory_df["station_id"] == station_id)
        & (
            inventory_df["feed_last_updated"]
            >= prediction_time
        )
        & (
            inventory_df["feed_last_updated"]
            <= future_end_time
        )
    ].sort_values("feed_last_updated")

    print("\nPositive-label validation:")
    print("Station:", station_id)
    print("Prediction time:", prediction_time)

    columns_to_show = [
        "feed_last_updated",
        "num_bikes_available",
        "stockout_next_60min",
    ]

    print(
        validation_window[
            columns_to_show
        ].to_string(index=False)
    )


def main():
    # 18. Build historical inventory dataset
    inventory_df = build_inventory_dataset(
        "data/snapshots"
    )

    # 19. Create future-stockout labels
    inventory_df = add_stockout_labels(
        inventory_df
    )

    print("\nCombined shape:", inventory_df.shape)

    print(
        "Unique snapshot times:",
        inventory_df["feed_last_updated"].nunique(),
    )

    # 20. Show stockout label distribution
    print("\nStockout label counts:")

    print(
        inventory_df["stockout_next_60min"]
        .value_counts(dropna=False)
    )

    # 21. Inspect one station's inventory movement
    example_station, station_history = (
        get_station_history(inventory_df)
    )

    print("\nExample station:")
    print(example_station)

    columns_to_show = [
        "feed_last_updated",
        "num_bikes_available",
        "num_docks_available",
        "inventory_change",
    ]

    print("\nStation inventory history:")

    print(
        station_history[
            columns_to_show
        ].to_string(index=False)
    )

    # 22. Verify that a positive label is correct
    validate_positive_label(
        inventory_df
    )


if __name__ == "__main__":
    main()