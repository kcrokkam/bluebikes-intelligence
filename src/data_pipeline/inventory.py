from pathlib import Path
import json

import pandas as pd


def load_inventory_snapshot(file_path):
    """Load and clean one Bluebikes station-status snapshot."""

    with open(file_path, "r") as file:
        data = json.load(file)

    stations_df = pd.DataFrame(
        data["data"]["stations"]
    )

    stations_df["feed_last_updated"] = pd.to_datetime(
        data["last_updated"],
        unit="s",
        utc=True,
    )

    stations_df["station_last_reported"] = pd.to_datetime(
        stations_df["last_reported"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    stations_df["report_age_minutes"] = (
        stations_df["feed_last_updated"]
        - stations_df["station_last_reported"]
    ).dt.total_seconds() / 60

    # Keep only operational stations
    stations_df = stations_df[
        (stations_df["is_installed"] == 1)
        & (stations_df["is_renting"] == 1)
    ].copy()

    return stations_df


def build_inventory_dataset(snapshot_dir):
    """Combine all inventory snapshots into one dataset."""

    snapshot_files = sorted(
        Path(snapshot_dir).rglob(
            "station_status_*.json"
        )
    )

    print("Snapshots found:", len(snapshot_files))

    snapshots = [
        load_inventory_snapshot(file_path)
        for file_path in snapshot_files
    ]

    inventory_df = pd.concat(
        snapshots,
        ignore_index=True,
    )

    return inventory_df


def add_inventory_features(inventory_df):
    """Create simple inventory movement features."""

    inventory_df = inventory_df.sort_values(
        ["station_id", "feed_last_updated"]
    ).copy()

    inventory_df["previous_bikes"] = (
        inventory_df
        .groupby("station_id")["num_bikes_available"]
        .shift(1)
    )

    inventory_df["inventory_change"] = (
        inventory_df["num_bikes_available"]
        - inventory_df["previous_bikes"]
    )

    return inventory_df


def add_stockout_labels(inventory_df):
    """Create the 60-minute future stockout target."""

    inventory_df = inventory_df.sort_values(
        ["station_id", "feed_last_updated"]
    ).copy()

    inventory_df["stockout_next_60min"] = pd.NA

    for station_id, station_df in inventory_df.groupby(
        "station_id"
    ):
        station_end_time = (
            station_df["feed_last_updated"].max()
        )

        for index, row in station_df.iterrows():

            current_time = row["feed_last_updated"]
            current_bikes = row["num_bikes_available"]

            # Already stocked out
            if current_bikes == 0:
                continue

            future_end = (
                current_time
                + pd.Timedelta(minutes=60)
            )

            # Not enough future data
            if station_end_time < future_end:
                continue

            future_rows = station_df[
                (station_df["feed_last_updated"] > current_time)
                & (
                    station_df["feed_last_updated"]
                    <= future_end
                )
            ]

            stockout = (
                future_rows["num_bikes_available"] == 0
            ).any()

            inventory_df.loc[
                index,
                "stockout_next_60min",
            ] = int(stockout)

    return inventory_df


def main():
    # 1. Load historical inventory
    inventory_df = build_inventory_dataset(
        "data/snapshots"
    )

    # 2. Create prediction features
    inventory_df = add_inventory_features(
        inventory_df
    )

    # 3. Create prediction target
    inventory_df = add_stockout_labels(
        inventory_df
    )

    # 4. Basic checks
    print("\nDataset shape:", inventory_df.shape)

    print("\nStockout labels:")
    print(
        inventory_df["stockout_next_60min"]
        .value_counts(dropna=False)
    )


if __name__ == "__main__":
    main()