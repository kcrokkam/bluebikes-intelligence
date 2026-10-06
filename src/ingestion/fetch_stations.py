import requests
import pandas as pd

from datetime import datetime
from pathlib import Path

STATUS_URL = "https://gbfs.bluebikes.com/gbfs/en/station_status.json"
INFO_URL = "https://gbfs.bluebikes.com/gbfs/en/station_information.json"


def fetch_station_status():
    response = requests.get(STATUS_URL)
    data = response.json()

    stations = data["data"]["stations"]

    return pd.DataFrame(stations)


def fetch_station_information():
    response = requests.get(INFO_URL)
    data = response.json()

    stations = data["data"]["stations"]

    return pd.DataFrame(stations)


def build_station_snapshot():
    status_df = fetch_station_status()
    info_df = fetch_station_information()

    merged_df = status_df.merge(
        info_df,
        on="station_id",
        how="left"
    )

    merged_df["bike_availability_ratio"] = (
        merged_df["num_bikes_available"] / merged_df["capacity"]
    )

    return merged_df


def get_critical_stations(df):
    active_stations = df[
        (df["is_installed"] == 1)
        & (df["is_renting"] == 1)
        & (df["is_returning"] == 1)
    ]

    critical_stations = active_stations[
        active_stations["bike_availability_ratio"] <= 0.10
    ]

    return critical_stations


station_snapshot = build_station_snapshot()

output_dir = Path("data/raw")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_file = output_dir / f"station_snapshot_{timestamp}.csv"

station_snapshot.to_csv(output_file, index=False)

print(f"Saved snapshot to {output_file}")

critical_stations = get_critical_stations(station_snapshot)

print(
    critical_stations[
        [
            "name",
            "num_bikes_available",
            "capacity",
            "bike_availability_ratio",
        ]
    ]
    .sort_values("bike_availability_ratio")
    .head(10)
)