import pandas as pd
import requests
import holidays


WEATHER_FILE = (
    "data/processed/tomorrow_weather.csv"
)

STATION_PROFILE_FILE = (
    "data/processed/station_demand_profile.csv"
)

NETWORK_PROFILE_FILE = (
    "data/processed/network_demand_profile.csv"
)

OUTPUT_FILE = (
    "data/processed/tomorrow_features.csv"
)


STATUS_URL = (
    "https://gbfs.bluebikes.com/gbfs/en/station_status.json"
)

INFO_URL = (
    "https://gbfs.bluebikes.com/gbfs/en/station_information.json"
)


def get_active_stations():
    """Get active stations and map live GBFS IDs to historical short IDs."""

    status = requests.get(
        STATUS_URL,
        timeout=20,
    ).json()["data"]["stations"]

    info = requests.get(
        INFO_URL,
        timeout=20,
    ).json()["data"]["stations"]

    status_df = pd.DataFrame(status)

    info_df = pd.DataFrame(info)

    stations = status_df.merge(
        info_df[
            [
                "station_id",
                "short_name",
                "name",
            ]
        ],
        on="station_id",
        how="left",
    )

    stations = stations[
        (stations["is_installed"] == 1)
        & (stations["is_renting"] == 1)
        & (stations["is_returning"] == 1)
    ].copy()

    # Keep the live GBFS ID for live inventory joins later
    stations = stations.rename(
        columns={
            "station_id": "live_station_id",
            "short_name": "station_id",
        }
    )

    return stations[
        [
            "station_id",
            "live_station_id",
            "name",
        ]
    ]


def main():

    print("Loading tomorrow's weather...")

    weather = pd.read_csv(
        WEATHER_FILE,
        parse_dates=["datetime"],
    )

    print("Loading active Bluebikes stations...")

    stations = get_active_stations()

    print(
        f"Active stations: {len(stations)}"
    )

    # 1. Create station × hour combinations
    tomorrow = stations.merge(
        weather,
        how="cross",
    )

    # 2. Calendar features
    tomorrow["hour"] = (
        tomorrow["datetime"].dt.hour
    )

    tomorrow["day_of_week"] = (
        tomorrow["datetime"].dt.dayofweek
    )

    tomorrow["month"] = (
        tomorrow["datetime"].dt.month
    )

    tomorrow["is_weekend"] = (
        tomorrow["day_of_week"] >= 5
    ).astype(int)

    year = tomorrow["datetime"].dt.year.iloc[0]

    ma_holidays = holidays.US(
        subdiv="MA",
        years=[year],
    )

    tomorrow["is_holiday"] = (
        tomorrow["datetime"]
        .dt.date
        .isin(ma_holidays)
        .astype(int)
    )

    # 3. Load historical demand profiles
    station_profile = pd.read_csv(
        STATION_PROFILE_FILE
    )

    network_profile = pd.read_csv(
        NETWORK_PROFILE_FILE
    )

    # 4. Add station historical behavior
    tomorrow = tomorrow.merge(
        station_profile,
        on=[
            "station_id",
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    # 5. Add network fallback
    tomorrow = tomorrow.merge(
        network_profile,
        on=[
            "day_of_week",
            "hour",
        ],
        how="left",
    )

    tomorrow["historical_departures"] = (
        tomorrow["historical_departures"]
        .fillna(
            tomorrow["network_departures"]
        )
    )

    tomorrow["historical_arrivals"] = (
        tomorrow["historical_arrivals"]
        .fillna(
            tomorrow["network_arrivals"]
        )
    )

    # 6. Save model-ready tomorrow features
    tomorrow.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Tomorrow prediction rows: "
        f"{len(tomorrow):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()