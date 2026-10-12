from pathlib import Path

import pandas as pd


# Trip files used for the first version of the dataset
TRIP_FILES = sorted(
    Path("data/raw").glob("*-bluebikes-tripdata.csv")
)


def load_trips():
    """Load the trip columns needed for demand forecasting."""

    columns = [
        "started_at",
        "ended_at",
        "start_station_id",
        "end_station_id",
    ]

    monthly_data = []

    for file_path in TRIP_FILES:
        print(f"Loading {file_path.name}...")

        month_df = pd.read_csv(
            file_path,
            usecols=columns,
        )

        monthly_data.append(month_df)

    trips = pd.concat(
        monthly_data,
        ignore_index=True,
    )

    trips["started_at"] = pd.to_datetime(
    trips["started_at"],
    format="mixed",
)

    trips["ended_at"] = pd.to_datetime(
        trips["ended_at"],
        format="mixed",
    )

    return trips


def build_hourly_counts(trips):
    """Count departures and arrivals for each station and hour."""

    departures = (
        trips
        .dropna(subset=["start_station_id"])
        .assign(
            datetime=trips["started_at"].dt.floor("h")
        )
        .groupby(
            ["start_station_id", "datetime"]
        )
        .size()
        .reset_index(name="departures")
        .rename(
            columns={
                "start_station_id": "station_id"
            }
        )
    )

    arrivals = (
        trips
        .dropna(subset=["end_station_id"])
        .assign(
            datetime=trips["ended_at"].dt.floor("h")
        )
        .groupby(
            ["end_station_id", "datetime"]
        )
        .size()
        .reset_index(name="arrivals")
        .rename(
            columns={
                "end_station_id": "station_id"
            }
        )
    )

    hourly = departures.merge(
        arrivals,
        on=["station_id", "datetime"],
        how="outer",
    )

    hourly[["departures", "arrivals"]] = (
        hourly[["departures", "arrivals"]]
        .fillna(0)
        .astype(int)
    )

    return hourly


def add_zero_demand_hours(hourly):
    """Add zero-demand hours within months that actually contain trip data."""

    hourly = hourly.copy()

    hourly["month"] = (
        hourly["datetime"]
        .dt.to_period("M")
    )

    station_hours = []

    for (station_id, month), station_df in hourly.groupby(
        ["station_id", "month"]
    ):
        first_hour = station_df["datetime"].min()
        last_hour = station_df["datetime"].max()

        hours = pd.date_range(
            start=first_hour,
            end=last_hour,
            freq="h",
        )

        station_grid = pd.DataFrame(
            {
                "station_id": station_id,
                "datetime": hours,
            }
        )

        station_hours.append(station_grid)

    complete_grid = pd.concat(
        station_hours,
        ignore_index=True,
    )

    complete_hourly = complete_grid.merge(
        hourly.drop(columns="month"),
        on=["station_id", "datetime"],
        how="left",
    )

    complete_hourly[["departures", "arrivals"]] = (
        complete_hourly[["departures", "arrivals"]]
        .fillna(0)
        .astype(int)
    )

    return complete_hourly


def main():
    trips = load_trips()

    hourly = build_hourly_counts(trips)

    hourly = add_zero_demand_hours(hourly)

    output_path = Path(
        "data/processed/station_hour_demand.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly.to_csv(
        output_path,
        index=False,
    )

    print("\nStation-hour dataset created.")
    print(f"Rows: {len(hourly):,}")
    print(f"Stations: {hourly['station_id'].nunique()}")
    print(f"Saved to: {output_path}")

    print("\nFirst 10 rows:")
    print(hourly.head(10).to_string(index=False))


if __name__ == "__main__":
    main()