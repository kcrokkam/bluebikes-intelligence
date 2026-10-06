import pandas as pd


# 1. DEFINE THE TRIP FILES

trip_files = [
    "data/raw/202606-bluebikes-tripdata.csv",
    "data/raw/202607-bluebikes-tripdata.csv",
    "data/raw/202608-bluebikes-tripdata.csv"
]


# 2. LOAD EACH MONTH

monthly_data = []

for file_path in trip_files:

    print(f"Loading {file_path}...")

    month_df = pd.read_csv(file_path)

    monthly_data.append(month_df)


# 3. COMBINE ALL MONTHS INTO ONE DATAFRAME

df = pd.concat(
    monthly_data,
    ignore_index=True
)

print("Total trip rows:", len(df))


# 4. CONVERT TIME COLUMNS FROM TEXT TO DATETIME

df["started_at"] = pd.to_datetime(df["started_at"])
df["ended_at"] = pd.to_datetime(df["ended_at"])


# 5. CHECK THE DATE RANGE

print("Earliest trip:", df["started_at"].min())
print("Latest trip:", df["started_at"].max())


# 6. CREATE 30-MINUTE TIME WINDOWS
# Example:
# 08:07 -> 08:00
# 08:22 -> 08:00
# 08:41 -> 08:30

df["start_time_window"] = (
    df["started_at"]
    .dt.floor("30min")
)

df["end_time_window"] = (
    df["ended_at"]
    .dt.floor("30min")
)


# 7. COUNT DEPARTURES FOR EACH STATION
# IN EACH 30-MINUTE WINDOW

departures = (
    df
    .dropna(subset=["start_station_id"])
    .groupby(
        [
            "start_station_id",
            "start_time_window"
        ]
    )
    .size()
    .reset_index(name="departures")
)


# 8. COUNT ARRIVALS FOR EACH STATION
# IN EACH 30-MINUTE WINDOW

arrivals = (
    df
    .dropna(subset=["end_station_id"])
    .groupby(
        [
            "end_station_id",
            "end_time_window"
        ]
    )
    .size()
    .reset_index(name="arrivals")
)


# 9. RENAME COLUMNS SO BOTH TABLES
# USE THE SAME COLUMN NAMES

departures = departures.rename(
    columns={
        "start_station_id": "station_id",
        "start_time_window": "time_window"
    }
)

arrivals = arrivals.rename(
    columns={
        "end_station_id": "station_id",
        "end_time_window": "time_window"
    }
)


# 10. COMBINE DEPARTURES AND ARRIVALS

station_flow = departures.merge(
    arrivals,
    on=[
        "station_id",
        "time_window"
    ],
    how="outer"
)


# 11. REPLACE MISSING COUNTS WITH ZERO

station_flow[
    [
        "departures",
        "arrivals"
    ]
] = (
    station_flow[
        [
            "departures",
            "arrivals"
        ]
    ]
    .fillna(0)
    .astype(int)
)


# 12. CALCULATE NET BIKE FLOW
# Positive = station gained bikes
# Negative = station lost bikes

station_flow["net_flow"] = (
    station_flow["arrivals"]
    - station_flow["departures"]
)


# 13. CREATE A COMPLETE 30-MINUTE TIMELINE
# FOR EACH STATION

complete_station_data = []

for station_id, station_data in station_flow.groupby("station_id"):

    station_start = (
        station_data["time_window"].min()
    )

    station_end = (
        station_data["time_window"].max()
    )

    station_time_windows = pd.date_range(
        start=station_start,
        end=station_end,
        freq="30min"
    )

    station_index = pd.DataFrame({
        "station_id": station_id,
        "time_window": station_time_windows
    })

    station_complete = station_index.merge(
        station_data,
        on=[
            "station_id",
            "time_window"
        ],
        how="left"
    )

    complete_station_data.append(
        station_complete
    )


# 14. COMBINE ALL STATIONS BACK
# INTO ONE DATAFRAME

station_flow = pd.concat(
    complete_station_data,
    ignore_index=True
)


# 15. FILL MISSING TRIP COUNTS WITH ZERO

station_flow[
    [
        "departures",
        "arrivals",
        "net_flow"
    ]
] = (
    station_flow[
        [
            "departures",
            "arrivals",
            "net_flow"
        ]
    ]
    .fillna(0)
    .astype(int)
)


# 16. CREATE TIME-BASED FEATURES

station_flow["hour"] = (
    station_flow["time_window"].dt.hour
)

station_flow["time_slot"] = (
    station_flow["time_window"].dt.hour * 2
    + station_flow["time_window"].dt.minute // 30
)

station_flow["day_of_week"] = (
    station_flow["time_window"].dt.dayofweek
)

station_flow["is_weekend"] = (
    station_flow["day_of_week"] >= 5
).astype(int)


# 17. CREATE LAG FEATURES
# lag_1 = departures from 30 minutes ago
# lag_2 = departures from 60 minutes ago

station_flow["departures_lag_1"] = (
    station_flow
    .groupby("station_id")["departures"]
    .shift(1)
)

station_flow["departures_lag_2"] = (
    station_flow
    .groupby("station_id")["departures"]
    .shift(2)
)


# 18. CREATE A ROLLING AVERAGE FEATURE
# Average departures over the previous
# 3 time windows = previous 90 minutes

station_flow["departures_rolling_3"] = (
    station_flow
    .groupby("station_id")["departures"]
    .transform(
        lambda x: (
            x
            .shift(1)
            .rolling(3)
            .mean()
        )
    )
)


# 19. INSPECT ONE STATION
# TO CHECK THE LAG FEATURES

example_station = station_flow[
    station_flow["station_id"] == "A23097"
]

print(
    example_station[
        [
            "time_window",
            "departures",
            "departures_lag_1",
            "departures_lag_2",
            "departures_rolling_3"
        ]
    ].head(15)
)


# 20. SAVE THE PROCESSED DATA

output_file = (
    "data/processed/"
    "station_flow_202606_202608.csv"
)

station_flow.to_csv(
    output_file,
    index=False
)

print(
    f"Saved processed data to {output_file}"
)