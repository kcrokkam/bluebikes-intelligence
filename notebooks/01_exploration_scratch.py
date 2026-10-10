# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt


# 2. LOAD THE PROCESSED DATA

file_path = "data/processed/station_flow_202606_202608.csv"

df = pd.read_csv(
    file_path,
    parse_dates=["time_window"]
)


# 3. CHECK THE DATASET

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Stations:", df["station_id"].nunique())
print("Start:", df["time_window"].min())
print("End:", df["time_window"].max())

print("\nColumns:")
print(df.columns.tolist())

# 4. CHECK HOW COMMON ZERO-DEPARTURE WINDOWS ARE

zero_departure_rows = (
    df["departures"] == 0
).sum()

total_rows = len(df)

zero_departure_pct = (
    zero_departure_rows / total_rows
) * 100


print("\nZERO-DEPARTURE ANALYSIS")

print(
    "Zero-departure rows:",
    zero_departure_rows
)

print(
    "Total rows:",
    total_rows
)

print(
    "Percent with zero departures:",
    round(zero_departure_pct, 2),
    "%"
)

# 5. COMPARE AGAINST AN "ALWAYS ZERO" PREDICTION

always_zero_mae = (
    df["departures"].abs().mean()
)

active_rows = df[
    df["departures"] > 0
]

busy_rows = df[
    df["departures"] >= 3
]


print("\nDEMAND DISTRIBUTION")

print(
    "Always-zero MAE:",
    round(always_zero_mae, 3)
)

print(
    "Percent of rows with departures > 0:",
    round(
        len(active_rows) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Percent of rows with departures >= 3:",
    round(
        len(busy_rows) / len(df) * 100,
        2
    ),
    "%"
)

print(
    "Average departures when active:",
    round(
        active_rows["departures"].mean(),
        2
    )
)

print(
    "Median departures when active:",
    active_rows["departures"].median()
)

# 6. CALCULATE DEMAND BY 30-MINUTE TIME SLOT

demand_by_time = (
    df
    .groupby("time_slot")
    .agg(
        average_departures=("departures", "mean"),
        active_rate=("departures", lambda x: (x > 0).mean())
    )
    .reset_index()
)


# CONVERT TIME SLOT BACK INTO READABLE TIME

demand_by_time["hour"] = (
    demand_by_time["time_slot"] // 2
)

demand_by_time["minute"] = (
    demand_by_time["time_slot"] % 2
) * 30

demand_by_time["time_label"] = (
    demand_by_time["hour"].astype(str).str.zfill(2)
    + ":"
    + demand_by_time["minute"].astype(str).str.zfill(2)
)


print("\nBUSIEST TIME SLOTS")

print(
    demand_by_time
    .sort_values(
        "average_departures",
        ascending=False
    )
    .head(10)
    [
        [
            "time_label",
            "average_departures",
            "active_rate"
        ]
    ]
)

# 7. COMPARE WEEKDAY AND WEEKEND DEMAND

demand_by_day_type = (
    df
    .groupby(
        [
            "is_weekend",
            "time_slot"
        ]
    )["departures"]
    .mean()
    .reset_index()
)


# LABEL WEEKDAY VS WEEKEND

demand_by_day_type["day_type"] = (
    demand_by_day_type["is_weekend"]
    .map({
        0: "Weekday",
        1: "Weekend"
    })
)


# SHOW THE 5 BUSIEST WEEKDAY PERIODS

weekday_demand = (
    demand_by_day_type[
        demand_by_day_type["day_type"] == "Weekday"
    ]
    .sort_values(
        "departures",
        ascending=False
    )
    .head(5)
)


# SHOW THE 5 BUSIEST WEEKEND PERIODS

weekend_demand = (
    demand_by_day_type[
        demand_by_day_type["day_type"] == "Weekend"
    ]
    .sort_values(
        "departures",
        ascending=False
    )
    .head(5)
)


print("\nBUSIEST WEEKDAY TIME SLOTS")
print(weekday_demand)


print("\nBUSIEST WEEKEND TIME SLOTS")
print(weekend_demand)

# 8. PLOT WEEKDAY VS WEEKEND DEMAND

weekday_curve = (
    demand_by_day_type[
        demand_by_day_type["day_type"] == "Weekday"
    ]
)

weekend_curve = (
    demand_by_day_type[
        demand_by_day_type["day_type"] == "Weekend"
    ]
)


plt.figure(figsize=(12, 6))

plt.plot(
    weekday_curve["time_slot"],
    weekday_curve["departures"],
    label="Weekday"
)

plt.plot(
    weekend_curve["time_slot"],
    weekend_curve["departures"],
    label="Weekend"
)

plt.xlabel("30-minute time slot")
plt.ylabel("Average departures")
plt.title("Bluebikes demand by time of day")

plt.xticks(
    ticks=range(0, 48, 4),
    labels=[
        "00:00",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
        "14:00",
        "16:00",
        "18:00",
        "20:00",
        "22:00"
    ]
)

plt.legend()
plt.tight_layout()

# 9. COMPARE DEMAND ACROSS STATIONS

station_demand = (
    df
    .groupby("station_id")
    .agg(
        average_departures=("departures", "mean"),
        active_rate=("departures", lambda x: (x > 0).mean()),
        total_departures=("departures", "sum")
    )
    .reset_index()
)


# SHOW THE 10 BUSIEST STATIONS

busiest_stations = (
    station_demand
    .sort_values(
        "average_departures",
        ascending=False
    )
    .head(10)
)


print("\nBUSIEST STATIONS")

print(
    busiest_stations[
        [
            "station_id",
            "average_departures",
            "active_rate",
            "total_departures"
        ]
    ]
)

# 10. LOOK AT ONE BUSY STATION

station_id = "M32006"

station = df[
    df["station_id"] == station_id
].copy()

print("\nSTATION:", station_id)

print(
    "Average departures:",
    round(station["departures"].mean(), 2)
)

print(
    "Percent of active windows:",
    round(
        (station["departures"] > 0).mean() * 100,
        2
    ),
    "%"
)

# 11. DEMAND PATTERN FOR ONE STATION

station_pattern = (
    station
    .groupby(
        [
            "is_weekend",
            "time_slot"
        ]
    )["departures"]
    .mean()
    .reset_index()
)


weekday_station = station_pattern[
    station_pattern["is_weekend"] == 0
]

weekend_station = station_pattern[
    station_pattern["is_weekend"] == 1
]


plt.figure(figsize=(12, 6))

plt.plot(
    weekday_station["time_slot"],
    weekday_station["departures"],
    label="Weekday"
)

plt.plot(
    weekend_station["time_slot"],
    weekend_station["departures"],
    label="Weekend"
)

plt.xlabel("30-minute time slot")
plt.ylabel("Average departures")

plt.title(
    f"Demand pattern for station {station_id}"
)

plt.xticks(
    ticks=range(0, 48, 4),
    labels=[
        "00:00",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
        "14:00",
        "16:00",
        "18:00",
        "20:00",
        "22:00"
    ]
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "notebooks/station_M32006_demand.png"
)

print(
    "Saved chart to notebooks/station_M32006_demand.png"
)

# 12. ANALYZE NET BIKE FLOW FOR ONE STATION

station_flow_pattern = (
    station
    .groupby(
        [
            "is_weekend",
            "time_slot"
        ]
    )
    .agg(
        average_departures=("departures", "mean"),
        average_arrivals=("arrivals", "mean"),
        average_net_flow=("net_flow", "mean")
    )
    .reset_index()
)

# 13. SEPARATE WEEKDAY AND WEEKEND FLOW

weekday_flow = station_flow_pattern[
    station_flow_pattern["is_weekend"] == 0
]

weekend_flow = station_flow_pattern[
    station_flow_pattern["is_weekend"] == 1
]

# 14. PLOT NET BIKE FLOW BY TIME OF DAY

plt.figure(figsize=(12, 6))

plt.plot(
    weekday_flow["time_slot"],
    weekday_flow["average_net_flow"],
    label="Weekday"
)

plt.plot(
    weekend_flow["time_slot"],
    weekend_flow["average_net_flow"],
    label="Weekend"
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("30-minute time slot")
plt.ylabel("Average net bike flow")

plt.title(
    f"Net bike flow for station {station_id}"
)

plt.xticks(
    ticks=range(0, 48, 4),
    labels=[
        "00:00",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
        "14:00",
        "16:00",
        "18:00",
        "20:00",
        "22:00"
    ]
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "notebooks/station_M32006_net_flow.png"
)

print(
    "Saved chart to notebooks/station_M32006_net_flow.png"
)

# 15. CREATE A DATE COLUMN

station["date"] = (
    station["time_window"].dt.date
)


# 16. GET THE MOST RECENT 5 WEEKDAYS

weekday_station = station[
    station["is_weekend"] == 0
]

weekday_dates = (
    weekday_station["date"]
    .drop_duplicates()
    .sort_values()
    .tail(5)
)

print("\nWEEKDAYS BEING COMPARED:")
print(weekday_dates.tolist())

# 17. COMPARE INDIVIDUAL WEEKDAYS

plt.figure(figsize=(12, 6))

for date in weekday_dates:

    day_data = weekday_station[
        weekday_station["date"] == date
    ]

    plt.plot(
        day_data["time_slot"],
        day_data["net_flow"],
        label=str(date)
    )


plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("30-minute time slot")
plt.ylabel("Net bike flow")

plt.title(
    f"Daily net flow comparison for station {station_id}"
)

plt.xticks(
    ticks=range(0, 48, 4),
    labels=[
        "00:00",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
        "14:00",
        "16:00",
        "18:00",
        "20:00",
        "22:00"
    ]
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "notebooks/station_M32006_daily_net_flow.png"
)

print(
    "Saved chart to notebooks/station_M32006_daily_net_flow.png"
)

# 18. COMPARE THE SAME WEEKDAY ACROSS DIFFERENT WEEKS

tuesday_station = station[
    station["day_of_week"] == 1
]

tuesday_dates = (
    tuesday_station["date"]
    .drop_duplicates()
    .sort_values()
    .tail(4)
)

print("\nTUESDAYS BEING COMPARED:")
print(tuesday_dates.tolist())


plt.figure(figsize=(12, 6))

for date in tuesday_dates:

    day_data = tuesday_station[
        tuesday_station["date"] == date
    ]

    plt.plot(
        day_data["time_slot"],
        day_data["net_flow"],
        label=str(date)
    )


plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("30-minute time slot")
plt.ylabel("Net bike flow")

plt.title(
    f"Tuesday net flow comparison for station {station_id}"
)

plt.xticks(
    ticks=range(0, 48, 4),
    labels=[
        "00:00",
        "02:00",
        "04:00",
        "06:00",
        "08:00",
        "10:00",
        "12:00",
        "14:00",
        "16:00",
        "18:00",
        "20:00",
        "22:00"
    ]
)

plt.legend()
plt.tight_layout()

plt.savefig(
    "notebooks/station_M32006_tuesday_comparison.png"
)

print(
    "Saved chart to notebooks/station_M32006_tuesday_comparison.png"
)

# 19. MEASURE TUESDAY CONSISTENCY BY TIME SLOT

tuesday_consistency = (
    tuesday_station
    .groupby("time_slot")
    .agg(
        average_net_flow=("net_flow", "mean"),
        std_net_flow=("net_flow", "std"),
        positive_rate=("net_flow", lambda x: (x > 0).mean()),
        negative_rate=("net_flow", lambda x: (x < 0).mean())
    )
    .reset_index()
)

# 20. FIND CONSISTENT BIKE-LOSS PERIODS

consistent_outflow = (
    tuesday_consistency
    .sort_values(
        "negative_rate",
        ascending=False
    )
    .head(10)
)

print("\nMOST CONSISTENT TUESDAY OUTFLOW PERIODS")

print(
    consistent_outflow[
        [
            "time_slot",
            "average_net_flow",
            "std_net_flow",
            "negative_rate"
        ]
    ]
)

# 21. FIND CONSISTENT BIKE-GAIN PERIODS

consistent_inflow = (
    tuesday_consistency
    .sort_values(
        "positive_rate",
        ascending=False
    )
    .head(10)
)

print("\nMOST CONSISTENT TUESDAY INFLOW PERIODS")

print(
    consistent_inflow[
        [
            "time_slot",
            "average_net_flow",
            "std_net_flow",
            "positive_rate"
        ]
    ]
)

# 22. CREATE THE PREVIOUS 30-MINUTE NET FLOW

station["net_flow_lag_1"] = (
    station["net_flow"].shift(1)
)


# 23. CHECK WHETHER FLOW DIRECTION CONTINUES

previously_negative = station[
    station["net_flow_lag_1"] < 0
]

previously_positive = station[
    station["net_flow_lag_1"] > 0
]


negative_continuation_rate = (
    previously_negative["net_flow"] < 0
).mean()

positive_continuation_rate = (
    previously_positive["net_flow"] > 0
).mean()


print("\nFLOW CONTINUATION")

print(
    "If previous window was negative,"
    " next window is also negative:",
    round(
        negative_continuation_rate * 100,
        2
    ),
    "%"
)

print(
    "If previous window was positive,"
    " next window is also positive:",
    round(
        positive_continuation_rate * 100,
        2
    ),
    "%"
)

# 24. LABEL NET FLOW DIRECTION

def flow_direction(value):

    if value < 0:
        return "Negative"

    elif value > 0:
        return "Positive"

    else:
        return "Zero"


station["flow_direction"] = (
    station["net_flow"]
    .apply(flow_direction)
)


# 25. CREATE THE PREVIOUS FLOW DIRECTION

station["previous_flow_direction"] = (
    station["flow_direction"]
    .shift(1)
)


# 26. BUILD A FLOW TRANSITION TABLE

flow_transitions = pd.crosstab(
    station["previous_flow_direction"],
    station["flow_direction"],
    normalize="index"
) * 100


print("\nFLOW TRANSITION TABLE")

print(
    flow_transitions.round(1)
)

# 27. CREATE SAME-TIME-LAST-WEEK FLOW

station["net_flow_lag_336"] = (
    station["net_flow"]
    .shift(336)
)

# 28. COMPARE RECENT FLOW VS SAME TIME LAST WEEK

recent_corr = (
    station["net_flow"]
    .corr(station["net_flow_lag_1"])
)

weekly_corr = (
    station["net_flow"]
    .corr(station["net_flow_lag_336"])
)


print("\nFLOW CORRELATION")

print(
    "Previous 30 minutes:",
    round(recent_corr, 3)
)

print(
    "Same time last week:",
    round(weekly_corr, 3)
)

# 29. CREATE NET FLOW LAGS FOR ALL STATIONS

df["net_flow_lag_1"] = (
    df
    .groupby("station_id")["net_flow"]
    .shift(1)
)

df["net_flow_lag_336"] = (
    df
    .groupby("station_id")["net_flow"]
    .shift(336)
)


# 30. CHECK SYSTEM-WIDE CORRELATION

system_recent_corr = (
    df["net_flow"]
    .corr(df["net_flow_lag_1"])
)

system_weekly_corr = (
    df["net_flow"]
    .corr(df["net_flow_lag_336"])
)


print("\nSYSTEM-WIDE FLOW CORRELATION")

print(
    "Previous 30 minutes:",
    round(system_recent_corr, 3)
)

print(
    "Same time last week:",
    round(system_weekly_corr, 3)
)
