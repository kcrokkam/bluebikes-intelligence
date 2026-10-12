import pandas as pd
import holidays


DEMAND_FILE = "data/processed/station_hour_demand.csv"
WEATHER_FILE = "data/processed/boston_hourly_weather.csv"
OUTPUT_FILE = "data/processed/station_hour_features.csv"


def add_calendar_features(df):
    """Add calendar features available before the forecast is made."""

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    # Calendar features
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Federal + Massachusetts holidays
    years = range(
        df["datetime"].dt.year.min(),
        df["datetime"].dt.year.max() + 1,
    )

    ma_holidays = holidays.US(
        subdiv="MA",
        years=years,
    )

    df["is_holiday"] = (
        df["datetime"]
        .dt.date
        .isin(ma_holidays)
        .astype(int)
    )

    return df


def add_weather_features(df):
    """Attach Boston hourly weather to each station-hour row."""

    weather = pd.read_csv(
        WEATHER_FILE
    )

    weather["datetime"] = pd.to_datetime(
        weather["datetime"]
    )

    df = df.merge(
        weather,
        on="datetime",
        how="left",
    )

    return df

def add_lag_features(df):
    """Add yesterday and last-week demand for the same station and hour."""

    station_groups = df.groupby(
        "station_id",
        observed=True,
    )

    # Same hour yesterday
    previous_time = station_groups[
        "datetime"
    ].shift(24)

    valid_24h = (
        df["datetime"] - previous_time
        == pd.Timedelta(hours=24)
    )

    df["departures_lag_24h"] = (
        station_groups["departures"]
        .shift(24)
        .where(valid_24h)
    )

    df["arrivals_lag_24h"] = (
        station_groups["arrivals"]
        .shift(24)
        .where(valid_24h)
    )

    del previous_time
    del valid_24h

    # Same hour one week ago
    previous_time = station_groups[
        "datetime"
    ].shift(168)

    valid_168h = (
        df["datetime"] - previous_time
        == pd.Timedelta(hours=168)
    )

    df["departures_lag_168h"] = (
        station_groups["departures"]
        .shift(168)
        .where(valid_168h)
    )

    df["arrivals_lag_168h"] = (
        station_groups["arrivals"]
        .shift(168)
        .where(valid_168h)
    )

    return df

def main():
    print("Loading station-hour demand data...")

    df = pd.read_csv(
        DEMAND_FILE
    )

    df = add_calendar_features(
        df
    )

    df = add_weather_features(
        df
    )

    df = add_lag_features(
        df
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nFeature dataset created.")
    print(df.columns.tolist())
    print(f"Rows: {len(df):,}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()