import pandas as pd


file_path = "data/raw/202608-bluebikes-tripdata.csv"

df = pd.read_csv(file_path)

df["started_at"] = pd.to_datetime(df["started_at"])
df["ended_at"] = pd.to_datetime(df["ended_at"])

departures = (
    df.dropna(subset=["start_station_id"])
      .groupby([
          "start_station_id",
          pd.Grouper(key="started_at", freq="30min")
      ])
      .size()
      .reset_index(name="departures")
)

arrivals = (
    df.dropna(subset=["end_station_id"])
      .groupby([
          "end_station_id",
          pd.Grouper(key="ended_at", freq="30min")
      ])
      .size()
      .reset_index(name="arrivals")
)

print(arrivals.head(10))

