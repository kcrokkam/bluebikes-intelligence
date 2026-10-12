from pathlib import Path

import pandas as pd
import requests


WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

OUTPUT_FILE = Path(
    "data/processed/boston_hourly_weather.csv"
)


def download_weather():
    """Download historical hourly weather for Boston."""

    params = {
        "latitude": 42.3601,
        "longitude": -71.0589,
        "start_date": "2023-09-01",
        "end_date": "2026-08-31",
        "hourly": [
            "temperature_2m",
            "precipitation",
        ],
        "timezone": "America/New_York",
    }

    print("Downloading Boston historical weather...")

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    weather = pd.DataFrame(
        data["hourly"]
    )

    weather["time"] = pd.to_datetime(
        weather["time"]
    )

    weather = weather.rename(
        columns={
            "time": "datetime",
            "temperature_2m": "temperature",
        }
    )

    return weather


def main():
    weather = download_weather()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nWeather dataset created.")
    print(f"Rows: {len(weather):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nFirst 10 rows:")
    print(weather.head(10).to_string(index=False))


if __name__ == "__main__":
    main()