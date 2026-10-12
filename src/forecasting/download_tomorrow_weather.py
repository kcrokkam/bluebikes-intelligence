from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

OUTPUT_FILE = Path(
    "data/processed/tomorrow_weather.csv"
)


def main():

    # 1. Find tomorrow's date in Boston time
    boston_time = ZoneInfo(
        "America/New_York"
    )

    today = datetime.now(
        boston_time
    ).date()

    tomorrow = (
        today + timedelta(days=1)
    )

    print(
        f"Downloading weather for {tomorrow}..."
    )

    # 2. Request hourly Boston forecast
    params = {
        "latitude": 42.3601,
        "longitude": -71.0589,
        "hourly": [
            "temperature_2m",
            "precipitation",
        ],
        "timezone": "America/New_York",
        "forecast_days": 2,
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    # 3. Convert API response into a DataFrame
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

    # 4. Keep only tomorrow's 24 hours
    weather = weather[
        weather["datetime"].dt.date
        == tomorrow
    ].copy()

    # 5. Save
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Rows: {len(weather)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print(
        weather.head().to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()