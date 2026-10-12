"""Run the complete next-day forecasting workflow."""

from src.forecasting.build_tomorrow_features import main as build_features
from src.forecasting.download_tomorrow_weather import main as download_weather
from src.forecasting.final_watchlist import main as build_watchlist
from src.forecasting.predict_tomorrow import main as predict


def main():
    print("\n1/4 Downloading tomorrow's weather...")
    download_weather()

    print("\n2/4 Building tomorrow's station-hour features...")
    build_features()

    print("\n3/4 Training final models and predicting tomorrow...")
    predict()

    print("\n4/4 Building final watchlist with live inventory...")
    build_watchlist()

    print("\nNext-day Bluebikes watchlist complete.")


if __name__ == "__main__":
    main()
