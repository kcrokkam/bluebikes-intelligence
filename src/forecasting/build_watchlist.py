import pandas as pd


DEPARTURES_FILE = (
    "data/processed/test_departures_predictions.csv"
)

ARRIVALS_FILE = (
    "data/processed/test_arrivals_predictions.csv"
)

OUTPUT_FILE = (
    "data/processed/test_hourly_net_flow.csv"
)


def main():

    print("Loading predictions...")

    departures = pd.read_csv(
        DEPARTURES_FILE,
        parse_dates=["datetime"],
    )

    arrivals = pd.read_csv(
        ARRIVALS_FILE,
        parse_dates=["datetime"],
    )

    # Combine departure and arrival predictions
    df = departures.merge(
        arrivals,
        on=[
            "station_id",
            "datetime",
        ],
    )

    # Actual net bike flow
    df["actual_net_flow"] = (
        df["arrivals"]
        - df["departures"]
    )

    # Predicted net bike flow
    df["predicted_net_flow"] = (
        df["predicted_arrivals"]
        - df["predicted_departures"]
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nHourly net-flow dataset created.")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nFirst 10 rows:")

    print(
        df.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()