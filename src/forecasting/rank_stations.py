import pandas as pd


INPUT_FILE = (
    "data/processed/test_hourly_net_flow.csv"
)

OUTPUT_FILE = (
    "data/processed/test_station_watchlist.csv"
)


def main():

    print("Loading hourly net-flow predictions...")

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["datetime"],
    )

    # 1. Create calendar date
    df["date"] = df["datetime"].dt.date

    # 2. Sort rows in time order
    df = df.sort_values(
        [
            "station_id",
            "date",
            "datetime",
        ]
    )

    # 3. Predicted cumulative bike movement
    df["predicted_cumulative_flow"] = (
        df
        .groupby(
            ["station_id", "date"]
        )["predicted_net_flow"]
        .cumsum()
    )

    # 4. Actual cumulative bike movement
    df["actual_cumulative_flow"] = (
        df
        .groupby(
            ["station_id", "date"]
        )["actual_net_flow"]
        .cumsum()
    )

    # 5. Find the hour of greatest predicted depletion
    worst_rows = (
        df
        .groupby(
            ["station_id", "date"]
        )["predicted_cumulative_flow"]
        .idxmin()
    )

    watchlist = df.loc[
        worst_rows,
        [
            "station_id",
            "date",
            "datetime",
            "predicted_cumulative_flow",
            "actual_cumulative_flow",
        ],
    ].copy()

    # 6. Convert negative cumulative flow
    # into positive depletion values
    watchlist["predicted_depletion"] = (
        -watchlist["predicted_cumulative_flow"]
    ).clip(lower=0)

    watchlist["actual_depletion_at_that_hour"] = (
        -watchlist["actual_cumulative_flow"]
    ).clip(lower=0)

    # 7. Calculate the actual worst depletion
    # reached by each station during the whole day
    actual_worst = (
        df
        .groupby(
            ["station_id", "date"]
        )["actual_cumulative_flow"]
        .min()
        .reset_index()
    )

    actual_worst["actual_max_depletion"] = (
        -actual_worst["actual_cumulative_flow"]
    ).clip(lower=0)

    actual_worst = actual_worst[
        [
            "station_id",
            "date",
            "actual_max_depletion",
        ]
    ]

    # 8. Add actual worst daily depletion
    # to the predicted watchlist
    watchlist = watchlist.merge(
        actual_worst,
        on=[
            "station_id",
            "date",
        ],
        how="left",
    )

    # 9. Rename predicted worst time
    watchlist = watchlist.rename(
        columns={
            "datetime": "peak_depletion_time"
        }
    )

    # 10. Rank stations from highest
    # predicted depletion to lowest
    watchlist = watchlist.sort_values(
        [
            "date",
            "predicted_depletion",
        ],
        ascending=[
            True,
            False,
        ],
    )

    # 11. Save full historical watchlist
    watchlist.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # 12. Show top stations
    # for the latest test date
    latest_date = watchlist["date"].max()

    latest_watchlist = watchlist[
        watchlist["date"] == latest_date
    ].head(15)

    print(
        f"\nTop stations for {latest_date}:"
    )

    print(
        latest_watchlist[
            [
                "station_id",
                "predicted_depletion",
                "peak_depletion_time",
                "actual_max_depletion",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()