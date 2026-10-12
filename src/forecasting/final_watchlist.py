import pandas as pd
import requests


PREDICTIONS_FILE = (
    "data/processed/tomorrow_predictions.csv"
)

OUTPUT_FILE = (
    "data/processed/tomorrow_watchlist.csv"
)

STATUS_URL = (
    "https://gbfs.bluebikes.com/gbfs/en/station_status.json"
)


def get_live_inventory():
    """Get the latest live Bluebikes inventory."""

    data = requests.get(
        STATUS_URL,
        timeout=20,
    ).json()

    inventory = pd.DataFrame(
        data["data"]["stations"]
    )

    inventory = inventory[
        [
            "station_id",
            "num_bikes_available",
            "num_docks_available",
        ]
    ]

    inventory = inventory.rename(
        columns={
            "station_id": "live_station_id",
            "num_bikes_available": "current_bikes",
            "num_docks_available": "current_docks",
        }
    )

    return inventory


def main():

    print("Loading tomorrow predictions...")

    df = pd.read_csv(
        PREDICTIONS_FILE,
        parse_dates=["datetime"],
    )

    # 1. Calculate cumulative predicted bike movement
    df = df.sort_values(
        [
            "station_id",
            "datetime",
        ]
    )

    df["predicted_cumulative_flow"] = (
        df
        .groupby("station_id")[
            "predicted_net_flow"
        ]
        .cumsum()
    )

    # 2. Find the expected worst depletion point
    worst_rows = (
        df
        .groupby("station_id")[
            "predicted_cumulative_flow"
        ]
        .idxmin()
    )

    watchlist = df.loc[
        worst_rows,
        [
            "station_id",
            "live_station_id",
            "name",
            "datetime",
            "predicted_cumulative_flow",
        ],
    ].copy()

    # 3. Convert negative flow into positive depletion
    watchlist["predicted_depletion"] = (
        -watchlist["predicted_cumulative_flow"]
    ).clip(lower=0)

    watchlist = watchlist.rename(
        columns={
            "datetime": "peak_depletion_time"
        }
    )

    # 4. Add current live inventory
    inventory = get_live_inventory()

    watchlist = watchlist.merge(
        inventory,
        on="live_station_id",
        how="left",
    )

    # 5. Rank by expected depletion
    watchlist = watchlist.sort_values(
        "predicted_depletion",
        ascending=False,
    )

    # 6. Keep the useful final columns
    watchlist = watchlist[
        [
            "station_id",
            "name",
            "predicted_depletion",
            "peak_depletion_time",
            "current_bikes",
            "current_docks",
        ]
    ]

    watchlist.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\nTOMORROW'S TOP 15 STATION WATCHLIST")

    print(
        watchlist
        .head(15)
        .to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()