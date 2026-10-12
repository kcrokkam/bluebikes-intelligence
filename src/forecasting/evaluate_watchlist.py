import pandas as pd


INPUT_FILE = (
    "data/processed/test_station_watchlist.csv"
)

TOP_K = 10


def main():

    df = pd.read_csv(
        INPUT_FILE
    )

    daily_results = []

    for date, day_df in df.groupby("date"):

        predicted_top = day_df.nlargest(
            TOP_K,
            "predicted_depletion",
        )

        actual_top = day_df.nlargest(
            TOP_K,
            "actual_max_depletion",
        )

        predicted_stations = set(
            predicted_top["station_id"]
        )

        actual_stations = set(
            actual_top["station_id"]
        )

        overlap = len(
            predicted_stations
            & actual_stations
        )

        hit_rate = overlap / TOP_K

        number_of_stations = len(day_df)

        random_hit_rate = (
            TOP_K / number_of_stations
        )

        daily_results.append(
            {
                "date": date,
                "overlap": overlap,
                "hit_rate": hit_rate,
                "random_hit_rate": random_hit_rate,
            }
        )

    results = pd.DataFrame(
        daily_results
    )

    print("\nWATCHLIST VALIDATION")

    print(
        f"Test days: {len(results)}"
    )

    print(
        f"Average Top-{TOP_K} overlap: "
        f"{results['overlap'].mean():.2f} stations"
    )

    print(
        f"Average Top-{TOP_K} hit rate: "
        f"{results['hit_rate'].mean():.1%}"
    )

    print(
        f"Median Top-{TOP_K} hit rate: "
        f"{results['hit_rate'].median():.1%}"
    )

    print(
        f"Random Top-{TOP_K} expected hit rate: "
        f"{results['random_hit_rate'].mean():.1%}"
    )


if __name__ == "__main__":
    main()