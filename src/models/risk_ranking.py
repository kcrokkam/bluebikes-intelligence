from pathlib import Path
import json
import urllib.request

from sklearn.linear_model import LogisticRegression

from src.data_pipeline.inventory import (
    build_inventory_dataset,
    add_inventory_features,
    add_stockout_labels,
)


FEATURES = [
    "num_bikes_available",
    "previous_bikes",
    "inventory_change",
    "hour",
    "day_of_week",
]


def get_station_names():
    """Download human-readable Bluebikes station names."""

    url = (
        "https://gbfs.bluebikes.com/gbfs/en/"
        "station_information.json"
    )

    with urllib.request.urlopen(
        url,
        timeout=10,
    ) as response:
        data = json.load(response)

    stations = data["data"]["stations"]

    station_names = {
        station["station_id"]: station["name"]
        for station in stations
    }

    return station_names


def main():
    # 1. Load all inventory history
    inventory_df = build_inventory_dataset(
        "data/snapshots"
    )

    # 2. Create inventory features
    inventory_df = add_inventory_features(
        inventory_df
    )

    # 3. Create historical stockout labels
    inventory_df = add_stockout_labels(
        inventory_df
    )

    # 4. Create Boston local-time features
    local_time = inventory_df[
        "feed_last_updated"
    ].dt.tz_convert("America/New_York")

    inventory_df["hour"] = local_time.dt.hour

    inventory_df["day_of_week"] = (
        local_time.dt.dayofweek
    )

    # 5. Find newest available snapshot
    latest_time = inventory_df[
        "feed_last_updated"
    ].max()

    # 6. Prepare historical training data
    train_df = inventory_df[
        inventory_df["stockout_next_60min"].notna()
    ].copy()

    train_df = train_df.dropna(
        subset=FEATURES
    )

    # 7. Get stations from latest snapshot
    latest_df = inventory_df[
        inventory_df["feed_last_updated"] == latest_time
    ].copy()

    latest_df = latest_df.dropna(
        subset=FEATURES
    )

    # 8. Separate stations already stocked out
    current_stockouts = latest_df[
        latest_df["num_bikes_available"] == 0
    ].copy()

    # Only stations with bikes remaining
    # should receive future-stockout predictions
    current_df = latest_df[
        latest_df["num_bikes_available"] > 0
    ].copy()

    # 9. Train logistic regression
    X_train = train_df[FEATURES]

    y_train = (
        train_df["stockout_next_60min"]
        .astype(int)
    )

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )

    model.fit(
        X_train,
        y_train,
    )

    # 10. Score current stations
    current_df["risk_score"] = (
        model.predict_proba(
            current_df[FEATURES]
        )[:, 1]
    )

    # 11. Add simple low-bike rule baseline
    current_df["low_bike_alert"] = (
        current_df["num_bikes_available"] <= 1
    )

    # 12. Add human-readable station names
    station_names = get_station_names()

    current_df["station_name"] = (
        current_df["station_id"]
        .map(station_names)
        .fillna("Unknown station")
    )

    # 13. Keep useful output columns
    risk_df = current_df[
        [
            "station_name",
            "station_id",
            "num_bikes_available",
            "previous_bikes",
            "inventory_change",
            "risk_score",
            "low_bike_alert",
        ]
    ].copy()

    # 14. Rank stations highest risk first
    risk_df = risk_df.sort_values(
        "risk_score",
        ascending=False,
    )

    # 15. Save ranking
    output_path = Path(
        "data/processed/latest_station_risk.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    risk_df.to_csv(
        output_path,
        index=False,
    )

    # 16. Print results
    print("\nLatest snapshot:")
    print(latest_time)

    print(
        "\nStations already stocked out:",
        len(current_stockouts),
    )

    print(
        "Stations available for future-risk prediction:",
        len(risk_df),
    )

    print("\nTop 20 stations by model risk:")

    print(
        risk_df.head(20).to_string(
            index=False,
            formatters={
                "risk_score": lambda x: f"{x:.1%}"
            },
        )
    )

    print(
        "\nSaved to:",
        output_path,
    )


if __name__ == "__main__":
    main()