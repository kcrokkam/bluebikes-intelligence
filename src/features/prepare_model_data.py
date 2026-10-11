import pandas as pd

from build_inventory_dataset import (
    build_inventory_dataset,
    add_inventory_features,
    add_stockout_labels,
)


def prepare_model_data(snapshot_dir):
    """Create a small dataset ready for baseline modeling."""

    # 1. Build inventory history
    inventory_df = build_inventory_dataset(
        snapshot_dir
    )

    # 2. Create inventory features
    inventory_df = add_inventory_features(
        inventory_df
    )

    # 3. Create the prediction target
    inventory_df = add_stockout_labels(
        inventory_df
    )

    local_time = inventory_df[
    "feed_last_updated"].dt.tz_convert("America/New_York")

    # 4. Add time features
    inventory_df["hour"] = local_time.dt.hour

    inventory_df["day_of_week"] = (
        local_time.dt.dayofweek
    )

    # 5. Keep only rows with a known target
    model_df = inventory_df[
        inventory_df["stockout_next_60min"].notna()
    ].copy()

    # 6. Keep only our first simple feature set
    columns = [
        "num_bikes_available",
        "previous_bikes",
        "inventory_change",
        "hour",
        "day_of_week",
        "stockout_next_60min",
    ]

    model_df = model_df[columns]

    # 7. Remove rows missing previous inventory
    model_df = model_df.dropna().reset_index(drop=True)

    # 8. Convert target to normal integers
    model_df["stockout_next_60min"] = (
        model_df["stockout_next_60min"].astype(int)
    )

    return model_df


def main():
    model_df = prepare_model_data(
        "data/snapshots"
    )

    print("\nModel dataset shape:")
    print(model_df.shape)

    print("\nColumns:")
    print(model_df.columns.tolist())

    print("\nTarget counts:")
    print(
        model_df["stockout_next_60min"]
        .value_counts()
    )

    print("\nFirst 5 rows:")
    print(model_df.head())


if __name__ == "__main__":
    main()