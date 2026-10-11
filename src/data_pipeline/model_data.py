import pandas as pd

from src.data_pipeline.inventory import (
    build_inventory_dataset,
    add_inventory_features,
    add_stockout_labels,
)


def prepare_model_data(snapshot_dir):
    """Prepare inventory observations for stockout modeling."""

    # 1. Build inventory history
    inventory_df = build_inventory_dataset(
        snapshot_dir
    )

    # 2. Create inventory features
    inventory_df = add_inventory_features(
        inventory_df
    )

    # 3. Create stockout target
    inventory_df = add_stockout_labels(
        inventory_df
    )

    # 4. Convert UTC time to Boston local time
    local_time = inventory_df[
        "feed_last_updated"
    ].dt.tz_convert("America/New_York")

    # 5. Add time features
    inventory_df["hour"] = local_time.dt.hour
    inventory_df["day_of_week"] = local_time.dt.dayofweek

    # 6. Keep only rows with a known target
    model_df = inventory_df[
        inventory_df["stockout_next_60min"].notna()
    ].copy()

    # 7. Keep model features, timestamp, and target
    columns = [
        "feed_last_updated",
        "num_bikes_available",
        "previous_bikes",
        "inventory_change",
        "hour",
        "day_of_week",
        "stockout_next_60min",
    ]

    model_df = model_df[columns]

    # 8. Remove incomplete feature rows
    model_df = (
        model_df
        .dropna()
        .reset_index(drop=True)
    )

    # 9. Convert target to integer
    model_df["stockout_next_60min"] = (
        model_df["stockout_next_60min"]
        .astype(int)
    )

    return model_df


def main():
    model_df = prepare_model_data(
        "data/snapshots"
    )

    print("\nModel dataset shape:")
    print(model_df.shape)

    print("\nTarget counts:")
    print(
        model_df["stockout_next_60min"]
        .value_counts()
    )

    print("\nFirst 5 rows:")
    print(model_df.head())


if __name__ == "__main__":
    main()