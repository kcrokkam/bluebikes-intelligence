from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.data_pipeline.model_data import prepare_model_data


def evaluate_rule(test_df, bike_threshold):
    """Evaluate a simple low-bike alert rule."""

    y_test = test_df["stockout_next_60min"]

    predictions = (
        test_df["num_bikes_available"] <= bike_threshold
    ).astype(int)

    print(
        f"\nRule: alert when bikes <= {bike_threshold}"
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "Precision:",
        round(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            3,
        ),
    )

    print(
        "Recall:",
        round(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            3,
        ),
    )


def main():
    # 1. Prepare inventory data
    model_df = prepare_model_data(
        "data/snapshots"
    )

    # 2. Convert timestamps to Boston time
    local_time = model_df[
        "feed_last_updated"
    ].dt.tz_convert("America/New_York")

    model_df["local_date"] = local_time.dt.date

    # 3. Use the latest day as the test day
    test_date = model_df["local_date"].max()

    test_df = model_df[
        model_df["local_date"] == test_date
    ].copy()

    print("\nTest date:", test_date)
    print("Test rows:", len(test_df))

    # 4. Compare simple operational rules
    evaluate_rule(test_df, 1)
    evaluate_rule(test_df, 2)
    evaluate_rule(test_df, 3)


if __name__ == "__main__":
    main()