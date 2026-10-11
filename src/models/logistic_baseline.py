import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)

from src.data_pipeline.model_data import prepare_model_data


FEATURES = [
    "num_bikes_available",
    "previous_bikes",
    "inventory_change",
    "hour",
    "day_of_week",
]


def main():
    # 1. Prepare all available inventory data
    model_df = prepare_model_data(
        "data/snapshots"
    )

    # 2. Convert timestamps to Boston time
    local_time = model_df[
        "feed_last_updated"
    ].dt.tz_convert("America/New_York")

    model_df["local_date"] = local_time.dt.date

    # 3. Latest day = test day
    test_date = model_df["local_date"].max()

    # Day before = training day
    train_date = test_date - pd.Timedelta(days=1)

    # 4. Find when today's observations begin
    test_start = model_df.loc[
        model_df["local_date"] == test_date,
        "feed_last_updated",
    ].min()

    # 5. Leave a 60-minute gap before test data
    train_cutoff = (
        test_start
        - pd.Timedelta(minutes=60)
    )

    # 6. Yesterday = training data
    train_df = model_df[
        (model_df["local_date"] == train_date)
        & (
            model_df["feed_last_updated"]
            < train_cutoff
        )
    ].copy()

    # 7. Today = completely unseen test data
    test_df = model_df[
        model_df["local_date"] == test_date
    ].copy()

    print("\nTrain date:", train_date)
    print("Test date:", test_date)

    print("Train rows:", len(train_df))
    print("Test rows:", len(test_df))

    print("\nTrain target counts:")
    print(
        train_df["stockout_next_60min"]
        .value_counts()
    )

    print("\nTest target counts:")
    print(
        test_df["stockout_next_60min"]
        .value_counts()
    )

    # 8. Select features and targets
    X_train = train_df[FEATURES]
    y_train = train_df["stockout_next_60min"]

    X_test = test_df[FEATURES]
    y_test = test_df["stockout_next_60min"]

    # 9. Train baseline model
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
    )

    model.fit(
        X_train,
        y_train,
    )

    # 10. Predict probabilities
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # 11. Standard 50% threshold
    predictions = (
        probabilities >= 0.50
    ).astype(int)

    # 12. Evaluate on today
    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "\nPrecision:",
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


if __name__ == "__main__":
    main()