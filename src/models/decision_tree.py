import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# 1. LOAD THE PROCESSED DATA

file_path = "data/processed/station_flow_202606_202608.csv"

df = pd.read_csv(file_path)

df["time_window"] = pd.to_datetime(df["time_window"])


# 2. REMOVE ROWS WHERE LAG FEATURES ARE MISSING

df = df.dropna(
    subset=[
        "departures_lag_1",
        "departures_lag_2",
        "departures_rolling_3"
    ]
)


# 3. SPLIT INTO TRAIN AND TEST DATA

train = df[df["time_window"] < "2026-08-25"].copy()

test = df[df["time_window"] >= "2026-08-25"].copy()


# 4. CHOOSE THE FEATURES

features = [
    "station_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "departures_lag_1",
    "departures_lag_2",
    "departures_rolling_3"
]


# 5. CREATE X AND y

X_train = train[features]
y_train = train["departures"]

X_test = test[features]
y_test = test["departures"]


# 6. TELL SCIKIT-LEARN WHICH COLUMN IS CATEGORICAL

categorical_features = ["station_id"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "station_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# 7. CREATE THE DECISION TREE

model = DecisionTreeRegressor(
    max_depth=10,
    random_state=42
)


# 8. CREATE A PIPELINE
# First encode station_id
# Then send everything into the decision tree

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 9. TRAIN THE MODEL

pipeline.fit(X_train, y_train)


# 10. MAKE PREDICTIONS

predictions = pipeline.predict(X_test)


# 11. SAVE PREDICTIONS INTO TEST DATA

test["predicted_departures"] = predictions


# 12. CALCULATE ABSOLUTE ERROR

test["absolute_error"] = (
    test["departures"] - test["predicted_departures"]
).abs()


# 13. CALCULATE OVERALL MAE

mae = test["absolute_error"].mean()

print("Decision Tree MAE:", mae)


# 14. EVALUATE WINDOWS WHERE THERE WAS ACTUAL DEMAND

active_test = test[
    test["departures"] > 0
]

active_mae = active_test["absolute_error"].mean()

print(
    "Decision Tree MAE when departures > 0:",
    active_mae
)


# 15. EVALUATE BUSIER WINDOWS

busy_test = test[
    test["departures"] >= 3
]

busy_mae = busy_test["absolute_error"].mean()

print(
    "Decision Tree MAE when departures >= 3:",
    busy_mae
)