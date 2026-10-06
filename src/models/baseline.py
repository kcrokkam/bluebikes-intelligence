import pandas as pd


# 1. LOAD THE PROCESSED DATA

file_path = "data/processed/station_flow_202606_202608.csv"

df = pd.read_csv(file_path)


# 2. CONVERT TIME BACK TO DATETIME

df["time_window"] = pd.to_datetime(df["time_window"])


# 3. SPLIT THE DATA INTO TRAINING AND TEST DATA

train = df[df["time_window"] < "2026-08-25"].copy()

test = df[df["time_window"] >= "2026-08-25"].copy()


# 4. CALCULATE THE BASELINE PREDICTION
# For each station and hour, calculate average historical departures

baseline = (
    train
    .groupby(
    ["station_id", "day_of_week", "time_slot"])["departures"]
    .mean()
    .reset_index(name="predicted_departures")
)


# 5. ATTACH THE BASELINE PREDICTION TO TEST DATA

test = test.merge(
    baseline,
    on=[
    "station_id",
    "day_of_week",
    "time_slot"
],
    how="left"
)


# 6. CALCULATE THE ABSOLUTE ERROR
# Example:
# Actual = 3
# Prediction = 1.5
# Absolute error = 1.5

test["absolute_error"] = (
    test["departures"] - test["predicted_departures"]
).abs()


# 7. CALCULATE OVERALL MAE

mae = test["absolute_error"].mean()

print("Baseline MAE:", mae)


# 8. EVALUATE WINDOWS WHERE THERE WAS ACTUAL DEMAND

active_test = test[
    test["departures"] > 0
]

active_mae = active_test["absolute_error"].mean()

print("MAE when departures > 0:", active_mae)


# 9. EVALUATE BUSIER WINDOWS

busy_test = test[
    test["departures"] >= 3
]

busy_mae = busy_test["absolute_error"].mean()

print("MAE when departures >= 3:", busy_mae)




# 10. CHECK BASELINE PREDICTION COVERAGE

print("Total test rows:", len(test))

print(
    "Missing baseline predictions:",
    test["predicted_departures"].isna().sum()
)