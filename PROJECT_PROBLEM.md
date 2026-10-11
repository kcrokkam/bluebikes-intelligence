# Bluebikes Rebalancing Intelligence

## Business Problem

Bluebikes Rebalancing Intelligence predicts near-term bike stockout risk across Bluebikes stations in the Boston area.

The operational goal is to give the rebalancing team enough warning to prioritize stations before riders encounter an empty station.

The decision is not simply whether a station is currently busy. It is:

- Which stations are at risk of running out of bikes?
- Which stations should be prioritized for rebalancing?
- How much warning can be provided before a stockout occurs?
- Does a predictive model add operational value over a simple low-inventory rule?

## User

The primary user is the Bluebikes operations and rebalancing team.

## Prediction Question

Given the information available about a station at the current time, will the station reach zero available bikes within the next 60 minutes?

The current model produces a risk score used for ranking stations. Because the logistic-regression model has not been probability calibrated, the score is not interpreted as a literal stockout probability.

## Prediction Target

`stockout_next_60min`

- `1` = the station currently has bikes and reaches zero available bikes at least once during the next 60 minutes
- `0` = the station currently has bikes and remains above zero during the next 60 minutes
- `NA` = the station is already at zero or a complete future 60-minute window is unavailable

Stations already at zero are handled separately because the stockout has already occurred.

## Prediction Horizon

Primary modeling horizon:

**60 minutes**

This is intended to provide enough lead time for an operational response while remaining close enough to current station conditions to be useful.

## V1 Features

The initial model deliberately uses a small and interpretable feature set:

- current bikes available
- previous bikes available
- inventory change
- hour of day
- day of week

Future versions may add:

- station capacity
- availability ratio
- 30- and 60-minute inventory trends
- station-specific historical stockout behavior
- historical trip demand
- weather
- holidays and major events

## Data Leakage Rule

For a prediction generated at time `t`, model features may use only information available at or before `t`.

Future observations are used only to construct the training label.

The model must not use:

- future bike counts
- future inventory movement
- future trips
- any other observation occurring after the prediction timestamp

## Data Available

### Historical trip data

Historical Bluebikes trip data covers multiple months and includes station-level departures, arrivals, net flow, time-of-day, and weekday/weekend behavior.

Trip data is useful for historical demand context, but it cannot reconstruct true station inventory because operator rebalancing is not visible in trip records.

### Live station inventory

A cloud pipeline collects the Bluebikes GBFS `station_status` feed every 10 minutes:

```text
Bluebikes GBFS
      ↓
AWS EventBridge Scheduler
      ↓
AWS Lambda
      ↓
Amazon S3
```

Each raw JSON snapshot contains station-level fields such as:

- `station_id`
- `num_bikes_available`
- `num_docks_available`
- `num_bikes_disabled`
- `num_docks_disabled`
- `is_installed`
- `is_renting`
- `is_returning`
- `last_reported`

These snapshots are transformed into the historical inventory dataset used for feature engineering and future stockout labels.

## Evaluation

Accuracy is not a useful primary metric because stockout-risk observations are relatively rare.

V1 focuses on:

- recall
- precision
- confusion matrix
- comparison with simple low-bike rules
- out-of-time testing rather than random splitting

The first V1 out-of-time evaluation produced:

| Method | Precision | Recall |
|---|---:|---:|
| Bikes ≤ 1 | 15.8% | 44.7% |
| Bikes ≤ 2 | 11.3% | 67.5% |
| Bikes ≤ 3 | 8.7% | 78.4% |
| Logistic Regression | 3.2% | 96.7% |

The logistic model captured most positive observations but generated too many false alerts to outperform the simple rule operationally.

## V1 Status

Completed:

- historical Bluebikes trip ingestion
- exploratory station-demand analysis
- local GBFS inventory collection
- AWS Lambda station-status collector
- EventBridge scheduling every 10 minutes
- Amazon S3 raw snapshot storage
- inventory-history construction
- leakage-aware 60-minute stockout target
- model-ready dataset pipeline
- logistic-regression baseline
- simple low-bike rule baselines
- out-of-time evaluation with a 60-minute boundary gap
- separation of current stockouts from future-risk candidates
- human-readable current station risk ranking

## Current Limitations

V1 currently has a short live inventory history.

Additional limitations include:

- no station-specific historical behavior in the model
- no capacity or availability-ratio features
- no longer-window inventory trends
- no weather or event context
- risk scores are not probability calibrated
- repeated 10-minute alerts are evaluated as individual observations rather than operational episodes
- snapshot-gap validation can be improved

## Next Phase

The AWS pipeline will continue accumulating inventory history.

V2 will prioritize:

1. more historical inventory
2. stronger snapshot and coverage validation
3. station capacity and inventory-trend features
4. station-specific historical behavior
5. episode-level operational evaluation
6. cost-sensitive decision thresholds
7. more complex models only if they outperform the simple rule on genuinely future data
