# Bluebikes Intelligence — Project Problem

## Goal

Build a reproducible next-day demand forecasting system that helps Bluebikes operations identify which stations are expected to experience the greatest bike depletion tomorrow and when that depletion is likely to peak.

## Business question

**Which Bluebikes stations are likely to experience the greatest net bike depletion tomorrow, and during which hours should operations prioritize monitoring them?**

The primary user is a Bluebikes operations or rebalancing team.

## Machine learning task

Predict hourly:

- bike departures
- bike arrivals

for every active station on the next day.

Departures and arrivals are modeled separately.

Predicted net flow is:

```text
predicted_net_flow
= predicted_arrivals - predicted_departures
```

Negative net flow represents expected bike depletion from customer trips.

## Business output

For each station, calculate the lowest cumulative predicted net flow during the forecast day and produce a ranked watchlist containing:

- station ID
- station name
- predicted cumulative depletion
- expected peak depletion time
- current bikes available
- current docks available

Current inventory is operational context only.

## What the project does not claim

Predicted net depletion is **not** the same as:

- a confirmed future stockout
- a verified need to dispatch a rebalancing truck
- true inventory change after operator rebalancing
- a financial savings estimate

Trip records capture customer departures and arrivals. They do not capture all operator-driven inventory movement.

## Historical data

The V1 forecasting dataset covers approximately:

**September 2023 through August 2026**

Historical trips are aggregated to station-hour records with valid zero-demand hours.

The full dataset contains approximately:

- 13.26 million station-hour observations
- 640 historical station IDs

## Features

The production-feasible model uses information that can be known before the forecast day:

- hour
- day of week
- month
- weekend indicator
- U.S. + Massachusetts holiday indicator
- hourly temperature
- hourly precipitation
- historical station departures for the weekday/hour
- historical station arrivals for the weekday/hour

For new stations without sufficient history, the model uses a network-level weekday/hour demand profile.

## Research model vs production model

A research model tested recent-demand lag features:

- same hour yesterday
- same hour one week earlier

Those features improved offline performance.

However, the public Bluebikes trip archive is not fresh enough to reliably supply those lagged trip counts for a true next-day run.

The final production-feasible model therefore uses frozen historical demand profiles instead.

This retains nearly all of the test performance while using inputs that are realistically available.

## Evaluation design

Evaluation is chronological rather than random.

Production-model evaluation periods:

```text
Historical profile:
through Aug 31, 2025

Model training:
Sep 1, 2025 – Feb 28, 2026

Validation:
Mar 1, 2026 – May 31, 2026

Test:
Jun 1, 2026 – Aug 31, 2026
```

This keeps the information boundary aligned with the forecasting problem.

## Forecasting metrics

Primary metrics:

- MAE
- WAPE

Production-feasible test results:

| Model | Departures MAE | Departures WAPE | Arrivals MAE | Arrivals WAPE |
|---|---:|---:|---:|---:|
| Frozen profile baseline | 0.860 | 70.0% | 0.859 | 70.1% |
| LightGBM | **0.828** | **67.4%** | **0.824** | **67.2%** |

## Ranking validation

The station watchlist is validated against actual cumulative trip net outflow during the unseen test period.

Across 92 test days:

- average Top-10 overlap: **4.04 stations**
- average Top-10 hit rate: **40.4%**
- median Top-10 hit rate: **40.0%**
- random Top-10 expected hit rate: **1.7%**

The operational objective is ranking useful stations, not predicting an exact future inventory level.

## Live context

The project also uses the Bluebikes GBFS station-status feed to provide current:

- bikes available
- docks available

A separate AWS pipeline captures GBFS station-status snapshots every 10 minutes using:

```text
EventBridge → Lambda → S3
```

This live inventory history is preserved for future inventory-aware modeling.

## V1 completion criteria

V1 is complete when the repository contains:

- reproducible historical trip ingestion
- station-hour demand dataset
- weather and calendar features
- historical demand profiles
- simple historical baseline
- production-feasible LightGBM model
- chronological evaluation
- historical watchlist validation
- real next-day weather forecast
- current active-station mapping
- next-day hourly demand predictions
- ranked watchlist with live inventory context
- clear documentation and limitations

## Out of scope

V1 intentionally excludes:

- deep learning
- LSTMs
- reinforcement learning
- autonomous agents
- truck routing optimization
- large dashboards
- elaborate MLOps
- stockout probability claims
- financial savings claims without causal evidence
