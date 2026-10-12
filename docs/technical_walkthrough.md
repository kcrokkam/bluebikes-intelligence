# Technical Walkthrough

This note explains the project in a way that is useful for technical review and interview preparation.

## 30-second summary

Bluebikes Intelligence forecasts hourly bike departures and arrivals for every active Bluebikes station one day ahead. It uses three years of trip history, calendar features, Massachusetts holidays, weather, and frozen station-level demand profiles. The forecasts are converted into expected cumulative net bike depletion and ranked into a daily watchlist. Live GBFS bike counts are added as operational context. The production-feasible LightGBM model beats a frozen historical-profile baseline, and the Top-10 watchlist captures 40.4% of the actual daily Top-10 depletion stations across 92 unseen test days.

## Problem framing

The goal is not to predict a guaranteed stockout.

The goal is to help operations answer:

> Which stations are most likely to experience strong customer-driven bike depletion tomorrow, and when should we pay attention?

This distinction matters because actual inventory also changes through operator rebalancing and other system activity.

## Core pipeline

```text
historical trips
    ↓
station-hour departures and arrivals
    ↓
calendar + holidays + weather
    ↓
historical demand profiles
    ↓
LightGBM departures model
LightGBM arrivals model
    ↓
predicted net flow
    ↓
cumulative daily depletion
    ↓
ranked station watchlist
    ↓
live bike/dock context
```

## What is one training row?

One row represents:

```text
one station × one hour
```

The targets are:

- departures during that hour
- arrivals during that hour

Zero-demand hours are included so the model does not learn only from hours where trips happened.

## Why three years?

Three years gives repeated seasonal and holiday patterns while keeping behavior reasonably current.

Using much older data could introduce stale mobility patterns and station-network behavior.

## Why not a random train/test split?

Forecasting is a time problem.

The model should learn from the past and be tested on later periods.

A random split would mix future and past observations and make the evaluation less realistic.

## Baseline

The simple baseline asks:

> What usually happens at this station on this weekday at this hour?

This is intentionally strong.

The ML model has to beat a useful operational rule, not a weak dummy predictor.

## Why LightGBM?

The dataset is large and tabular.

LightGBM handles millions of rows efficiently and can learn nonlinear interactions between:

- station behavior
- time
- weather
- holidays

## Research model vs production model

The research model used recent demand:

- same hour yesterday
- same hour last week

Those features improved offline performance.

The public Bluebikes trip archive is not fresh enough to reliably provide them for a true next-day forecast.

The production model therefore uses frozen historical demand profiles instead.

That decision gives up very little performance while keeping the model deployable.

## Metrics

### MAE

MAE is the average absolute forecast error in trip counts.

A departures MAE of 0.828 means the model is off by about 0.83 departures per station-hour on average.

### WAPE

WAPE is:

```text
total absolute error / total actual demand
```

It is useful when demand volume changes across periods.

It can exceed 100% for low-volume stations because the denominator is very small.

## Production-model test results

| Model | Departures MAE | Departures WAPE | Arrivals MAE | Arrivals WAPE |
|---|---:|---:|---:|---:|
| Frozen profile baseline | 0.860 | 70.0% | 0.859 | 70.1% |
| LightGBM | **0.828** | **67.4%** | **0.824** | **67.2%** |

## Ranking metric

Forecast accuracy is only part of the business problem.

Operations needs a priority list.

For every test day, the system compares:

- predicted Top-10 depletion stations
- actual Top-10 trip-net-depletion stations

Across 92 unseen days:

- average overlap: 4.04 stations
- average hit rate: 40.4%
- median hit rate: 40.0%
- random expectation: 1.7%

## Live inventory

Live GBFS provides current bikes and docks.

This is added after forecasting as context.

It is not treated as tomorrow's starting inventory because rebalancing and customer activity can change inventory before the forecast day begins.

## Cold-start stations

New stations may not have enough historical demand.

The system falls back to the network weekday/hour profile rather than inventing station-specific history.

## Questions to be ready to answer

### Why did the first ML model not win?

The first model did not include recent station behavior and underperformed the historical baseline on the final test period.

That result motivated a better feature design rather than being hidden.

### Why not keep the lag features if they worked best?

Because a feature that cannot be produced reliably at prediction time is not deployable.

### Why does a 40.4% Top-10 hit rate matter?

It measures the actual decision-support task: whether the model points operations toward stations that later experience the strongest depletion.

### Why is random not the only baseline?

Random selection shows the ranking contains signal, but the forecasting model is also compared against a strong historical-profile baseline.

A future improvement would compare ranking quality against additional operational heuristics.

### What is the biggest limitation?

Trip net flow is not true inventory change because operator rebalancing is not visible in trip history.

### What would you build next?

The highest-value next steps are archived forecast-weather backtesting, special-event/location features, scheduled retraining, and eventually inventory-aware models using the continuously collected S3 history.
