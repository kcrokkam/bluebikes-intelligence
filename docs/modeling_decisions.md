# Modeling Decisions

This document explains the choices behind the forecasting system and the tradeoffs that matter most in an interview or technical review.

## Why three years of trip history?

The project uses roughly Sep 2023-Aug 2026.

Three years provides repeated seasonal patterns, holidays, weather regimes, and weekday/weekend behavior while keeping the data relatively recent. Older mobility behavior can become stale as the station network and travel patterns change.

## Why station-hour rows?

The business decision is hourly station monitoring, so the modeling unit is:

```text
one station × one hour
```

Departures and arrivals are counted separately.

## Why include zero-demand hours?

A groupby over trips naturally drops hours with no trips.

Leaving those rows out would bias the dataset toward busy periods and make the model behave as if every station-hour had positive demand.

The pipeline creates continuous hourly rows inside each station's observed monthly activity window and fills true no-trip hours with zero.

## Why model departures and arrivals separately?

Operational depletion depends on both flows:

```text
net flow = arrivals - departures
```

Separate models let the system learn different patterns for outgoing and incoming demand.

## Why not use a random train/test split?

Nearby rows share strong time structure.

A random split would mix future and past behavior and produce an unrealistically easy evaluation.

All model evaluation is chronological.

## Why build a simple baseline first?

The baseline predicts normal demand from historical station/weekday/hour behavior.

If a more complex ML model cannot outperform that rule, the extra complexity is not justified.

The first LightGBM experiment actually underperformed the simple baseline on the final test period. That result was preserved rather than hidden.

## Why did recent-demand lags help?

The research model added:

- same hour one day earlier
- same hour one week earlier

Those features tell the model whether recent station activity is above or below its usual pattern.

They improved test performance.

## Why are lag features not used in the final production model?

The public Bluebikes trip archive is not published frequently enough to guarantee yesterday's customer trip counts for a real next-day forecast.

A feature that exists offline but cannot be produced at prediction time is not deployable.

The final model replaces recent-trip lags with a frozen historical station demand profile.

## How are new stations handled?

Some stations in later periods do not exist in the historical profile.

For those cold-start stations, the pipeline falls back to the network average for the same weekday and hour.

This avoids inventing station history.

## Why use observed weather in historical experiments?

Historical actual weather is easy to reproduce and useful for testing whether weather carries signal.

But it is more informative than a real forecast would have been at the time.

The README therefore distinguishes:

- historical backtest with observed weather
- real next-day run with forecast weather

A stronger V2 would backtest archived weather forecasts.

## Why LightGBM?

The dataset is large, sparse, and tabular.

LightGBM is well suited for:

- millions of rows
- nonlinear interactions
- mixed demand regimes
- efficient gradient boosting

The project deliberately does not compare many model families because the goal is an end-to-end operational system, not a model tournament.

## Why MAE?

MAE answers:

> On an average station-hour, how many trips is the forecast off by?

It is easy to interpret in bike-trip units.

## Why WAPE?

WAPE normalizes total absolute error by total actual demand.

It is useful for comparing periods with different demand volumes.

WAPE can exceed 100% on low-volume stations because the denominator is very small. That is why the project also reports MAE and examines busy versus low-volume stations separately.

## Why evaluate a Top-10 watchlist?

Operations rarely act on every model score.

The ranking question is more useful:

> If operations can focus on 10 stations, how many of the stations that actually experience the strongest depletion did the model surface?

In a 92-day out-of-time test, the **production-feasible LightGBM model** identified **4.02 of the actual Top-10 stations per day** (40.2% mean hit rate), versus **3.90 stations/day** (39.0%) for the **frozen historical-profile ranking**. Both were evaluated using identical test days and a common maximum cumulative net-depletion definition in `src/forecasting/evaluate_production_ranking.py`. The mean difference of **1.2 percentage points** is modest and has not been shown statistically significant.

The **offline research model**, with previous-day and previous-week demand lags, achieved **4.04 stations/day** (40.4%) in a separate ranking evaluation using `src/forecasting/gradient_boosting.py`, `build_watchlist.py`, `rank_stations.py`, and `evaluate_watchlist.py`. Research lag features are not available reliably for next-day production use. Historical ranking backtests use observed rather than forecast weather.

## Why isn't predicted depletion a stockout prediction?

Trip net flow does not equal actual inventory change.

Actual station inventory can also change because of:

- operator rebalancing
- station downtime
- bike maintenance
- inventory changes before the forecast day begins

The project therefore calls the output a depletion or monitoring priority, not a guaranteed stockout.

## Why add current live inventory?

Live bike and dock counts make the ranking easier for an operator to interpret.

They are shown as context after the demand forecast is generated.

Current inventory is not treated as tomorrow's starting inventory.

## What would improve V2?

The highest-value extensions are:

1. archived weather-forecast backtesting
2. special-event features
3. station-location features
4. scheduled retraining when new trip archives arrive
5. drift and forecast monitoring
6. stronger ranking baselines
7. inventory-aware modeling once enough historical inventory has accumulated
