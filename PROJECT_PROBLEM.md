# Bluebikes Rebalancing Intelligence

## Business Problem

Bluebikes Rebalancing Intelligence is a machine learning project designed
to predict future bike stockout risk across Bluebikes stations in the
Boston area.

The goal is to provide the operations team with enough warning to decide
whether a station should be rebalanced before riders encounter an empty
station.

The operational decision is not simply whether a station is busy. It is:

- Which stations are at risk of running out of bikes?
- How soon could that happen?
- Which stations should be prioritized for rebalancing?
- Is intervention worth the operational cost compared with waiting?

## User

The primary user is the Bluebikes operations and rebalancing team.

## Prediction Question

Given everything known about a station at the current time, what is the
probability that the station will experience a bike stockout within the
next 60 minutes?

Example:

At 7:00 AM:

P(stockout between 7:00 AM and 8:00 AM) = 0.82

The probability can later be translated into an operational action such
as monitor, prioritize, or rebalance.

## Prediction Target

Initial target:

`stockout_next_60min`

- 1 = the station reaches zero available bikes at any point during the
  next 60 minutes
- 0 = the station does not reach zero available bikes during the next
  60 minutes

The exact definition may later be expanded to include near-stockout
conditions such as one bike remaining.

## Prediction Horizon

Initial modeling horizon:

60 minutes

Later experiments may compare:

- 30 minutes — urgent intervention
- 60 minutes — primary operating horizon
- 120 minutes — planning horizon

## Information Available at Prediction Time

The model may use only information that would actually be known at the
moment a prediction is generated.

Potential features include:

### Current station state

- bikes available
- docks available
- disabled bikes
- disabled docks
- station operating status
- station capacity

### Recent inventory behavior

Using 10-minute station snapshots:

- inventory 10 minutes ago
- inventory 20 minutes ago
- inventory 30 minutes ago
- inventory change over recent windows
- recent inventory trend

### Historical station behavior

Using historical trip data:

- typical departures by station and time
- typical arrivals by station and time
- historical net flow
- weekday versus weekend behavior
- same-time historical patterns
- station-specific demand patterns

### Context

Potential future features:

- hour of day
- day of week
- weekend indicator
- holidays
- weather
- weather forecast

## Data Leakage Rule

For a prediction made at 7:00 AM, the model may use information available
at or before 7:00 AM.

It must not use:

- inventory observations after 7:00 AM
- trips occurring after 7:00 AM
- future weather observations that were not known at 7:00 AM

When aligning inventory snapshots to a prediction timestamp, use the most
recent snapshot available at or before the prediction time rather than
rounding a future observation backward.

## Data Available

### Historical trip data

Bluebikes trip history currently includes:

- station IDs
- trip start station
- trip end station
- trip start time
- trip end time
- departures
- arrivals
- net flow
- hour
- 30-minute time slot
- day of week
- weekend indicator

Historical data currently covers approximately June through August 2026.

### Live station inventory

A cloud pipeline now collects Bluebikes GBFS station status approximately
every 10 minutes.

Architecture:

Bluebikes GBFS
→ AWS EventBridge Scheduler
→ AWS Lambda
→ Amazon S3

Each raw JSON snapshot contains station-level information including:

- station_id
- num_bikes_available
- num_docks_available
- num_bikes_disabled
- num_docks_disabled
- is_installed
- is_renting
- is_returning
- last_reported

These snapshots will become the historical inventory dataset used to
create true stockout labels.

## Evaluation

Accuracy alone will not be sufficient because stockouts are expected to
be relatively rare.

Initial evaluation should focus on:

- Recall — how many real stockouts were detected?
- Precision — how many stockout alerts were correct?
- Precision-recall tradeoff
- PR-AUC

Eventually the model should also be evaluated using an operational cost
framework:

Cost =
missed stockouts × stockout cost
+
unnecessary rebalancing actions × intervention cost

## Current Project Status

Completed:

- Historical trip ingestion
- Multi-month station flow dataset
- Time and station feature engineering
- Baseline forecasting experiments
- Decision tree experiment
- XGBoost experiment
- Exploratory analysis of station demand and net flow
- Identification of strong weekly station patterns
- Identification of missing historical inventory as the main data gap
- Local Bluebikes GBFS station-status collector
- AWS Lambda station-status collector
- Amazon S3 raw snapshot storage
- EventBridge Scheduler running every 10 minutes
- First successful cloud inventory snapshots

Current phase:

Collect historical inventory while defining the stockout prediction
dataset and modeling contract.