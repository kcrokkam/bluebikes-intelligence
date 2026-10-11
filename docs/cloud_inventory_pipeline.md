# Cloud Inventory Collection Pipeline

## Purpose

Bluebikes publishes current station inventory through its GBFS `station_status` feed, but historical station inventory was not available for the period needed by this project.

This pipeline creates that history by continuously storing raw snapshots.

## Architecture

```text
Bluebikes GBFS station_status
        ↓
EventBridge Scheduler
        ↓
AWS Lambda
        ↓
Amazon S3
```

## Schedule

The Lambda function runs every 10 minutes.

This produces approximately:

- 144 snapshots per day
- roughly 90,000 station observations per day at the current network size

## Storage

Raw GBFS responses are stored as JSON rather than transformed during ingestion.

Example structure:

```text
raw/
└── station_status/
    └── YYYY/
        └── MM/
            └── DD/
                └── station_status_YYYYMMDD_HHMMSS.json
```

Raw responses are preserved so downstream transformations and modeling can be reproduced independently.

## AWS Components

Lambda function:

`collect-bluebikes-station-status`

S3 bucket:

`bluebikes-inventory-snapshots-kc`

Region:

`us-east-2`

## Permissions

The Lambda execution role has:

- standard CloudWatch logging permissions
- `s3:PutObject` permission for the inventory snapshot bucket

EventBridge Scheduler has permission to invoke the Lambda function.

No AWS credentials are stored in this repository.

## Local Collection

The local collector at:

`src/ingestion/collect_station_status.py`

now writes the same raw JSON snapshot format expected by the downstream inventory pipeline.

Local snapshots are written to:

`data/snapshots/`

This keeps local development aligned with the AWS collection format.

## Downstream Processing

The raw JSON snapshots are consumed by:

`src/data_pipeline/inventory.py`

The pipeline:

1. loads and cleans station-status snapshots
2. orders observations by station and timestamp
3. creates inventory movement features
4. constructs the `stockout_next_60min` target

For each prediction timestamp `t`:

- features use information available at or before `t`
- future inventory observations are used only to construct the label

The resulting model-ready dataset is prepared in:

`src/data_pipeline/model_data.py`

and used by the baseline evaluation and live station risk-ranking scripts.
