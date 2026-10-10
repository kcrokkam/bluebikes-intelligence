# Cloud Inventory Collection Pipeline

## Purpose

Bluebikes publishes current station inventory through its GBFS
`station_status` feed, but historical station inventory was not available
for the period needed by this project.

This pipeline creates that history by continuously storing raw snapshots.

## Architecture

Bluebikes GBFS station_status
→ EventBridge Scheduler
→ AWS Lambda
→ Amazon S3

## Schedule

The Lambda function runs every 10 minutes.

This produces approximately:

- 144 snapshots per day
- roughly 90,000 station observations per day at the current network size

## Storage

Raw GBFS responses are stored as JSON rather than being transformed
during ingestion.

Example structure:

raw/
  station_status/
    YYYY/
      MM/
        DD/
          station_status_YYYYMMDD_HHMMSS.json

Raw data is kept unchanged so downstream transformations and modeling can
be reproduced independently.

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

## Future Processing

A later transformation job will convert raw snapshots into station-time
observations and create labels such as:

`stockout_next_60min`

For each prediction timestamp t, features will use information at or
before t, while future inventory observations will be used only to
construct the training label.