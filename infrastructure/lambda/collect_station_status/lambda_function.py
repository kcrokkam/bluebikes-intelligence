# 1. Imports
import json
import boto3
import urllib.request
from datetime import datetime, timezone


# 2. Configuration
BUCKET_NAME = "bluebikes-inventory-snapshots-kc"

STATION_STATUS_URL = (
    "https://gbfs.bluebikes.com/gbfs/en/station_status.json"
)

s3 = boto3.client("s3")


# 3. Lambda entry point
def lambda_handler(event, context):

    # Fetch the current Bluebikes station-status feed
    with urllib.request.urlopen(
        STATION_STATUS_URL,
        timeout=10,
    ) as response:
        raw_data = response.read()

    # Record when this snapshot was collected
    snapshot_time = datetime.now(timezone.utc)

    # Build the file name
    file_name = snapshot_time.strftime(
        "station_status_%Y%m%d_%H%M%S.json"
    )

    # Organize snapshots by date in S3
    key = (
        f"raw/station_status/"
        f"{snapshot_time:%Y/%m/%d}/"
        f"{file_name}"
    )

    # Save the original JSON response
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=raw_data,
        ContentType="application/json",
    )

    print(
        f"Saved snapshot to "
        f"s3://{BUCKET_NAME}/{key}"
    )

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "bucket": BUCKET_NAME,
                "key": key,
            }
        ),
    }