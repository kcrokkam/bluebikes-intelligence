import json
from datetime import datetime, timezone
from pathlib import Path

import requests


STATION_STATUS_URL = (
    "https://gbfs.bluebikes.com/gbfs/en/station_status.json"
)


def collect_snapshot():
    """Collect one raw Bluebikes station-status JSON snapshot."""

    # 1. Fetch the live GBFS station-status feed
    response = requests.get(
        STATION_STATUS_URL,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    # 2. Record when the snapshot was collected
    snapshot_time = datetime.now(timezone.utc)

    # 3. Create the local snapshot folder
    snapshot_dir = Path("data/snapshots")
    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 4. Use the same JSON naming convention as the AWS pipeline
    file_name = snapshot_time.strftime(
        "station_status_%Y%m%d_%H%M%S.json"
    )

    output_file = snapshot_dir / file_name

    # 5. Preserve the raw GBFS JSON structure expected downstream
    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(data, file)

    print(
        f"Saved snapshot to {output_file}"
    )

    return output_file


if __name__ == "__main__":
    collect_snapshot()
