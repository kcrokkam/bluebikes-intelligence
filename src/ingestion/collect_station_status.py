import requests
import pandas as pd
from pathlib import Path


def collect_snapshot():

    # 1. FETCH LIVE STATION STATUS

    url = "https://gbfs.bluebikes.com/gbfs/en/station_status.json"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    stations = data["data"]["stations"]


    # 2. CONVERT TO DATAFRAME

    stations_df = pd.DataFrame(stations)


    # 3. ADD SNAPSHOT TIME

    snapshot_time = pd.Timestamp.now(tz="UTC")

    stations_df["snapshot_time"] = snapshot_time


    # 4. CREATE SNAPSHOT FOLDER

    snapshot_dir = Path("data/snapshots")

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # 5. CREATE FILE NAME

    file_name = snapshot_time.strftime(
        "station_status_%Y%m%d_%H%M%S.csv"
    )

    output_file = snapshot_dir / file_name


    # 6. SAVE SNAPSHOT

    stations_df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved snapshot to {output_file}"
    )

    return output_file


if __name__ == "__main__":

    collect_snapshot()