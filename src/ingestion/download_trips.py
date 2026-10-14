import requests
from pathlib import Path
from zipfile import ZipFile


# 1. MONTHS WE WANT TO DOWNLOAD

months = (
    [f"2023{month:02d}" for month in range(9, 13)]
    + [f"2024{month:02d}" for month in range(1, 13)]
    + [f"2025{month:02d}" for month in range(1, 13)]
    + [f"2026{month:02d}" for month in range(1, 9)]
)


# 2. LOCATION FOR RAW DATA

data_dir = Path("data/raw")
data_dir.mkdir(parents=True, exist_ok=True)


# 3. DOWNLOAD AND EXTRACT EACH MONTH

for month in months:

    zip_name = f"{month}-bluebikes-tripdata.zip"
    csv_name = f"{month}-bluebikes-tripdata.csv"

    url = f"https://s3.amazonaws.com/hubway-data/{zip_name}"

    zip_path = data_dir / zip_name
    csv_path = data_dir / csv_name


    # DOWNLOAD ZIP IF WE DO NOT ALREADY HAVE IT

    if not zip_path.exists():

        print(f"Downloading {month}...")

        response = requests.get(url)

        if response.status_code == 404:
            print(f"{month} is not available — skipping.")
            continue

        response.raise_for_status()

        with open(zip_path, "wb") as file:
            file.write(response.content)

        print(f"Downloaded {zip_name}")

    else:

        print(f"{zip_name} already exists")


    # EXTRACT CSV IF WE DO NOT ALREADY HAVE IT

    if not csv_path.exists():

        print(f"Extracting {month}...")

        with ZipFile(zip_path, "r") as zip_file:
            zip_file.extract(csv_name, data_dir)

        print(f"Extracted {csv_name}")

    else:

        print(f"{csv_name} already exists")