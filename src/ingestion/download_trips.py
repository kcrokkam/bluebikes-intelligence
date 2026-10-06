import requests
from pathlib import Path
from zipfile import ZipFile


# 1. MONTHS WE WANT TO DOWNLOAD

months = [
    "202606",
    "202607",
    "202608"
]


# 2. LOCATION FOR RAW DATA

data_dir = Path("data/raw")


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