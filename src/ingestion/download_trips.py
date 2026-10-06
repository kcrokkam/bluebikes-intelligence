import requests
from pathlib import Path


url = "https://s3.amazonaws.com/hubway-data/202608-bluebikes-tripdata.zip"

output_dir = Path("data/raw")


output_file = output_dir / "202608-bluebikes-tripdata.zip"

response = requests.get(url)

with open(output_file, "wb") as file:
    file.write(response.content)

print(f"Saved trip data to {output_file}")