import requests
import pandas as pd
from pathlib import Path

print("Starting extract.py...")

# Constants
# Dataset used is SeaDataNet North Atlantic Climatology (1955-2015) accessed via the Ifremer ERDDAP server

DATASET_ID = "SDC_NAT_CLIM_TS_V1_025_m"
BASE_URL = f"https://erddap.ifremer.fr/erddap/griddap/{DATASET_ID}.csv"
SAVE_PATH = Path("data") / "sea_surface_sample.csv"

# Time = January (0), Depth = Surface (106), Lat = ~38°N (90), Lon = ~-10°W (60)
# Depth index is reversed in this dataset so 6000m = 0, 0m = 106.

params_encoded = (
    "Temperature%5B0%5D%5B106%5D%5B90%5D%5B60%5D%2C"
    "Salinity%5B0%5D%5B106%5D%5B90%5D%5B60%5D"
)
url = f"{BASE_URL}?{params_encoded}"

def download_sea_surface_data():
    print("🔄 Downloading sea surface data from ERDDAP...")
    response = requests.get(url)
    response.raise_for_status()

    # Ensure data directory exists
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Save raw CSV
    with open(SAVE_PATH, 'wb') as f:
        f.write(response.content)

    print(f"✅ Data saved to {SAVE_PATH}")

    # Load into DataFrame
    df = pd.read_csv(SAVE_PATH)
    print(f"📊 Retrieved {len(df)} row(s):")
    print(df)

    return df

if __name__ == "__main__":
    df = download_sea_surface_data()
