import pandas as pd
from pathlib import Path

print("Starting transform.py...")

# File paths
RAW_PATH = Path("data") / "sea_surface_sample.csv"
CLEAN_PATH = Path("data") / "clean" / "sea_surface_clean.csv"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("🧹 Cleaning data...")

    # Standardize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Convert data types
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Convert numeric columns (if any)
    for col in ["temperature", "salinity", "depth", "latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no useful data
    df.dropna(subset=["temperature", "salinity"], inplace=True)

    return df

def run():
    print(f"📂 Loading data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH)

    df_clean = clean_data(df)

    # Save to clean folder
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)

    print(f"✅ Cleaned data saved to {CLEAN_PATH}")
    print(df_clean.head())

if __name__ == "__main__":
    run()
