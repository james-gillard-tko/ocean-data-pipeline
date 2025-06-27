import duckdb
import pandas as pd
from pathlib import Path

print("Starting load.py...")

# File paths
CLEAN_PATH = Path("data") / "clean" / "sea_surface_clean.csv"
DB_PATH = Path("data") / "ocean_data.duckdb"

def load_to_duckdb():
    print(f"🔄 Loading data from {CLEAN_PATH} into {DB_PATH}...")

    # Load cleaned CSV
    df = pd.read_csv(CLEAN_PATH)

    # Connect to DuckDB (creates file if needed)
    conn = duckdb.connect(str(DB_PATH))

    # Create table and insert data
    conn.execute("CREATE TABLE IF NOT EXISTS sea_surface AS SELECT * FROM df")

    # Close connection
    conn.close()
    print("✅ Data loaded successfully.")

if __name__ == "__main__":
    load_to_duckdb()
