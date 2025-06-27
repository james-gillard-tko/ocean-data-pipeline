# Ocean Data Pipeline - Sensor Data Aquisition to Streamlit Viz

This project demonstrates a simple, real-world data engineering pipeline built with Python to ingest, clean, store, and visualize marine sensor data (e.g. from NOAA, EMODnet). Designed to support research and monitoring.

## Features
- Pulls real-time or historical oceanographic data via API
- Cleans and standardizes the dataset (timestamp, units, missing values)
- Stores processed data in SQLite
- Displays interactive visualizations in a Streamlit dashboard

## Tech Stack
- Python, Pandas, Requests
- DuckDB
- Streamlit
- Git + GitHub for version control

## License

This project is licensed under the MIT License – see the [LICENSE] file for details.

## Example Dashboard

*(Include screenshot once built)*

## How to Run

```bash
# Clone and enter
git clone https://github.com/yourusername/ocean-data-pipeline.git
cd ocean-data-pipeline

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the pipeline
python run_pipeline.py

# Launch the dashboard
streamlit run dashboard/app.py

