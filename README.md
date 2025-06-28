# 🌊 Ocean Data Pipeline

A complete data engineering pipeline that extracts oceanographic data, processes it through an ETL system, and visualizes it with an interactive dashboard.

## Features

- **ETL Pipeline**: Extract data from NOAA/ERDDAP, clean and store in DuckDB
- **Interactive Dashboard**: Streamlit app with maps, time series, and data quality metrics
- **Real-time Queries**: SQL interface and data filtering
- **Error Handling**: Comprehensive logging and validation

## Tech Stack

- Python, Pandas, DuckDB
- Streamlit, Plotly, Folium
- ERDDAP API integration

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/ocean-data-pipeline.git
cd ocean-data-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Screenshots

### Dashboard Overview
<!-- Add dashboard screenshot here -->

### Interactive Map
<!-- Add map screenshot here -->

### Time Series Analysis
<!-- Add time series screenshot here -->

### Data Quality Metrics
<!-- Add data quality screenshot here -->

## Project Structure

```
ocean-data-pipeline/
├── data/                   # Data storage
├── pipeline/               # ETL modules
│   ├── extract.py         # Data extraction
│   ├── transform.py       # Data cleaning
│   └── load.py           # Database loading
├── dashboard/
│   └── app.py            # Streamlit dashboard
├── run_pipeline.py        # Main orchestrator
└── requirements.txt       # Dependencies
```

## Usage

```bash
# Run full pipeline
python run_pipeline.py

# Run individual steps
python run_pipeline.py --step extract
python run_pipeline.py --step transform
python run_pipeline.py --step load

# Validate data
python run_pipeline.py --validate-only
```

## Data Source

**SeaDataNet North Atlantic Climatology** via Ifremer ERDDAP  
Variables: Temperature, Salinity, Location, Time

## License

MIT License - see [LICENSE](LICENSE) file for details.