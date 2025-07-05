"""
Configuration module for Ocean Data Pipeline

This module contains all configuration settings for ERDDAP API integration,
coordinate conversion, and data processing parameters.
"""

from datetime import datetime
from typing import Tuple, Dict, Any
import math

# ERDDAP Dataset Configuration
ERDDAP_CONFIG = {
    "base_url": "https://erddap.ifremer.fr/erddap/griddap",
    "dataset_id": "SDC_NAT_CLIM_TS_V1_025_m",
    "variables": ["Temperature", "Salinity"],
    "timeout": 30,  # seconds
    "max_retries": 3,
    "retry_delay": 1.0,  # seconds
}

# Grid Configuration for SeaDataNet North Atlantic Climatology
GRID_CONFIG = {
    "latitude": {
        "min": 20.0,    # 20°N
        "max": 80.0,    # 80°N  
        "resolution": 0.25,  # 0.25° resolution
        "grid_size": 241,    # Number of grid points
    },
    "longitude": {
        "min": -80.0,   # 80°W
        "max": 40.0,    # 40°E
        "resolution": 0.25,  # 0.25° resolution
        "grid_size": 481,    # Number of grid points
    },
    "time": {
        "start": "1955-01-01",
        "end": "2015-12-31", 
        "resolution": "monthly",
        "grid_size": 732,  # 61 years * 12 months
    },
    "depth": {
        "surface_index": 106,  # Surface level index (0m depth)
        "available_depths": [0, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000]
    }
}

# API Rate Limiting
RATE_LIMIT_CONFIG = {
    "requests_per_minute": 30,
    "requests_per_hour": 100,
    "backoff_factor": 2.0,
    "max_wait_time": 60,  # seconds
}

# Cache Configuration
CACHE_CONFIG = {
    "enabled": True,
    "ttl_hours": 24,  # Time to live in hours
    "max_cache_size_mb": 100,
    "cleanup_interval_hours": 6,
}

# Data Quality Thresholds
QUALITY_CONFIG = {
    "min_data_points": 1,
    "max_missing_ratio": 0.5,  # 50% missing data is still acceptable
    "temperature_bounds": (-5.0, 35.0),  # °C
    "salinity_bounds": (0.0, 45.0),  # PSU
}

class CoordinateConverter:
    """Handles coordinate conversions between lat/lon and ERDDAP grid indices."""
    
    @staticmethod
    def lat_to_grid_index(latitude: float) -> int:
        """Convert latitude to ERDDAP grid index."""
        lat_config = GRID_CONFIG["latitude"]
        
        # Clamp to valid range
        lat_clamped = max(lat_config["min"], min(lat_config["max"], latitude))
        
        # Convert to grid index (grid starts from max latitude)
        grid_index = int((lat_config["max"] - lat_clamped) / lat_config["resolution"])
        
        # Ensure within bounds
        return max(0, min(lat_config["grid_size"] - 1, grid_index))
    
    @staticmethod
    def lon_to_grid_index(longitude: float) -> int:
        """Convert longitude to ERDDAP grid index."""
        lon_config = GRID_CONFIG["longitude"]
        
        # Clamp to valid range
        lon_clamped = max(lon_config["min"], min(lon_config["max"], longitude))
        
        # Convert to grid index (grid starts from min longitude)
        grid_index = int((lon_clamped - lon_config["min"]) / lon_config["resolution"])
        
        # Ensure within bounds
        return max(0, min(lon_config["grid_size"] - 1, grid_index))
    
    @staticmethod
    def grid_index_to_lat(grid_index: int) -> float:
        """Convert grid index back to latitude."""
        lat_config = GRID_CONFIG["latitude"]
        return lat_config["max"] - (grid_index * lat_config["resolution"])
    
    @staticmethod
    def grid_index_to_lon(grid_index: int) -> float:
        """Convert grid index back to longitude."""
        lon_config = GRID_CONFIG["longitude"]
        return lon_config["min"] + (grid_index * lon_config["resolution"])
    
    @staticmethod
    def date_to_time_index(date_str: str) -> int:
        """Convert date string to ERDDAP time index."""
        try:
            # Parse input date
            if isinstance(date_str, str):
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                date_obj = date_str
            
            # Get start date
            start_date = datetime.strptime(GRID_CONFIG["time"]["start"], "%Y-%m-%d")
            
            # Calculate months difference
            months_diff = (date_obj.year - start_date.year) * 12 + (date_obj.month - start_date.month)
            
            # Ensure within bounds
            return max(0, min(GRID_CONFIG["time"]["grid_size"] - 1, months_diff))
            
        except Exception:
            # Default to first time index if conversion fails
            return 0
    
    @staticmethod
    def time_index_to_date(time_index: int) -> str:
        """Convert time index back to date string."""
        start_date = datetime.strptime(GRID_CONFIG["time"]["start"], "%Y-%m-%d")
        
        # Add months to start date
        year = start_date.year + (time_index // 12)
        month = start_date.month + (time_index % 12)
        
        # Handle month overflow
        if month > 12:
            year += 1
            month -= 12
        
        return f"{year}-{month:02d}-01"
    
    @staticmethod
    def validate_coordinates(latitude: float, longitude: float) -> Tuple[bool, str]:
        """Validate if coordinates are within the dataset bounds."""
        lat_config = GRID_CONFIG["latitude"]
        lon_config = GRID_CONFIG["longitude"]
        
        if not (lat_config["min"] <= latitude <= lat_config["max"]):
            return False, f"Latitude {latitude} outside valid range [{lat_config['min']}, {lat_config['max']}]"
        
        if not (lon_config["min"] <= longitude <= lon_config["max"]):
            return False, f"Longitude {longitude} outside valid range [{lon_config['min']}, {lon_config['max']}]"
        
        return True, "Coordinates valid"
    
    @staticmethod
    def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, str]:
        """Validate if date range is within dataset bounds."""
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            
            dataset_start = datetime.strptime(GRID_CONFIG["time"]["start"], "%Y-%m-%d")
            dataset_end = datetime.strptime(GRID_CONFIG["time"]["end"], "%Y-%m-%d")
            
            if start_dt < dataset_start:
                return False, f"Start date {start_date} before dataset start {GRID_CONFIG['time']['start']}"
            
            if end_dt > dataset_end:
                return False, f"End date {end_date} after dataset end {GRID_CONFIG['time']['end']}"
            
            if start_dt > end_dt:
                return False, f"Start date {start_date} after end date {end_date}"
            
            # Check if time range is reasonable (not too large)
            months_diff = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
            if months_diff > 120:  # 10 years max
                return False, f"Date range too large: {months_diff} months (max 120 months)"
            
            return True, "Date range valid"
            
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"

class ERDDAPQueryBuilder:
    """Builds ERDDAP query URLs for dynamic data requests."""
    
    def __init__(self):
        self.converter = CoordinateConverter()
    
    def build_query_url(self, 
                       latitude: float, 
                       longitude: float, 
                       start_date: str, 
                       end_date: str,
                       variables: list = None) -> str:
        """Build complete ERDDAP query URL."""
        
        # Use default variables if none provided
        if variables is None:
            variables = ERDDAP_CONFIG["variables"]
        
        # Convert coordinates to grid indices
        lat_idx = self.converter.lat_to_grid_index(latitude)
        lon_idx = self.converter.lon_to_grid_index(longitude)
        
        # Convert dates to time indices
        start_time_idx = self.converter.date_to_time_index(start_date)
        end_time_idx = self.converter.date_to_time_index(end_date)
        
        # Get surface depth index
        depth_idx = GRID_CONFIG["depth"]["surface_index"]
        
        # Build variable queries
        variable_queries = []
        for var in variables:
            var_query = f"{var}[{start_time_idx}:{end_time_idx}][{depth_idx}][{lat_idx}][{lon_idx}]"
            variable_queries.append(var_query)
        
        # Combine into full URL
        query_params = ",".join(variable_queries)
        base_url = f"{ERDDAP_CONFIG['base_url']}/{ERDDAP_CONFIG['dataset_id']}.csv"
        
        return f"{base_url}?{query_params}"
    
    def get_actual_coordinates(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """Get the actual grid coordinates that will be used."""
        lat_idx = self.converter.lat_to_grid_index(latitude)
        lon_idx = self.converter.lon_to_grid_index(longitude)
        
        actual_lat = self.converter.grid_index_to_lat(lat_idx)
        actual_lon = self.converter.grid_index_to_lon(lon_idx)
        
        return actual_lat, actual_lon
    
    def get_query_metadata(self, 
                          latitude: float, 
                          longitude: float, 
                          start_date: str, 
                          end_date: str) -> Dict[str, Any]:
        """Get metadata about the query for caching and validation."""
        
        # Validate inputs
        coord_valid, coord_msg = self.converter.validate_coordinates(latitude, longitude)
        date_valid, date_msg = self.converter.validate_date_range(start_date, end_date)
        
        # Get actual coordinates
        actual_lat, actual_lon = self.get_actual_coordinates(latitude, longitude)
        
        # Calculate expected data points
        start_time_idx = self.converter.date_to_time_index(start_date)
        end_time_idx = self.converter.date_to_time_index(end_date)
        expected_points = end_time_idx - start_time_idx + 1
        
        return {
            "request": {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date,
                "end_date": end_date,
            },
            "actual": {
                "latitude": actual_lat,
                "longitude": actual_lon,
                "start_time_index": start_time_idx,
                "end_time_index": end_time_idx,
            },
            "validation": {
                "coordinates_valid": coord_valid,
                "coordinates_message": coord_msg,
                "date_range_valid": date_valid,
                "date_range_message": date_msg,
                "overall_valid": coord_valid and date_valid,
            },
            "expected_data_points": expected_points,
            "variables": ERDDAP_CONFIG["variables"],
        }

def get_coverage_bounds() -> Dict[str, float]:
    """Get the geographic coverage bounds of the dataset."""
    return {
        "north": GRID_CONFIG["latitude"]["max"],
        "south": GRID_CONFIG["latitude"]["min"],
        "east": GRID_CONFIG["longitude"]["max"],
        "west": GRID_CONFIG["longitude"]["min"],
    }

def get_time_bounds() -> Dict[str, str]:
    """Get the temporal coverage bounds of the dataset."""
    return {
        "start": GRID_CONFIG["time"]["start"],
        "end": GRID_CONFIG["time"]["end"],
    }

# Global instances for easy access
coordinate_converter = CoordinateConverter()
query_builder = ERDDAPQueryBuilder()