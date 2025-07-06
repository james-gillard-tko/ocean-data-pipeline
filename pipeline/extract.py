"""
Enhanced extraction module with dynamic coordinate and time support.

This module now supports:
- Dynamic lat/lon coordinate selection
- Flexible date range queries
- Smart caching to reduce API calls
- Robust error handling and retries
- Rate limiting compliance
"""

import requests
import pandas as pd
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from datetime import datetime

# Import our new configuration and cache systems
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    ERDDAP_CONFIG, 
    RATE_LIMIT_CONFIG, 
    QUALITY_CONFIG,
    query_builder,
    coordinate_converter,
    get_coverage_bounds,
    get_time_bounds
)
from cache_manager import cache_manager

logger = logging.getLogger(__name__)

class ERDDAPExtractor:
    """Enhanced ERDDAP data extractor with dynamic coordinate support."""
    
    def __init__(self):
        self.last_request_time = 0
        self.request_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Ocean-Data-Pipeline/1.0 (Research Tool)'
        })
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        min_interval = 60.0 / RATE_LIMIT_CONFIG["requests_per_minute"]
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            logger.debug(f"⏱️ Rate limiting: sleeping for {sleep_time:.1f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _make_request_with_retry(self, url: str) -> requests.Response:
        """Make HTTP request with retry logic."""
        max_retries = ERDDAP_CONFIG["max_retries"]
        retry_delay = ERDDAP_CONFIG["retry_delay"]
        
        for attempt in range(max_retries):
            try:
                self._enforce_rate_limit()
                
                logger.debug(f"🌐 Making request to ERDDAP (attempt {attempt + 1}/{max_retries})")
                
                response = self.session.get(
                    url, 
                    timeout=ERDDAP_CONFIG["timeout"]
                )
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                
                wait_time = retry_delay * (RATE_LIMIT_CONFIG["backoff_factor"] ** attempt)
                logger.warning(f"⚠️ Request failed (attempt {attempt + 1}): {str(e)}")
                logger.info(f"⏱️ Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
    
    def fetch_data_for_location(self, 
                               latitude: float, 
                               longitude: float, 
                               start_date: str, 
                               end_date: str,
                               variables: List[str] = None,
                               use_cache: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fetch ocean data for a specific location and date range.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees  
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            variables: List of variables to fetch (default: Temperature, Salinity)
            use_cache: Whether to use caching (default: True)
            
        Returns:
            Tuple of (DataFrame with data, metadata dict)
        """
        
        # Use default variables if none provided
        if variables is None:
            variables = ERDDAP_CONFIG["variables"]
        
        logger.info(f"🔍 Fetching data for {latitude:.3f}°N, {longitude:.3f}°W from {start_date} to {end_date}")
        
        # Get query metadata and validation
        metadata = query_builder.get_query_metadata(latitude, longitude, start_date, end_date)
        
        # Check if query is valid
        if not metadata["validation"]["overall_valid"]:
            error_msg = f"Invalid query: {metadata['validation']['coordinates_message']}, {metadata['validation']['date_range_message']}"
            logger.error(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        # Show actual coordinates that will be used
        actual_lat = metadata["actual"]["latitude"]
        actual_lon = metadata["actual"]["longitude"]
        if abs(actual_lat - latitude) > 0.1 or abs(actual_lon - longitude) > 0.1:
            logger.info(f"📍 Snapped to nearest grid point: {actual_lat:.3f}°N, {actual_lon:.3f}°W")
        
        # Check cache first
        if use_cache:
            cached_data = cache_manager.get_cached_data(
                actual_lat, actual_lon, start_date, end_date, variables
            )
            if cached_data is not None:
                logger.info(f"⚡ Using cached data ({len(cached_data)} rows)")
                metadata["data_source"] = "cache"
                metadata["row_count"] = len(cached_data)
                return cached_data, metadata
        
        # Build query URL
        query_url = query_builder.build_query_url(latitude, longitude, start_date, end_date, variables)
        logger.info(f"🌐 Query URL: {query_url}")
        
        # Log the grid indices for debugging
        lat_idx = query_builder.converter.lat_to_grid_index(latitude)
        lon_idx = query_builder.converter.lon_to_grid_index(longitude)
        start_time_idx = query_builder.converter.date_to_time_index(start_date)
        end_time_idx = query_builder.converter.date_to_time_index(end_date)
        logger.info(f"📍 Grid indices: lat={lat_idx}, lon={lon_idx}, time={start_time_idx}:{end_time_idx}")
        
        # Make API request
        try:
            response = self._make_request_with_retry(query_url)
            
            # Parse response directly (don't stream for small datasets)
            import io
            df = pd.read_csv(io.StringIO(response.text), skipinitialspace=True)
            
            # Basic data validation
            if df.empty:
                logger.warning("⚠️ API returned empty dataset")
                metadata["data_source"] = "api"
                metadata["row_count"] = 0
                return pd.DataFrame(), metadata
            
            # Clean and validate data
            df_clean = self._clean_api_response(df)
            
            # Quality checks
            quality_info = self._validate_data_quality(df_clean)
            metadata.update(quality_info)
            
            # Cache the results
            if use_cache and not df_clean.empty:
                cache_manager.cache_data(actual_lat, actual_lon, start_date, end_date, variables, df_clean)
            
            logger.info(f"✅ Retrieved {len(df_clean)} rows from API")
            metadata["data_source"] = "api"
            metadata["row_count"] = len(df_clean)
            
            return df_clean, metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch data: {str(e)}")
            raise
    
    def _clean_api_response(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize API response data."""
        logger.debug("🧹 Cleaning API response data...")
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove units row if present (ERDDAP sometimes includes this)
        if len(df_clean) > 1 and df_clean.iloc[0, 0] in ['UTC', 'units']:
            df_clean = df_clean.iloc[1:].reset_index(drop=True)
        
        # Standardize column names
        df_clean.columns = [col.strip().lower().replace(' ', '_') for col in df_clean.columns]
        
        # Convert data types
        if 'time' in df_clean.columns:
            df_clean['time'] = pd.to_datetime(df_clean['time'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['temperature', 'salinity', 'depth', 'latitude', 'longitude']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove rows with all NaN values for key variables
        key_vars = ['temperature', 'salinity']
        available_key_vars = [col for col in key_vars if col in df_clean.columns]
        
        if available_key_vars:
            df_clean = df_clean.dropna(subset=available_key_vars, how='all')
        
        # Sort by time if available
        if 'time' in df_clean.columns:
            df_clean = df_clean.sort_values('time').reset_index(drop=True)
        
        return df_clean
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return metrics."""
        
        if df.empty:
            return {
                "quality_score": 0.0,
                "quality_issues": ["No data available"],
                "completeness": 0.0,
                "value_ranges": {}
            }
        
        quality_issues = []
        
        # Check minimum data points
        if len(df) < QUALITY_CONFIG["min_data_points"]:
            quality_issues.append(f"Insufficient data points: {len(df)} < {QUALITY_CONFIG['min_data_points']}")
        
        # Check completeness
        total_values = len(df) * len(df.columns)
        non_null_values = df.count().sum()
        completeness = non_null_values / total_values if total_values > 0 else 0
        
        if completeness < (1 - QUALITY_CONFIG["max_missing_ratio"]):
            quality_issues.append(f"High missing data ratio: {(1-completeness)*100:.1f}%")
        
        # Check value ranges
        value_ranges = {}
        
        if 'temperature' in df.columns:
            temp_min, temp_max = QUALITY_CONFIG["temperature_bounds"]
            temp_values = df['temperature'].dropna()
            if not temp_values.empty:
                value_ranges['temperature'] = {
                    'min': float(temp_values.min()),
                    'max': float(temp_values.max()),
                    'mean': float(temp_values.mean()),
                    'count': len(temp_values)
                }
                
                # Check for unrealistic values
                if temp_values.min() < temp_min or temp_values.max() > temp_max:
                    quality_issues.append(f"Temperature values outside expected range [{temp_min}, {temp_max}]")
        
        if 'salinity' in df.columns:
            sal_min, sal_max = QUALITY_CONFIG["salinity_bounds"]
            sal_values = df['salinity'].dropna()
            if not sal_values.empty:
                value_ranges['salinity'] = {
                    'min': float(sal_values.min()),
                    'max': float(sal_values.max()),
                    'mean': float(sal_values.mean()),
                    'count': len(sal_values)
                }
                
                # Check for unrealistic values
                if sal_values.min() < sal_min or sal_values.max() > sal_max:
                    quality_issues.append(f"Salinity values outside expected range [{sal_min}, {sal_max}]")
        
        # Calculate overall quality score
        quality_score = min(1.0, completeness * (1 - len(quality_issues) * 0.1))
        
        return {
            "quality_score": quality_score,
            "quality_issues": quality_issues,
            "completeness": completeness,
            "value_ranges": value_ranges
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the dataset coverage and capabilities."""
        coverage = get_coverage_bounds()
        time_bounds = get_time_bounds()
        
        return {
            "dataset_id": ERDDAP_CONFIG["dataset_id"],
            "variables": ERDDAP_CONFIG["variables"],
            "coverage": coverage,
            "time_range": time_bounds,
            "grid_resolution": {
                "latitude": 0.25,
                "longitude": 0.25,
                "temporal": "monthly"
            },
            "api_limits": {
                "requests_per_minute": RATE_LIMIT_CONFIG["requests_per_minute"],
                "max_time_range_months": 120
            }
        }
    
    def test_api_connection(self) -> Tuple[bool, str]:
        """Test API connection with a simple query."""
        try:
            # Test with a known good coordinate
            test_lat, test_lon = 40.0, -30.0
            test_start, test_end = "2000-01-01", "2000-02-01"
            
            logger.info("🔬 Testing API connection...")
            df, metadata = self.fetch_data_for_location(
                test_lat, test_lon, test_start, test_end, use_cache=False
            )
            
            if df.empty:
                return False, "API returned no data for test query"
            
            return True, f"API connection successful ({len(df)} test records retrieved)"
            
        except Exception as e:
            return False, f"API connection failed: {str(e)}"


# Backwards compatibility functions for existing pipeline
def download_sea_surface_data(save_path: Path = None) -> pd.DataFrame:
    """
    Legacy function for backwards compatibility.
    Downloads data using the original fixed coordinates.
    """
    logger.info("🔄 Using legacy download function...")
    
    # Use original coordinates
    latitude, longitude = 32.5, -70.0
    start_date, end_date = "1960-01-01", "1960-02-01"
    
    if save_path is None:
        save_path = Path("data") / "sea_surface_sample.csv"
    
    extractor = ERDDAPExtractor()
    df, metadata = extractor.fetch_data_for_location(latitude, longitude, start_date, end_date)
    
    # Save to file for pipeline compatibility
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    
    logger.info(f"✅ Data saved to {save_path}")
    logger.info(f"📊 Retrieved {len(df)} row(s)")
    
    return df


# New main function for testing
def main():
    """Main function for testing the enhanced extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ocean Data Extractor")
    parser.add_argument("--lat", type=float, default=40.0, help="Latitude")
    parser.add_argument("--lon", type=float, default=-30.0, help="Longitude")
    parser.add_argument("--start", default="2000-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2000-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--test-api", action="store_true", help="Test API connection")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    extractor = ERDDAPExtractor()
    
    if args.test_api:
        success, message = extractor.test_api_connection()
        print(f"API Test: {'✅ PASS' if success else '❌ FAIL'} - {message}")
        return
    
    try:
        # Fetch data
        df, metadata = extractor.fetch_data_for_location(
            args.lat, args.lon, args.start, args.end, 
            use_cache=not args.no_cache
        )
        
        print(f"\n📊 Results:")
        print(f"Rows: {len(df)}")
        print(f"Data source: {metadata['data_source']}")
        print(f"Quality score: {metadata['quality_score']:.2f}")
        print(f"Completeness: {metadata['completeness']:.1%}")
        
        if not df.empty:
            print(f"\nData preview:")
            print(df.head())
        
        # Show cache stats
        cache_stats = cache_manager.get_cache_stats()
        print(f"\n💾 Cache stats: {cache_stats['active_entries']} entries, {cache_stats['total_size_mb']:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()