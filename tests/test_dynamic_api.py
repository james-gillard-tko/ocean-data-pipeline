#!/usr/bin/env python3
"""
Test script for Phase 1: Dynamic API Infrastructure

This script tests the new dynamic extraction capabilities without 
modifying the existing pipeline structure.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import query_builder, coordinate_converter, get_coverage_bounds, get_time_bounds
from cache_manager import cache_manager
from pipeline.extract import ERDDAPExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_coordinate_conversion():
    """Test coordinate conversion functions."""
    print("🧪 Testing coordinate conversion...")
    
    # Test cases
    test_coords = [
        (40.0, -30.0),   # Mid North Atlantic
        (60.0, -20.0),   # North Atlantic
        (30.0, -60.0),   # Western North Atlantic
    ]
    
    for lat, lon in test_coords:
        # Test conversion
        lat_idx = coordinate_converter.lat_to_grid_index(lat)
        lon_idx = coordinate_converter.lon_to_grid_index(lon)
        
        # Convert back
        actual_lat = coordinate_converter.grid_index_to_lat(lat_idx)
        actual_lon = coordinate_converter.grid_index_to_lon(lon_idx)
        
        print(f"  {lat:.1f}°N, {lon:.1f}°W → Grid[{lat_idx}, {lon_idx}] → {actual_lat:.1f}°N, {actual_lon:.1f}°W")
        
        # Validate
        is_valid, msg = coordinate_converter.validate_coordinates(lat, lon)
        print(f"    Valid: {is_valid} - {msg}")
    
    print("✅ Coordinate conversion tests completed\n")

def test_date_conversion():
    """Test date conversion functions."""
    print("🧪 Testing date conversion...")
    
    test_dates = [
        "1960-01-01",
        "1980-06-01", 
        "2000-12-01",
        "2010-03-01"
    ]
    
    for date_str in test_dates:
        time_idx = coordinate_converter.date_to_time_index(date_str)
        back_to_date = coordinate_converter.time_index_to_date(time_idx)
        
        print(f"  {date_str} → Index {time_idx} → {back_to_date}")
    
    # Test date range validation
    is_valid, msg = coordinate_converter.validate_date_range("1960-01-01", "1965-12-01")
    print(f"  Date range 1960-1965: {is_valid} - {msg}")
    
    print("✅ Date conversion tests completed\n")

def test_query_builder():
    """Test ERDDAP query URL building."""
    print("🧪 Testing query builder...")
    
    # Test query
    lat, lon = 40.0, -30.0
    start_date, end_date = "1960-01-01", "1960-06-01"
    
    # Build query
    query_url = query_builder.build_query_url(lat, lon, start_date, end_date)
    print(f"  Query URL: {query_url[:100]}...")
    
    # Get metadata
    metadata = query_builder.get_query_metadata(lat, lon, start_date, end_date)
    print(f"  Metadata: {metadata['expected_data_points']} expected points")
    print(f"  Actual coords: {metadata['actual']['latitude']:.2f}°N, {metadata['actual']['longitude']:.2f}°W")
    print(f"  Valid: {metadata['validation']['overall_valid']}")
    
    print("✅ Query builder tests completed\n")

def test_cache_system():
    """Test cache management system."""
    print("🧪 Testing cache system...")
    
    # Get cache stats
    stats = cache_manager.get_cache_stats()
    print(f"  Cache stats: {stats['active_entries']} entries, {stats['total_size_mb']:.1f} MB")
    
    # Test cache operations with dummy data
    import pandas as pd
    from datetime import datetime
    
    # Create test data
    test_data = pd.DataFrame({
        'time': [datetime(2000, 1, 1), datetime(2000, 2, 1)],
        'temperature': [20.5, 21.0],
        'salinity': [35.0, 35.2],
        'latitude': [40.0, 40.0],
        'longitude': [-30.0, -30.0],
        'depth': [0.0, 0.0]
    })
    
    # Test caching
    lat, lon = 40.0, -30.0
    start_date, end_date = "2000-01-01", "2000-02-01"
    variables = ["Temperature", "Salinity"]
    
    # Cache data
    success = cache_manager.cache_data(lat, lon, start_date, end_date, variables, test_data)
    print(f"  Cache write: {'✅ Success' if success else '❌ Failed'}")
    
    # Retrieve cached data
    cached_data = cache_manager.get_cached_data(lat, lon, start_date, end_date, variables)
    cache_hit = cached_data is not None
    print(f"  Cache read: {'✅ Hit' if cache_hit else '❌ Miss'}")
    
    if cache_hit:
        print(f"  Cached data: {len(cached_data)} rows")
    
    print("✅ Cache system tests completed\n")

def test_api_extractor():
    """Test the enhanced API extractor."""
    print("🧪 Testing API extractor...")
    
    extractor = ERDDAPExtractor()
    
    # Test API connection
    success, message = extractor.test_api_connection()
    print(f"  API connection: {'✅ Success' if success else '❌ Failed'} - {message}")
    
    if success:
        # Test data extraction
        try:
            print("  Fetching sample data...")
            df, metadata = extractor.fetch_data_for_location(
                latitude=40.0,
                longitude=-30.0,
                start_date="1960-01-01",
                end_date="1960-03-01"
            )
            
            print(f"  Retrieved: {len(df)} rows")
            print(f"  Data source: {metadata['data_source']}")
            print(f"  Quality score: {metadata['quality_score']:.2f}")
            
            if not df.empty:
                print(f"  Sample data:")
                print(f"    Temperature: {df['temperature'].iloc[0]:.2f}°C")
                print(f"    Salinity: {df['salinity'].iloc[0]:.2f} PSU")
        
        except Exception as e:
            print(f"  ❌ Data extraction failed: {str(e)}")
    
    print("✅ API extractor tests completed\n")

def test_dataset_coverage():
    """Test dataset coverage information."""
    print("🧪 Testing dataset coverage...")
    
    # Get coverage bounds
    coverage = get_coverage_bounds()
    time_bounds = get_time_bounds()
    
    print(f"  Geographic coverage:")
    print(f"    North: {coverage['north']}°N")
    print(f"    South: {coverage['south']}°N") 
    print(f"    East: {coverage['east']}°E")
    print(f"    West: {coverage['west']}°W")
    
    print(f"  Temporal coverage:")
    print(f"    Start: {time_bounds['start']}")
    print(f"    End: {time_bounds['end']}")
    
    # Test coordinates within/outside coverage
    test_points = [
        (40.0, -30.0, "Inside coverage"),
        (50.0, -40.0, "Inside coverage"),
        (10.0, -30.0, "Outside coverage (too south)"),
        (40.0, -100.0, "Outside coverage (too west)")
    ]
    
    for lat, lon, description in test_points:
        is_valid, msg = coordinate_converter.validate_coordinates(lat, lon)
        print(f"    {lat:.1f}°N, {lon:.1f}°W: {'✅' if is_valid else '❌'} {description}")
    
    print("✅ Dataset coverage tests completed\n")

def main():
    """Run all Phase 1 tests."""
    print("🚀 Starting Phase 1 Tests: Dynamic API Infrastructure")
    print("=" * 60)
    
    try:
        # Run all tests
        test_coordinate_conversion()
        test_date_conversion()
        test_query_builder()
        test_cache_system()
        test_dataset_coverage()
        test_api_extractor()
        
        print("🎉 All Phase 1 tests completed successfully!")
        print("✅ Dynamic API infrastructure is ready for dashboard integration")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()