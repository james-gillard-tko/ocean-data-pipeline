#!/usr/bin/env python3
"""
Test the ERDDAP fix with the exact coordinates that were failing
"""

import requests
from config import query_builder, coordinate_converter

def test_problem_coordinates():
    """Test with coordinates that were causing the 400 error."""
    
    # Use coordinates similar to what was failing
    test_lat, test_lon = 40.0, -30.0
    start_date, end_date = "1980-01-01", "1985-12-01"
    
    print(f"🧪 Testing coordinates that were failing:")
    print(f"Coordinates: {test_lat}°N, {test_lon}°W")
    print(f"Date range: {start_date} to {end_date}")
    
    # Get grid indices
    lat_idx = coordinate_converter.lat_to_grid_index(test_lat)
    lon_idx = coordinate_converter.lon_to_grid_index(test_lon)
    start_time_idx = coordinate_converter.date_to_time_index(start_date)
    end_time_idx = coordinate_converter.date_to_time_index(end_date)
    
    print(f"Grid indices: lat={lat_idx}, lon={lon_idx}, time={start_time_idx}:{end_time_idx}")
    
    # Build URL
    url = query_builder.build_query_url(test_lat, test_lon, start_date, end_date)
    print(f"Generated URL: {url}")
    
    # Test the URL
    print(f"\n🌐 Testing URL...")
    try:
        # Use a smaller request first
        response = requests.head(url, timeout=15)
        print(f"HEAD request status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ URL is valid! Now testing data download...")
            
            # Try downloading just a few bytes to verify data format
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Read first few lines
            lines = []
            for i, line in enumerate(response.iter_lines(decode_unicode=True)):
                lines.append(line)
                if i >= 3:  # Just get first few lines
                    break
            
            print("✅ Data download successful!")
            print("Sample data:")
            for line in lines:
                print(f"  {line}")
                
        else:
            print(f"❌ URL failed with status {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_small_date_range():
    """Test with a smaller date range to ensure it works."""
    
    test_lat, test_lon = 40.0, -30.0
    start_date, end_date = "1980-01-01", "1980-03-01"  # Just 2 months
    
    print(f"\n🧪 Testing with smaller date range:")
    print(f"Coordinates: {test_lat}°N, {test_lon}°W")
    print(f"Date range: {start_date} to {end_date}")
    
    url = query_builder.build_query_url(test_lat, test_lon, start_date, end_date)
    print(f"Generated URL: {url}")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        
        print("✅ Small date range works!")
        
        # Parse the data
        import pandas as pd
        import io
        
        df = pd.read_csv(io.StringIO(response.text))
        print(f"Retrieved {len(df)} rows")
        print("Sample data:")
        print(df.head())
        
    except Exception as e:
        print(f"❌ Small date range failed: {e}")

def main():
    """Run the test."""
    print("🔧 Testing ERDDAP Fix")
    print("=" * 40)
    
    test_problem_coordinates()
    test_small_date_range()
    
    print(f"\n🎯 If both tests pass, the dashboard should now work!")

if __name__ == "__main__":
    main()