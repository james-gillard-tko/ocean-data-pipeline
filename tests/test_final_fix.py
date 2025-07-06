#!/usr/bin/env python3
"""
Test the final fix with correct dataset bounds
"""

import requests
from config import coordinate_converter, query_builder

def test_correct_bounds():
    """Test with the correct dataset bounds."""
    
    print("🧪 Testing with correct dataset bounds (1955-1960)...")
    
    # Test coordinates and dates that should work
    test_cases = [
        (32.5, -70.0, "1955-01-01", "1955-12-01", "Original working case"),
        (32.5, -70.0, "1960-01-01", "1960-12-01", "End of dataset"),
        (32.5, -70.0, "1957-01-01", "1958-12-01", "Mid-range"),
        (40.0, -30.0, "1955-01-01", "1955-12-01", "Different coordinates"),
    ]
    
    for lat, lon, start_date, end_date, description in test_cases:
        print(f"\n{description}:")
        print(f"  Coordinates: {lat}°N, {lon}°W")
        print(f"  Date range: {start_date} to {end_date}")
        
        # Get grid indices
        lat_idx = coordinate_converter.lat_to_grid_index(lat)
        lon_idx = coordinate_converter.lon_to_grid_index(lon)
        start_time_idx = coordinate_converter.date_to_time_index(start_date)
        end_time_idx = coordinate_converter.date_to_time_index(end_date)
        
        print(f"  Grid indices: lat={lat_idx}, lon={lon_idx}, time={start_time_idx}:{end_time_idx}")
        
        # Validate
        coord_valid, coord_msg = coordinate_converter.validate_coordinates(lat, lon)
        date_valid, date_msg = coordinate_converter.validate_date_range(start_date, end_date)
        
        print(f"  Coordinate validation: {coord_valid} - {coord_msg}")
        print(f"  Date validation: {date_valid} - {date_msg}")
        
        if coord_valid and date_valid:
            # Test URL
            url = query_builder.build_query_url(lat, lon, start_date, end_date)
            print(f"  URL: {url}")
            
            try:
                response = requests.head(url, timeout=10)
                status = "✅" if response.status_code == 200 else "❌"
                print(f"  Result: {response.status_code} {status}")
                
                if response.status_code == 200:
                    # Try downloading a bit of data
                    response = requests.get(url, timeout=15)
                    response.raise_for_status()
                    
                    lines = response.text.split('\n')[:5]
                    print(f"  Sample data: {len(lines)} lines retrieved")
                    
            except Exception as e:
                print(f"  Error: {e}")
        else:
            print(f"  Skipped due to validation error")

def test_dashboard_defaults():
    """Test the default values that will be used in the dashboard."""
    
    print(f"\n🧪 Testing dashboard defaults...")
    
    # Default coordinates (original working)
    lat, lon = 32.5, -70.0
    # Default date range (full dataset)
    start_date, end_date = "1955-01-01", "1960-12-31"
    
    print(f"Dashboard defaults:")
    print(f"  Coordinates: {lat}°N, {lon}°W")
    print(f"  Date range: {start_date} to {end_date}")
    
    # Test this combination
    url = query_builder.build_query_url(lat, lon, start_date, end_date)
    print(f"  URL: {url}")
    
    try:
        response = requests.head(url, timeout=10)
        if response.status_code == 200:
            print("  ✅ Dashboard defaults work!")
            
            # Get some actual data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Count data points
            lines = response.text.strip().split('\n')
            data_lines = [line for line in lines if line and not line.startswith('time')]
            
            print(f"  📊 Retrieved {len(data_lines)} data points")
            print(f"  🕐 Covers full 6-year dataset")
            
            return True
            
        else:
            print(f"  ❌ Failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def test_coordinate_range():
    """Test what happens with different coordinates."""
    
    print(f"\n🧪 Testing coordinate handling...")
    
    # Test various coordinates to see how they map
    test_coords = [
        (32.5, -70.0, "Original"),
        (40.0, -30.0, "Far from original"),
        (25.0, -75.0, "Different region"),
        (50.0, -60.0, "North of original"),
    ]
    
    for lat, lon, description in test_coords:
        lat_idx = coordinate_converter.lat_to_grid_index(lat)
        lon_idx = coordinate_converter.lon_to_grid_index(lon)
        
        # Convert back to see actual coordinates
        actual_lat = coordinate_converter.grid_index_to_lat(lat_idx)
        actual_lon = coordinate_converter.grid_index_to_lon(lon_idx)
        
        print(f"  {description}: {lat}°N, {lon}°W → grid[{lat_idx}, {lon_idx}] → {actual_lat:.1f}°N, {actual_lon:.1f}°W")

def main():
    """Test the final fix."""
    print("🔧 Testing Final Fix - Correct Dataset Bounds")
    print("=" * 50)
    
    test_correct_bounds()
    
    success = test_dashboard_defaults()
    
    test_coordinate_range()
    
    if success:
        print(f"\n🎉 Final fix successful!")
        print(f"✅ Dashboard should now work with:")
        print(f"   📍 Coordinates around 32.5°N, -70°W")
        print(f"   📅 Dates between 1955-1960")
        print(f"🚀 Launch: streamlit run dashboard/app.py")
    else:
        print(f"\n❌ Final fix still has issues")

if __name__ == "__main__":
    main()